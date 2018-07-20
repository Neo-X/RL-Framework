


# import cPickle
import dill
import sys
import gc
# from theano.compile.io import Out
sys.setrecursionlimit(50000)
# from sim.PendulumEnvState import PendulumEnvState
# from sim.PendulumEnv import PendulumEnv
from multiprocessing import Process, Queue
# from pathos.multiprocessing import Pool
import threading
import time
import copy
import numpy as np
from model.ModelUtil import *
# import memory_profiler
# import resources

from simulation.simEpoch import simModelParrallel

        
# @profile(precision=5)
def collectExperience(actor, exp_val, model, settings, sim_work_queues=None, 
                      eval_episode_data_queue=None):
    from util.ExperienceMemory import ExperienceMemory
    
    ## Easy hack to fix issue with training for MBAE needing a LearningAgent with forward dyanmics model and not just algorithm
    settings = copy.deepcopy(settings)
    settings['use_model_based_action_optimization'] = False
    if (settings["exploration_method"] == "sampling"):
        settings['exploration_method'] = "gaussian_network"
    action_selection = range(len(settings["discrete_actions"]))
    print ("Action selection: " + str(action_selection))
    # state_bounds = np.array(settings['state_bounds'])
    # state_bounds = np.array([[0],[0]])
    reward_bounds=np.array(settings["reward_bounds"])
    action_bounds = np.array(settings["action_bounds"], dtype=float)
    state_bounds = np.array(settings['state_bounds'], dtype=float)
    
    if (settings["bootsrap_with_discrete_policy"]) and (settings['bootstrap_samples'] > 0):
        print ("Collecting bootstrap samples from simulation")
        (states, actions, resultStates, rewards_, falls_, G_ts_, exp_actions, advantage_) = collectExperienceActionsContinuous(actor, exp_val, model, settings['bootstrap_samples'], settings=settings, action_selection=action_selection, sim_work_queues=sim_work_queues, 
                                                                                                                   eval_episode_data_queue=eval_episode_data_queue)
        # states = np.array(states)
        # states = np.append(states, state_bounds,0) # Adding that already specified bounds will ensure the final calculated is beyond these
        print (" Shape states: ", states.shape)
        print (" Shape Actions: ", actions.shape)
        print (" Shape result states: ", resultStates.shape)
        print (" Shape rewards_: ", rewards_.shape)
        print (" Shape falls: ", falls_.shape)
        print (" Shape G_ts_: ", G_ts_.shape)
        print (" Shape advantage: ", advantage_.shape)
        print (" Shape exp_actions: ", exp_actions.shape)
        
        scale_factor = 1.0
        
        state_bounds = np.ones((2,states.shape[1]))
        
        if (settings['state_normalization'] == "minmax"):
            state_bounds[0] = np.min(states[:settings['bootstrap_samples']], axis=0)
            state_bounds[1] = np.max(states[:settings['bootstrap_samples']], axis=0)
            reward_bounds[0] = np.min(rewards_[:settings['bootstrap_samples']], axis=0)
            reward_bounds[1] = np.max(rewards_[:settings['bootstrap_samples']], axis=0)
            # action_bounds[0] = np.min(actions[:settings['bootstrap_samples']], axis=0)
            # action_bounds[1] = np.max(actions[:settings['bootstrap_samples']], axis=0)
        elif (settings['state_normalization'] == "variance" or
              settings['state_normalization'] == "adaptive"):
            state_avg = np.mean(states[:settings['bootstrap_samples']], axis=0)
            state_stddev = np.std(states[:settings['bootstrap_samples']], axis=0)
            reward_avg = np.mean(rewards_[:settings['bootstrap_samples']], axis=0)
            reward_stddev = np.std(rewards_[:settings['bootstrap_samples']], axis=0)
            action_avg = np.mean(actions[:settings['bootstrap_samples']], axis=0)
            action_stddev = np.std(actions[:settings['bootstrap_samples']], axis=0)
            print("Computed state min bound: ", state_avg - state_stddev)
            print("Computed state max bound: ", state_avg + state_stddev)
            print ("(state_avg - (state_stddev * ", scale_factor, ")): ", (state_avg - (state_stddev * scale_factor)))
            state_bounds[0] = (state_avg - (state_stddev * scale_factor))
            state_bounds[1] = (state_avg + (state_stddev * scale_factor))
        elif (settings['state_normalization'] == "given"):
            # pass # Use bound specified in file
            state_bounds = np.array(settings['state_bounds'], dtype=float)
        else:
            print ("State scaling strategy unknown: ", (settings['state_normalization']))
            assert False
            
        ## Cast data to the proper type
        state_bounds = np.array(state_bounds, dtype=settings['float_type'])
        reward_bounds = np.array(reward_bounds, dtype=settings['float_type'])
        action_bounds = np.array(action_bounds, dtype=settings['float_type'])
            
        if settings['action_space_continuous']:
            experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), settings['expereince_length'], 
                                          continuous_actions=True, settings = settings, 
                                          # result_state_length=settings["dense_state_size"]
                                          )
        else:
            experience = ExperienceMemory(len(state_bounds[0]), 1, settings['expereince_length'])
        experience.setSettings(settings)
        
        # print ("State Mean:" + str(state_avg))
        # print ("State Variance: " + str(state_stddev))
        # print ("Reward Mean:" + str(reward_avg))
        # print ("Reward Variance: " + str(reward_stddev))
        # print ("Action Mean:" + str(action_avg))
        # print ("Action Variance: " + str(action_stddev))
        print ("State Length:" + str(len(state_bounds[1])))
        print ("Max State:" + str(state_bounds[1]))
        print ("Min State:" + str(state_bounds[0]))
        print ("Max Reward:" + str(reward_bounds[1]))
        print ("Min Reward:" + str(reward_bounds[0]))
        print ("Max Action:" + str(action_bounds[1]))
        print ("Min Action:" + str(action_bounds[0]))
        
        experience.setStateBounds(state_bounds)
        experience.setRewardBounds(reward_bounds)
        experience.setActionBounds(action_bounds)
        
        for state, action, resultState, reward_, fall_, G_t, exp_action, adv in zip(states, actions, resultStates, rewards_, falls_, G_ts_, exp_actions, advantage_):
            # if reward_ > settings['reward_lower_bound']: # Skip if reward gets too bad, skips nan too?
            if ("use_dual_state_representations" in settings
                and (settings["use_dual_state_representations"] == True)):
                if ("use_viz_for_policy" in settings 
                    and settings["use_viz_for_policy"] == True):
                    state = state[1][0]
                    resultState = resultState[0][0]
                    ### Testing to create data for fd learning
                    # resultState = resultState[0]
                else:
                    state = state[0]
                    resultState = resultState[0]
            if settings['action_space_continuous']:
                # experience.insert(norm_state(state, state_bounds), norm_action(action, action_bounds), norm_state(resultState, state_bounds), norm_reward([reward_], reward_bounds))
                experience.insertTuple(([state], [action], [resultState], [reward_], [fall_], [G_t], [exp_action], [adv]))
            else:
                experience.insertTuple(([state], [action], [resultState], [reward_], [falls_], G_t, [exp_action], [adv]))
            # else:
                # print ("Tuple with reward: " + str(reward_) + " skipped")
        # sys.exit()
    else: ## Most likely performing continuation learning
        print ("Skipping bootstrap samples from simulation")
        print ("State length: ", len(model.getStateBounds()[0]))
        if settings['action_space_continuous']:
            experience = ExperienceMemory(len(model.getStateBounds()[0]), len(model.getActionBounds()[0]), settings['expereince_length'], continuous_actions=True, settings = settings)
        else:
            experience = ExperienceMemory(len(model.getStateBounds()[0]), 1, settings['expereince_length'])
            experience.setSettings(settings)
            experience.setStateBounds(model.getStateBounds())
            experience.setRewardBounds(model.getRewardBounds())
            experience.setActionBounds(model.getActionBounds())
        
        """
        (states, actions, resultStates, rewards_) = collectExperienceActionsContinuous(exp, settings['expereince_length'], settings=settings, action_selection=action_selection)
        # states = np.array(states)
        state_bounds[0] = states.min(0)
        state_bounds[1] = states.max(0)
        reward_bounds[0][0] = rewards_.min(0)
        print ("Max State:" + str(state_bounds[1]))
        print ("Min State:" + str(state_bounds[0]))
        print ("Min Reward:" + str(reward_bounds[0]))
        """
        
        
    return  experience, state_bounds, reward_bounds, action_bounds

# @profile(precision=5)
def collectExperienceActionsContinuous(actor, exp, model, samples, settings, action_selection, sim_work_queues=None, eval_episode_data_queue=None):
    i = 0
    states = []
    actions = []
    resultStates = []
    rewards = []
    falls = []
    G_ts = []
    advantage = []
    exp_actions = []
    # anchor_data_file = open(settings["anchor_file"])
    # _anchors = getAnchors(anchor_data_file)
    # print ("Length of anchors epochs: " + str(len(_anchors)))
    # anchor_data_file.close()
    episode_ = 0
    while i < samples:
        ## Actor should be FIRST here
        if ( ( sim_work_queues is None ) or (eval_episode_data_queue is None)): ## off-policy version
            out = simEpoch(actor=actor, exp=exp, model=model, discount_factor=settings['discount_factor'], anchors=episode_, 
                               action_space_continuous=settings['action_space_continuous'], settings=settings, print_data=False,
                                p=1.0, validation=settings['train_on_validation_set'], bootstrapping=True, epsilon=1.0)
        else:
            out = simModelParrallel( sw_message_queues=sim_work_queues,
                                 model=model, settings=settings, 
                                 eval_episode_data_queue=eval_episode_data_queue, 
                                 anchors=settings['epochs'],
                                 type='bootstrapping')
        # if self._p <= 0.0:
        #    self._output_queue.put(out)
        (tuples, discounted_sum_, q_value_, evalData) = out
        (states_, actions_, result_states_, rewards_, falls_, G_t_, advantage_, exp_actions_) = tuples
        if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
            print ("Shape other states_: ", np.array(states_).shape)
            print ("Shape other action_: ", np.array(actions_).shape)
        # print ("States: ", states_)
        states.extend(states_)
        actions.extend(actions_)
        rewards.extend(rewards_)
        resultStates.extend(result_states_)
        falls.extend(falls_)
        G_ts.extend(G_t_)
        advantage.extend(advantage_)
        exp_actions.extend(exp_actions_)
        
        i=i+len(states_)
        episode_ += 1
        episode_ = episode_ % settings["epochs"]
        if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
            print("Number of Experience samples so far: ", i)
        # print ("States: ", states)
        # print ("Actions: ", actions)
        # print ("Rewards: ", rewards)
        # print ("ResultStates: ", resultStates)
        

    print ("Done collecting experience.")
    return (np.array(states), np.array(actions), np.array(resultStates), np.array(rewards), 
            np.array(falls), np.array(G_ts), np.array(exp_actions), np.array(advantage))  

