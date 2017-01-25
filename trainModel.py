
import sys
sys.setrecursionlimit(50000)
import os
import json
from numpy import dtype
sys.path.append("../")
sys.path.append("../characterSimAdapter/")
import math
import numpy as np

# import characterSim

from actor.ActorInterface import ActorInterface

from model.ForwardDynamicsNetwork import ForwardDynamicsNetwork
from util.ExperienceMemory import ExperienceMemory

import random
# import cPickle
import dill

import cProfile, pstats, io
# import memory_profiler
# import psutil
import gc
# from guppy import hpy; h=hpy()
# from memprof import memprof

from model.ModelUtil import *
from ModelEvaluation import *
from util.SimulationUtil import *

"""
    Converting Python2.7 to python3
    match on print
    print ([ |+|-|-|A-Z|a-z|0-9|:|\"|\(|\)|_|\.|,|/|\[|\]|'|%|<|>|?|\*|\\]*)
"""

# @profile(precision=5)
# @memprof(plot = True)
def train(settingsFileName):
        
    # pr = cProfile.Profile()
    # pr.enable()
    # try:
        np.random.seed(23)
        file = open(settingsFileName)
        settings = json.load(file)
        print ("Settings: " , str(json.dumps(settings)))
        file.close()
        settings = validateSettings(settings)
        anchor_data_file = open(settings["anchor_file"])
        _anchors = getAnchors(anchor_data_file)
        print ("Length of anchors epochs: ", str(len(_anchors)))
        anchor_data_file.close()
        train_forward_dynamics=True
        model_type= settings["model_type"]
        directory= getDataDirectory(settings)
        num_actions= settings["num_actions"]
        rounds = settings["rounds"]
        epochs = settings["epochs"]
        num_states=settings["num_states"]
        epsilon = settings["epsilon"]
        discount_factor=settings["discount_factor"]
        # max_reward=settings["max_reward"]
        reward_bounds=np.array(settings["reward_bounds"])
        batch_size=settings["batch_size"]
        train_on_validation_set=settings["train_on_validation_set"]
        state_bounds = np.array(settings['state_bounds'])
        discrete_actions = np.array(settings['discrete_actions'])
        print ("Sim config file name: ", str(settings["sim_config_file"]))
        # c = characterSim.Configuration(str(settings["sim_config_file"]))
        # c = characterSim.Configuration("../data/epsilon0Config.ini")
        action_space_continuous=settings['action_space_continuous']
        
        if action_space_continuous:
            action_bounds = np.array(settings["action_bounds"], dtype=float)
            
        if action_space_continuous:
            experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), settings['expereince_length'], continuous_actions=True)
        else:
            experience = ExperienceMemory(len(state_bounds[0]), 1, settings['expereince_length'])
            
        if settings['visualize_learning']:  
            from RLVisualize import RLVisualize
            rlv = RLVisualize(directory+str(settings['agent_name']))
            rlv.setInteractive()
            rlv.init()
        if (settings['train_forward_dynamics']):
            print ("Created forward dynamics network")
            # forwardDynamicsModel = ForwardDynamicsNetwork(state_length=len(state_bounds[0]),action_length=len(action_bounds[0]), state_bounds=state_bounds, action_bounds=action_bounds, settings_=settings)
            forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, None, None)
            if settings['visulaize_forward_dynamics']:
                from NNVisualize import NNVisualize
                nlv = NNVisualize(title=str("Forward Dynamics Model") + " with " + str(settings["model_type"]))
                nlv.setInteractive()
                nlv.init()
        
        ### Using a wrapper for the type of actor now
        actor = createActor(settings['environment_type'], settings, experience)
        # this is the process that selects which game to play
        exp = createEnvironment(str(settings["sim_config_file"]), settings['environment_type'])
        # Create the model that will be used for learning
        if action_space_continuous:
            model = createRLAgent(settings['agent_name'], state_bounds, action_bounds, reward_bounds, settings)
        else:
            model = createRLAgent(settings['agent_name'], state_bounds, discrete_actions, reward_bounds, settings)
            
        agent = LearningAgent(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
        agent.setSettings(settings)
        exp.getActor().init()
        exp.getEnvironment().init()
        experience, state_bounds, _, _ = collectExperience(actor, exp, model, settings)
        agent.setExperience(experience)
        agent.getExperience().setStateBounds(state_bounds)
        agent.getExperience().setRewardBounds(reward_bounds)
        # Create learning agent
        
        # HACK
        # state_bounds[0,0] = state_bounds[0,2]
        # state_bounds[1,0] = state_bounds[1,2]
        model.setStateBounds(state_bounds)
        model.setActionBounds(action_bounds)
        model.setRewardBounds(reward_bounds)
        # These are already normalized to be -1 <  < 1
        print ("Reward History: ", agent.getExperience()._reward_history)
        print ("Action History: ", agent.getExperience()._action_history)
        print ("Action Mean: ", np.mean(agent.getExperience()._action_history))
        
        # paramSampler = exp.getActor().getParamSampler()
        best_eval=-100000000.0
        best_dynamicsLosses= best_eval*-1.0
                
        values = []
        discounted_values = []
        bellman_error = []
        reward_over_epoc = []
        dynamicsLosses = []
        
        trainData = {}
        trainData["mean_reward"]=[]
        trainData["std_reward"]=[]
        trainData["mean_bellman_error"]=[]
        trainData["std_bellman_error"]=[]
        trainData["mean_discount_error"]=[]
        trainData["std_discount_error"]=[]
        trainData["mean_forward_dynamics_loss"]=[]
        trainData["std_forward_dynamics_loss"]=[]
        trainData["mean_eval"]=[]
        trainData["std_eval"]=[]
        
        if (settings['use_sampling_exploration']):
            forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, actor, exp)
            forwardDynamicsModel.init(len(state_bounds[0]), len(action_bounds[0]), state_bounds, action_bounds, actor, exp, settings)
            sampler = createSampler(settings, exp)
            sampler.setPolicy(model)
            sampler.setForwardDynamics(forwardDynamicsModel)
            sampler.setSettings(settings)
            
            # sampler.setSimulator(exp)
        
        if (settings['train_forward_dynamics']):
            actor.setForwardDynamicsModel(forwardDynamicsModel)
            forwardDynamicsModel.setActor(actor)
            agent.setForwardDynamics(forwardDynamicsModel)
            # forwardDynamicsModel.setEnvironment(exp)
            forwardDynamicsModel.init(len(state_bounds[0]), len(action_bounds[0]), state_bounds, action_bounds, actor, exp, settings)
        actor.setPolicy(model)
        agent.setPolicy(model)
        
        if not os.path.exists(directory):
            os.makedirs(directory)

        if (settings["save_experience_memory"]):
            file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+"expBufferInit.pkl"
            actor.getExperience().saveToFile(file_name)
            
        # copy settings file
        file = open(settingsFileName, 'r')
        out_file_name=directory+os.path.basename(settingsFileName)
        print ("Saving settings file with data: ", out_file_name)
        out_file = open(out_file_name, 'w')
        out_file.write(file.read())
        file.close()
        out_file.close()
        
        print ("Starting first round")
        steps_=0
        for round_ in range(rounds):
            p = max((((rounds * settings['epsilon_annealing']) - round_) / float(rounds)), 0.1)
            for epoch in range(epochs):
                if settings['use_guided_policy_search']:
                    out = simEpoch(actor=actor, exp=exp, model=sampler, discount_factor=settings['discount_factor'], anchors=_anchors[epoch], 
                                   action_space_continuous=settings['action_space_continuous'], settings=settings, print_data=False,
                                    p=p, validation=settings['train_on_validation_set'])
                else:
                    out = simEpoch(actor=actor, exp=exp, model=agent.getPolicy(), discount_factor=settings['discount_factor'], anchors=_anchors[epoch], 
                                   action_space_continuous=settings['action_space_continuous'], settings=settings, print_data=False,
                                    p=p, validation=settings['train_on_validation_set'])
                # if self._p <= 0.0:
                #    self._output_queue.put(out)
                (tuples, discounted_sum, q_value, evalData) = out
                (states, actions, rewards, result_states, falls) = tuples
                for tr in range(settings['training_updates_per_sim_action']):
                    # print (len(states), " tuples to learn from")
                    for k in range(len(states)):
                        state_= states[k]
                        action= actions[k]
                        resultState = result_states[k]
                        reward= rewards[k]
                        fall= fall[k]
                        
                        if action_space_continuous:
                            # actor.getExperience().insert(norm_state(state_, state_bounds), [norm_action(action, action_bounds)], norm_state(resultState, state_bounds), norm_reward([reward], reward_bounds))
                            actor.getExperience().insert(state_, action, resultState, [reward], [fall])
                        else:
                            actor.getExperience().insert(state_, [action], resultState, [reward], [fall])
                        if actor.getExperience().samples() > batch_size and ( (steps_ % settings['sim_action_per_training_update']) == 0 ):
                            _states, _actions, _result_states, _rewards, _falls = actor.getExperience().get_batch(batch_size)
                            # print ("Q values:", agent.getPolicy().q_values(_states[0]))
                            # print ("States: " + str(_states) + " ResultsStates: " + str(_result_states) + " Rewards: " + str(_rewards) + " Actions: " + str(_actions))
                            cost=0
                            if (settings["train_rl_learning"]):
                                # cost = model.train(_states, _actions, _rewards, _result_states)
                                (cost, dynamicsLoss) = agent.train(_states=_states, _actions=_actions, _rewards=_rewards, _result_states=_result_states, _falls=_falls)
                            if not np.isfinite(cost):
                                print ("States: " + str(_states) + " ResultsStates: " + str(_result_states) + " Rewards: " + str(_rewards) + " Actions: " + str(_actions))
                                print ("Training cost is Nan: ", cost)
                                sys.exit()
                            if (settings['train_forward_dynamics']):
                                dynamicsLoss = forwardDynamicsModel.train(states=_states, actions=_actions, result_states=_result_states)
                            # grads = agent.getPolicy().getGrads(_states)
                            # print ("Grads: ", len(grads[0]), " values ", grads[0])
                            # grads = forwardDynamicsModel.getGrads(_states, _actions, _result_states)
                            # print ("Dynamics Grads: ", len(grads[0]), " values ", grads[0])
                        steps_ = steps_ + 1
                        # print ("Current Tuple: " + str(actor.getExperience().current()))
                        # rewards.append([reward])
                if actor.getExperience().samples() > batch_size:
                    states, actions, result_states, rewards, falls = actor.getExperience().get_batch(batch_size)
                    # print ("Batch size: " + str(batch_size))
                    error = agent.getPolicy().bellman_error(states, actions, rewards, result_states, falls)
                    # rewards = map(scale_reward, rewards, [reward_bounds]* len(rewards)) # scales the rewards back to env space
                    estimated_values = agent.getPolicy().q_values(states)
                    
                    if not all(np.isfinite(error)):
                        print ("States: " + str(states) + " ResultsStates: " + str(result_states) + " Rewards: " + str(rewards) + " Actions: " + str(actions))
                        print ("Bellman Error is Nan: " + str(error))
                        sys.exit()
                    
                    error = np.mean(np.fabs(error))
                    if (settings['train_forward_dynamics']):
                        dynamicsLoss = forwardDynamicsModel.bellman_error(np.array(states), np.array(actions), np.array(result_states))
                        dynamicsLoss = np.mean(np.fabs(dynamicsLoss))
                        dynamicsLosses.append(dynamicsLoss)
                    if (settings['train_forward_dynamics']):
                        print ("Round: " + str(round_) + " Epoch: " + str(epoch) + " With mean reward: " + str(np.mean(rewards)) + " and estimated value: " + str(np.mean(estimated_values)) + " bellman error: " + str(error) + " ForwardPredictionLoss: " + str(dynamicsLoss) + " p: " + str(p))
                    else:
                        print ("Round: " + str(round_) + " Epoch: " + str(epoch) + " With mean reward: " + str(np.mean(rewards)) + " and estimated value: " + str(np.mean(estimated_values)) + " bellman error: " + str(error) + " p: " + str(p))
                    # discounted_values.append(discounted_sum)
                    reward_over_epoc.append(np.mean(rewards))
                    bellman_error.append(error)
                    # print (sys.getsizeof(error))
            
            
                # this->_actor->iterate();
            if (round_ % settings['plotting_update_freq_num_rounds']) == 0:
                # Running less often helps speed learning up.
                print (" Evaluating model") 
                mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error, mean_eval, std_eval = evalModel(actor, exp, agent.getPolicy(), discount_factor, 
                                                    anchors=_anchors[:settings['eval_epochs']], action_space_continuous=action_space_continuous, settings=settings, evaluation=True)
                print (" Done Evaluating model")
                # print (mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error)
                if mean_bellman_error > 10000:
                    print ("Error to big: ")
                else:
                    
                    if (settings['train_forward_dynamics']):
                        mean_dynamicsLosses = np.mean(dynamicsLosses)
                        std_dynamicsLosses = np.std(dynamicsLosses)
                        dynamicsLosses = []
                        if mean_dynamicsLosses < best_dynamicsLosses:
                            best_dynamicsLosses = mean_dynamicsLosses
                            print ("Saving BEST current forward dynamics model: " + str(best_dynamicsLosses))
                            file_name_dynamics=directory+"forward_dynamics_"+str(settings['agent_name'])+"_Best.pkl"
                            f = open(file_name_dynamics, 'wb')
                            dill.dump(forwardDynamicsModel, f)
                            f.close()
                    
                    if (mean_eval > best_eval):
                        best_eval = mean_eval
                        print ("Saving BEST current model: " + str(best_eval))
                        file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+"_Best.pkl"
                        f = open(file_name, 'wb')
                        dill.dump(agent.getPolicy(), f)
                        f.close()
                    trainData["mean_reward"].append(mean_reward)
                    # print ("Mean Rewards: " + str(mean_rewards))
                    trainData["std_reward"].append(std_reward)
                    trainData["mean_bellman_error"].append(mean_bellman_error)
                    trainData["std_bellman_error"].append(std_bellman_error)
                    trainData["mean_discount_error"].append(mean_discount_error)
                    trainData["std_discount_error"].append(std_discount_error)
                    trainData["mean_eval"].append(mean_eval)
                    trainData["std_eval"].append(std_eval)
                    if (settings['train_forward_dynamics']):
                        trainData["mean_forward_dynamics_loss"].append(mean_dynamicsLosses)
                        trainData["std_forward_dynamics_loss"].append(std_dynamicsLosses)
                        
                    if settings['save_trainData']:
                        fp = open(directory+"trainingData.json", 'w')
                        json.dump(trainData, fp)
                        fp.close()
                    # draw data
                    if settings['visualize_learning']:
                        rlv.updateBellmanError(np.array(trainData["mean_bellman_error"]), np.array(trainData["std_bellman_error"]))
                        # rlv.updateReward(np.array(trainData["mean_reward"]), np.array(trainData["std_reward"]))
                        rlv.updateReward(np.array(trainData["mean_eval"]), np.array(trainData["std_eval"]))
                        rlv.updateDiscountError(np.fabs(trainData["mean_discount_error"]), np.array(trainData["std_discount_error"]))
                        rlv.redraw()
                        rlv.setInteractiveOff()
                        rlv.saveVisual(directory+"pendulum_agent_"+str(settings['agent_name']))
                        rlv.setInteractive()
                        # rlv.redraw()
                    if (settings['visulaize_forward_dynamics']):
                        nlv.updateLoss(np.array(trainData["mean_forward_dynamics_loss"]), np.array(trainData["std_forward_dynamics_loss"]))
                        nlv.redraw()
                        nlv.setInteractiveOff()
                        nlv.saveVisual(directory+"trainingGraphNN")
                        nlv.setInteractive()
                
                        
            # mean_reward = std_reward = mean_bellman_error = std_bellman_error = mean_discount_error = std_discount_error = None
            if ( round_ % 5) == 0 :
                print ("Saving current model")
                
                file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+".pkl"
                f = open(file_name, 'w')
                dill.dump(agent.getPolicy(), f)
                f.close()
                
                f = open(directory+"trainingData_" + str(settings['agent_name']) + ".json", "w")
                json.dump(trainData, f, sort_keys=True, indent=4)
                f.close()
                
                if (settings["save_experience_memory"]):
                    file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+"expBuffer.pkl"
                    f = open(file_name, 'w')
                    dill.dump(actor.getExperience(), f)
                    f.close()
                
            # gc.collect()    
            # print (h.heap())
            
        # bellman_error = np.fabs(np.array(bellman_error))
        # print ("Mean Bellman error: " + str(np.mean(np.fabs(bellman_error))))
        # print ("STD Bellman error: " + str(np.std(np.fabs(bellman_error))))
        
        # discounted_values = np.array(discounted_values)
        # values = np.array(values)
        
        # print ("Discounted reward difference: " + str(discounted_values - values))
        # print ("Discounted reward difference Avg: " +  str(np.mean(np.fabs(discounted_values - values))))
        # print ("Discounted reward difference STD: " +  str(np.std(np.fabs(discounted_values - values))))
        # reward_over_epoc = np.array(reward_over_epoc)
        print ("Reward Avg: " + str(sum(reward_over_epoc)/len(reward_over_epoc)))
        
        exp.finish()
        
        file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+".pkl"
        f = open(file_name, 'wb')
        dill.dump(agent.getPolicy(), f)
        f.close()
        
        f = open(directory+"trainingData_" + str(settings['agent_name']) + ".json", "w")
        json.dump(trainData, f, sort_keys=True, indent=4)
        f.close()
        
        if (settings['train_forward_dynamics']):
            file_name_dynamics=directory+"forward_dynamics_"+str(settings['agent_name'])+".pkl"
            f = open(file_name_dynamics, 'wb')
            dill.dump(forwardDynamicsModel, f)
            f.close()
        
        """
        pr.disable()
        f = open('x.prof', 'a')
        pstats.Stats(pr, stream=f).sort_stats('time').print_stats()
        f.close()
        """
        """except: # catch *all* exceptions
        e = sys.exc_info()[0]
        print ("Error: " + str(e))
        print ("State " + str(state_) + " action " + str(pa) + " newState " + str(resultState) + " Reward: " + str(reward))
        
        """ 
          
    
    
if (__name__ == "__main__"):
    
    train(sys.argv[1])
