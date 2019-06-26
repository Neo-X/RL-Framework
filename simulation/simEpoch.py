

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
from model.ModelUtil import *
from util.utils import checkSetting, checkSettingExists
# import memory_profiler
# import resources

    
    
# @profile(precision=5)
def simEpoch(actor, exp, model, discount_factor, anchors=None, action_space_continuous=False, settings=None, print_data=False, 
             p=0.0, validation=False, epoch=0, evaluation=False, _output_queue=None, bootstrapping=False, visualizeEvaluation=None,
             sampling=False, epsilon=None,
             worker_id=None, movieWriter=None):
    import numpy as np
    """
        
        evaluation: If True than the simulation is being evaluated and the episodes will not terminate early.
        bootstrapping: is used to collect initial random actions for the state bounds to be calculated and to init the expBuffer
        epoch: is an integer that can be used to help create repeatable episodes to evaluation
        _output_queue: is the queue exp tuples should be put in so the learning agents can pull them out
        p:  is the probability of selecting a random action
        actor: 
        
        The shape of the data collected in this simulation is 3D Time x agent# x value.
        This is more than the normal 2D vector that is Time x value because I want to be able to support
        multi-agent simulation/learning now. This organization only exists in this part of the code for now.
        Data exported out of here should be converted back to 2D data.
        
    """
    ### Maybe the model has state
    model.reset()
    if action_space_continuous:
        action_bounds = model.getActionBounds()
        omega = settings["omega"]
        # noise_ = action_bounds[0] * 0.0
        
    if ( (bootstrapping == True) and 
         (settings["exploration_method"] == "sampling") ):
        settings = copy.deepcopy(settings)
        settings["exploration_method"] = "gaussian_network"
        settings["evalaute_with_MBRL"] = False
        
    if (movieWriter is not None):
        exp.setMovieWriter(movieWriter)
    # print("bootstrapping: ", bootstrapping, " settings[exploration_method]: ", settings["exploration_method"])
    # print ("Start sim state bounds: ", model.getStateBounds())
    action_selection = range(len(settings["discrete_actions"]))   
    reward_bounds = np.array(settings['reward_bounds'] )
    pa=None
    # epsilon = settings["epsilon"]
    # Actor should be FIRST here
    exp.getActor().initEpoch()
    reset_prop = 1
    if (checkSettingExists(settings, "reset_on_fall_probability")):
        reset_prop = settings["reset_on_fall_probability"]
        
    reset_prop_tmp = np.random.rand(1)[0]
    """
    if validation:
        exp.generateValidation(anchors, epoch)
    else:
        exp.generateEnvironmentSample()
    """ 
    # exp.initEpoch()
    exp.initEpoch()
    # print ("sim EXP: ", exp)
    actor.initEpoch()
    if ("llc_index" in settings):
        ### Bad hack for now to use llc in env
        if (settings["environment_type"] == "Multiworld"
            or settings["environment_type"] == "MultiworldHRL"):
            exp.setLLC(model.getAgents()[settings["llc_index"]])
        else:
            exp.getEnvironment().getEnv().setLLC(model.getAgents()[settings["llc_index"]])
    # model.initEpoch(exp)
    state_ = exp.getState()
    # pa = model.predict(state_)
    """
    if (not bootstrapping):
        q_values_ = [model.q_value(state_)]
    else:
        q_values_ = []
    """
    viz_q_values_ = []
    
    ### Test to make sure the agent initial pose and imitation pose are the same.
    # pose_diff = exp.getEnvironment().getImitationState() - exp.getEnvironment()._sim.getState()
    # print ("pose_diff: ", pose_diff)
            
    # q_value = model.q_value(state_)
    # print ("Updated parameters: " + str(model.getNetworkParameters()[1]))
    # print ("q_values_: " + str(q_value) + " Action: " + str(action_))
    # original_val = q_value
    discounted_sum = 0
    discounted_sums = []
    G_t = []
    G_t_rewards = []
    baseline = []
    G_ts = []
    baselines_ = []
    advantage = []
    state_num=0
    i_=0
    last_epoch_end=0
    reward_=0
    states = [] 
    states__ = []
    actions = []
    rewards = []
    falls = []
    result_states___ = []
    exp_actions = []
    evalDatas=[]
    stds=[]
    bad_sim_state = False
    if ("divide_by_zero2" in settings
        and (settings["divide_by_zero2"] == True)
        and (not bootstrapping)):
        d = 3 / 0
    
    i_ = 0
    while (i_ < settings['max_epoch_length']):
        
        state_ = exp.getState()
        # print ("state_: ", repr(np.array(state_).shape))
        # print ("state_: ", state_)
        
        states.append(state_)
        states__.extend(state_)

        if (not (visualizeEvaluation == None)):
            viz_q_values_.append(model.q_value(state_)[0][0])
            if (len(viz_q_values_)>30):
                 viz_q_values_.pop(0)
            visualizeEvaluation.updateLoss(viz_q_values_, np.zeros(len(viz_q_values_)))
            visualizeEvaluation.redraw()
        action=None
        if action_space_continuous:
            """
                epsilon greedy action select
                pa1 is best action from policy
                ra1 is the noisy policy action action
                ra2 is the random action
                e is proabilty to select random action
                0 <= e < omega < 1.0
            """
            r = np.random.rand(1)[0]
            # print ("float(settings['anneal_exploration']), epsilon * p: ", float(settings['anneal_exploration']), epsilon * p)
            if ((not evaluation) ### This logic has gotten far to complicated.... 
                and 
                (
                    ( ### Explore if r < annealing value
                        (settings['on_policy']) 
                        and 
                        (
                            ("anneal_exploration" in settings) 
                            and (settings['anneal_exploration'] != False)
                            and (r < (max(float(settings['anneal_exploration']), epsilon * p))) 
                        )
                    ) 
                        or ### Always explore 
                        ( 
                            (settings['on_policy'])
                            and ("anneal_exploration" in settings) 
                            and (settings['anneal_exploration'] == False)
                        )
                        or ### Always explore 
                        (
                            (settings['on_policy'])
                            and (not "anneal_exploration" in settings) 
                        )
                        or  
                        ( ### Explore sometimes
                            (settings['on_policy'])
                            and (r < (epsilon * p)) 
                        )
                    )
                ): # explore random actions
                
                
                # print ("state_", repr(state_))
                (action, exp_action) = model.sample(state_, p=p, sim_index=worker_id, bootstrapping=bootstrapping)
                # print ("action", repr(action))
            else: 
                ### exploit policy
                exp_action = [[0]] *  len(state_)
                ## For sampling method to skip sampling during evaluation.
                use_MBRL = False
                if ("evalaute_with_MBRL" in settings and
                    (settings["evalaute_with_MBRL"] == True) ):
                    use_MBRL = True

                pa = model.predict(state_, evaluation_=evaluation, p=p, sim_index=worker_id, 
                                   bootstrapping=bootstrapping, use_mbrl=use_MBRL)
                
                action = pa
                # print ("Exploitation: ", action , " epsilon: ", epsilon * p)
            outside_bounds=False
            action_=None
            if (settings["clamp_actions_to_stay_inside_bounds"] or (settings['penalize_actions_outside_bounds'])):
                (action_, outside_bounds) = clampActionWarn(action, action_bounds)
                if (settings['clamp_actions_to_stay_inside_bounds']):
                    action = action_
            if (settings["visualize_forward_dynamics"] and settings['train_forward_dynamics']):
                predicted_next_state = model.getForwardDynamics().predict(np.array(state_), action)
                # exp.visualizeNextState(state_[0], [0,0]) # visualize current state
                exp.visualizeNextState(predicted_next_state, action)
                
                action__ = model.predict(state_)
                actions_ = []
                dirs = []
                deltas = np.linspace(-0.5,0.5,10)
                for d in range(len(deltas)):
                    action_ = np.zeros_like(action__)
                    for i in range(len(action_)):
                        action_[i] = action__[i]
                    action_[0] = action__[0] + deltas[d] 
                    if ( ('anneal_mbae' in settings) and settings['anneal_mbae'] ):
                        mbae_lr = p * settings["action_learning_rate"]
                    else:
                        mbae_lr = settings["action_learning_rate"]
                    action_new_ = getOptimalAction(model.getForwardDynamics(), model.getPolicy(), action_, state_, mbae_lr)
                    # actions.append(action_new_)
                    actions_.append(action_)
                    print("action_new_: ", action_new_[0], " action_: ", action_[0])
                    if ( (float(action_new_[0][0]) - float(action_[0])) > 0 ):
                        dirs.append(1.0)
                    else:
                        dirs.append(-1.0)
                    
                # return _getOptimalAction(forwardDynamicsModel, model, action, state)
                
                # action_ = _getOptimalAction(model.getForwardDynamics(), model.getPolicy(), action, state_)
                exp.getEnvironment().visualizeActions(actions_, dirs)
                ## The perfect action?
                exp.getEnvironment().visualizeAction(action__)
                
            
            if (not settings["train_actor"]): # hack to use debug critic only
                """
                    action = np.random.choice(action_selection)
                    action__ = actor.getActionParams(action)
                    action = action__
                    
                    pa = model.predict(state_)
                    action = pa
                """
                pass
                # action=[0.2]
            # print("exp_action: ", exp_action, " action", action)
            reward_ = actor.actContinuous(exp,action)
            a = 0

            # support for mixing rewards across levels
            if ("hlc_index" in settings
                    and "llc_index" in settings
                    and "hlc_intrinsic_weight" in settings):
                a = reward_[settings["llc_index"]] * settings["hlc_intrinsic_weight"]
            b = 0
            if ("hlc_index" in settings
                    and "llc_index" in settings
                    and "llc_task_weight" in settings):
                b = reward_[settings["hlc_index"]] * settings["llc_task_weight"]
            if ("hlc_index" in settings
                    and "llc_index" in settings):
                reward_[settings["hlc_index"]] += a
                reward_[settings["llc_index"]] += b

            """
            if ( settings['train_reward_predictor'] and (not bootstrapping)):
                predicted_reward = model.getForwardDynamics().predict_reward(state_, [action])
                print ("Actual Reward: ", reward_, " Predicted reward: ", predicted_reward)
            """
            agent_not_fell = actor.hasNotFallen(exp)
            if (outside_bounds and settings['penalize_actions_outside_bounds']):
                ### TODO: this penalty should really be a function of the distance the action was outside the bounds
                reward_ = reward_ + settings['reward_lower_bound']  
            # print ("Action: ", action, " reward: ", reward_, " p: ", p)
        elif not action_space_continuous:
            """
            action = random.choice(action_selection)
            action = eGreedy(pa, action, epsilon * p)
            reward_ = actor.act(exp, action)
            """
            pa = model.predict(state_)
            action = random.choice(action_selection)
            action = eGreedy(pa, action, epsilon * p)
            # print("Action selection:", action_selection, " action: ", action)
            action__ = actor.getActionParams(action)
            action = [action]
            # print ("Action selected: " + str(action__))
            # reward = act(action)
            # print ("performing action: ", action)
            reward_ = actor.actContinuous(exp, action__, bootstrapping=True)
            agent_not_fell = actor.hasNotFallen(exp)
            # print ("performed action: ", reward)
        # print ("Reward: ", reward_)
        resultState_ = exp.getState()
        # print ("resultState_: ", np.array(resultState_).shape)
        if (movieWriter is not None
            and (not exp.movieWriterSupport())):
            ### If the sim does not have it's own writing support
            vizData = exp.getEnvironment().getFullViewData()
            # movie_writer.append_data(np.transpose(vizData))
            # print ("sim image mean: ", np.mean(vizData), " std: ", np.std(vizData))
            image_ = np.zeros((vizData.shape))
            for row in range(len(vizData)):
                image_[row] = vizData[len(vizData)-row - 1]
            # print ("Writing image to video") 
            movieWriter.append_data(image_)
        if ("use_learned_reward_function" in settings
            and (settings["use_learned_reward_function"])
            and not ("return_rnn_sequence" in settings
                     and (settings["return_rnn_sequence"] == True))):
            rewmodel = model.getForwardDynamics()
            w_d = -2.0
            if ("learned_reward_function_norm_weight" in settings):
                w_d = settings["learned_reward_function_norm_weight"]
            if ("train_reward_distance_metric" in settings
                and (settings["train_reward_distance_metric"])): 
                rewmodel = model.getRewardModel() 
            if ("fd_algorithm" in settings
                and (settings["fd_algorithm"] == "algorithm.DiscriminatorKeras.DiscriminatorKeras")):
                ### Use Discriminator 
                # print ("state_[0]: ", np.array(state_).shape)
                # print ("resultState_[0]: ", np.array(resultState_).shape)
                # reward_ = model.getForwardDynamics().predict([state_[0][1]], [resultState_[0][1]])[0][0]
                # print ("learned imitation reward: ", reward_, " imitation state sum: ", np.sum(resultState_[0][1]))
                reward_ = rewmodel.predict([state_[0][0]], [resultState_[0][0]])[0][0]
                # print ("learned reward: ", reward_)

            elif ("use_encoding_for_reward" in settings
                and (settings["use_encoding_for_reward"] == True)):
                reward__ = exp.computeImitationReward(rewmodel.computeEncodingDiff)
                reward_ = np.exp((reward__*reward__)*w_d)
            else:
                if ( (("train_LSTM_Reward" in settings)
                        and (settings["train_LSTM_Reward"] == True))
                    and 
                        (settings["use_learned_reward_function"] == "dual")
                    ):    
                    reward__0 = exp.computeImitationReward(rewmodel.predict)
                    reward__1 = exp.computeImitationReward(rewmodel.predict_reward)
                    reward__ = ((reward__0 * 0.5) + (reward__1 * 0.5))
                    # print ("reward__: ", reward__, " reward__0: ", reward__0, " reward__1: ", reward__1)
                elif ( (("train_LSTM_Reward" in settings)
                        and (settings["train_LSTM_Reward"] == True))
                    and 
                        (settings["use_learned_reward_function"] == "fd_only")
                    ):    
                    reward__0 = exp.computeImitationReward(rewmodel.predict)
                    # reward__1 = exp.computeImitationReward(rewmodel.predict_reward)
                    reward__ = (reward__0)
                    # print (" reward__0: ", reward__0)
                elif (("train_LSTM_Reward" in settings)
                    and (settings["train_LSTM_Reward"] == True)
                    and (not (("train_LSTM_FD" in settings)
                    and (settings["train_LSTM_FD"] == True)))
                    ):    
                    reward__ = exp.computeImitationReward(rewmodel.predict_reward)
                    # print ("reward__: ", reward__)
                else:
                    reward__ = exp.computeImitationReward(rewmodel.predict)
                
                # print ("Reward: ", reward__)
                if ("learned_reward_smoother" in settings
                    and (settings["learned_reward_smoother"] == False)):
                    reward_ = reward__
                if ("learned_reward_smoother" in settings
                    and (settings["learned_reward_smoother"] == "bce")):
                    reward_ = -1.0 * reward__
                else:
                    reward_ = np.exp((reward__*reward__)*w_d)
                # print ("Reward: ", reward_)
                # reward_ = reward__
                    
                if ("use_sparse_sequence_based_reward" in settings
                    and (settings["use_sparse_sequence_based_reward"] == True)):
                    ### Only give reward at end of trajectory
                    if ((exp.endOfEpoch() and settings['reset_on_fall'] and (not evaluation))
                        or ((i_ + 1) >= settings['max_epoch_length'])): ### End of trajectory
                        # print ("End of trajectory")
                        pass
                    else:
                        reward_ = reward_ * 0
                        
        if ("use_learned_fast_function" in settings
            and (settings["use_learned_fast_function"] == True)):
            sim_time = exp.getAnimationTime()
            reward_ = exp.computeReward(resultState_, sim_time)
            # print("reward: ", reward_)
            
        if ((exp.endOfEpoch() and settings['reset_on_fall'])
            and 
            ("use_fall_reward_shaping2" in settings
             and (settings["use_fall_reward_shaping2"] == True))
            ):
            if (len(np.array(reward_).shape) == 2):
                reward_ = np.array(reward_) + (-1.0 * 1/(1-settings["discount_factor"]))
            else:
                reward_ = np.mean(reward_) + (-1.0 * 1/(1-settings["discount_factor"]))
        # print ("reward: ", reward_)
        # baseline.append(model.q_value(state_))
        
        G_t.append(np.array([[0]])) # *(1.0-discount_factor)))
        for i in range(len(G_t)):
            if isinstance(reward_, (list, tuple, np.ndarray)):
                assert len(np.array(reward_).shape) == 2, "reward shape is " + str(np.array(reward_).shape) + str(reward_) 
                # G_t[i] = G_t[i] + (((np.power(discount_factor,(len(G_t)-i)-1) * (np.array(reward_) ))))
                # print( "reward: ", repr(np.array(reward_)) )
                # print( "G_t: ", repr(np.array(G_t)) )
            else:
                # G_t[i] = G_t[i] + (((np.power(discount_factor,(len(G_t)-i)-1) * (np.array([reward_]) ))))
                reward_ = [[reward_]]
        
        
        if ("replace_next_state_with_imitation_viz_state" in settings
            and (settings["replace_next_state_with_imitation_viz_state"] == True)):
            # print ("resultState_: ", resultState_)
            # print ("Before resultState_[0][1]: ", np.array(resultState_[0][1]).shape)
            ### This only works properly in the dual state rep case.
            if ("replace_next_state_with_pose_state" in settings and
                  (settings["replace_next_state_with_pose_state"] == True)):
                # print ("Replacing result state data with imitation data")
                ob = np.asarray(exp.getEnvironment().getImitationState())
                ob = ob.flatten()
                resultState_[0][1] = ob
            elif ("use_dual_viz_state_representations" in settings
                  and (settings["use_dual_viz_state_representations"] == True)):
                ### Need agent data for simease net
                # ob = np.asarray(exp.getEnvironment().getVisualState())
                # ob = np.reshape(np.array(ob), (-1, 
                #             (np.prod(ob.shape))))
                state_[0][1] = resultState_[0][0]
            elif ("use_dual_dense_state_representations" in settings
                and (settings["use_dual_dense_state_representations"] == True)):
                pass 
            elif ("fd_use_multimodal_state" in settings and
                  (settings["fd_use_multimodal_state"] == True)):
                # print ("Replacing result state data with multi model imitation data")
                ob = np.asarray(exp.getEnvironment().getMultiModalImitationState())
                ob = ob.flatten()
                resultState_[0][1] = ob
            else:
                ob = np.asarray(exp.getEnvironment().getImitationVisualState())
                ob = ob.flatten()
                resultState_[0][1] = ob
            # print ("resultState_[0][0]: ", np.array(resultState_[0][0]).shape)    
            # print ("resultState_[0][1]: ", np.array(resultState_[0][1]).shape)
        ## For testing remove later
        if (settings["use_back_on_track_forcing"] and (not evaluation)):
            exp.getControllerBackOnTrack()
        # print ("reward_: ", reward_)
        if print_data:
            # print ("State " + str(state_) + " action " + str(pa) + " newState " + str(resultState) + " Reward: " + str(reward_))
            # print ("Value: ", model.q_value(state_), " Action " + str(pa) + " Reward: " + str(reward_) + " Discounted Sum: " + str(discounted_sum) )
            if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
                value__ = 0
                if ( not bootstrapping ):
                    value__ = model.q_value(state_)
                print ("Value: ", value__, " Action " + str(action) + " Reward: " + str(reward_) )
                # if ( settings['train_reward_predictor'] and (settings['train_forward_dynamics'])):
                    # predicted_reward = rewmodel.predict_reward(state_, action)
                    # print ("Predicted reward: ", predicted_reward) 
                print ("Agent has fallen: ", not agent_not_fell )
                # print ("Python Reward: " + str(reward(state_, resultState)))
                
            
        ### I can't just unpack the vector of states here in a multi char sim because the 
        ### Order needs to be preserved for computing the advantage.
        actions.append(action)
        # print ("reward_: ", reward_)
        rewards.append(reward_)
        result_states___.append(resultState_)
        if (worker_id is not None):
            # Pushing working id as fall value for multi task training
            if ("ask_env_for_multitask_id" in settings 
                and (settings["ask_env_for_multitask_id"] == True)):
                worker_id = exp.getTaskID()
                # print ("Task ID: ", worker_id)
                falls.append([[worker_id]] * len(state_))
            else:
                falls.append([[worker_id]] * len(state_))
        elif ("perform_multiagent_training" in settings
              and (settings["perform_multiagent_training"] > 1)):
            falls_ = []
            for f in range(len(state_)):
                falls_.append([f])
            falls.append(falls_)
        else:
            # print("Pushing actual fall value before : ", agent_not_fell)
            # print("Pushing actual fall value: ", [agent_not_fell] * np.array(state_).shape[0])
            # falls.append([[agent_not_fell]] * len(state_))
            if type(agent_not_fell) is list:
                falls.append(agent_not_fell)
            else:
                falls.append([[agent_not_fell]])
                
        exp_act = exp_action
        exp_actions.append(exp_act)
        if ((_output_queue != None) and (not evaluation) and (not bootstrapping)): # for multi-threading
            for state__, act__, res__, rew__, fall__, exp__ in zip (states[-1], actions[-1], result_states___[-1], rewards[-1],  falls[-1], exp_actions[-1]):
                _output_queue.put(([state__], [act__], [res__], [rew__],  [fall__], [[0]], [exp__]), timeout=timeout_)
        
        state_num += 1
        # else:
        # print ("****Reward was: ", reward_)
        pa = None
        i_ += 1
        ### Don't reset during evaluation...
        # print ("exp.endOfEpoch(): ", exp.endOfEpoch())
        if (((exp.endOfEpoch() and settings['reset_on_fall'] and ((not evaluation)))
             and (reset_prop_tmp <= reset_prop) ) ### Allow option to collect some full trajectories  
            # or ((reward_ < settings['reward_lower_bound']) and (not evaluation))
                ):
            # falls[-1] = [[0]]
            break
                
        
        
    """
    if (settings['on_policy']):
        evalDatas.append(np.sum(rewards[last_epoch_end:])/float(settings['max_epoch_length']))
    else:
    """
    evalDatas.append(actor.getEvaluationData()/float(settings['max_epoch_length']))
    evalData = [np.mean(evalDatas)]
    # G_ts.extend(copy.deepcopy(G_t))
    G_ts.extend(copy.deepcopy(discounted_rewards(np.array(rewards), discount_factor)))
    discounted_sum = G_ts
    # q_value = baselines_
    
    if print_data:
        print ("Evaluation: ", str(evalData))
        print ("Eval Datas: ", evalDatas) 
    # print ("Evaluation Data: ", evalData)
        # print ("Current Tuple: " + str(experience.current()))
    ### Reset before predicting values for trajectory
    model.reset()
    # if (len(states[last_epoch_end:]) > 0):
    for a in range(len(states[0])):
        path = {}
        ### timestep, agent, state
        # path['states'] = copy.deepcopy(np.array(states[last_epoch_end:])[:,a,:])
        ### In multi-agent sims agents can have different sized state vectors
        if ( "use_centralized_critic" in settings
             and (settings["use_centralized_critic"] == True)):
            states__ = copy.deepcopy(np.array([st[a] for st in states[last_epoch_end:]]))
            ### Add states from other agents
            for l in [i for i in range(len(states[0])) if i!=a]:
                other_states__ = copy.deepcopy(np.array([st[l] for st in states[last_epoch_end:]]))
                states__ = [np.concatenate((states__[k], other_states__[k])) for k in range(len(states__))]
            ### Add actions from other agents
            for l in [i for i in range(len(states[0])) if i!=a]:
                other_actions__ = copy.deepcopy(np.array([st[l] for st in actions[last_epoch_end:]]))
                states__ = [np.concatenate((states__[k], other_actions__[k])) for k in range(len(states__))]
            path['states'] = copy.deepcopy(np.array(states__))
        else:
            path['states'] = copy.deepcopy(np.array([st[a] for st in states[last_epoch_end:]]))
        path['reward'] = np.array(np.array(rewards[last_epoch_end:])[:,a,:])
        path['falls'] = np.array(np.array(falls[last_epoch_end:])[:,a,:])
        path["terminated"] = False
        # print ("path['states']", path['states'].shape)
        ## Append so that we can preserve the paths/trajectory structure.
        if (len(rewards[last_epoch_end:]) > 0):
            paths = compute_advantage_(model, [path], discount_factor, settings['GAE_lambda'])
            adv__ = paths["advantage"]
            baselines_.append(np.array(paths["baseline"]))
            advantage.append(np.array(adv__))

    # print ("base diff: ", np.array(baseline) - np.array(baselines_))
    # G_t_rewards.append(0)
    if ( ('print_level' in settings) and (settings["print_level"]== 'debug') ):
        adv_r = [ [x, y] for x,y in zip(advantage, G_t_rewards)]
        R_r = [ [x_r, y_r, z_r] for x_r,y_r,z_r in zip(path['reward'], rewards[last_epoch_end:], G_t)]
        A_r = [ [x_r, y_r, z_r] for x_r,y_r,z_r in zip(advantage, discounted_rewards(np.array(rewards[last_epoch_end:]), discount_factor), baseline)]
        # print ("Adv: ", advantage)
        print ("last_epoch_end: ", last_epoch_end, " i_ ", i_)
        print ("Advantage, R: ", adv_r)
        print ("Rewards: ", R_r)
        print ("Advantage, discounted Reward, baseline: ", np.array(A_r))
        # print("Advantage, rewards, baseline: ", np.concatenate((advantage, G_t_rewards, baseline), axis=1))
    
    ### Fix data, Might need to unpack some vectors
    tmp_states = []
    tmp_actions = []
    tmp_res_states = []
    tmp_rewards = []
    tmp_discounted_sum = []
    tmp_G_ts = []
    tmp_falls = []
    tmp_exp_actions = []
    tmp_baselines_ = []
    tmp_advantage = []
    ### data is in format (state, agent), this "extend" does not work well for multi-agent simulation
    # print ("states: ", np.array(states).shape)
    for s in range(len(states)):
        # print ("State shape: ", np.array(states[s]).shape)
        # print ("actions shape: ", np.array(actions[s]).shape)
        tmp_states.extend(states[s])
        tmp_actions.extend(actions[s])
        tmp_res_states.extend(result_states___[s])
        tmp_rewards.extend(rewards[s])
        tmp_discounted_sum.extend(discounted_sum[s])
        tmp_G_ts.extend(G_ts[s])
        # print ("falls[s], rewards[s]: ", falls[s], rewards[s])
        tmp_falls.extend(falls[s])
        # print ("exp_actions[",s,"]: ", np.array(exp_actions[s]).shape, repr(exp_actions[s]))
        tmp_exp_actions.extend(exp_actions[s])
        ### Advantage is in a different format (agent , state)
        adv__ = []
        base__ = []
        for a_ in range(len(advantage)):
            adv__.append(advantage[a_][s])
            base__.append(baselines_[a_][s])
        tmp_baselines_.extend(base__)
        tmp_advantage.extend(adv__)
    tmp_advantage = np.array(tmp_advantage)
    
        
    tuples = (tmp_states, tmp_actions, tmp_res_states, tmp_rewards, tmp_falls, tmp_G_ts, tmp_advantage, tmp_exp_actions)
    
    ### Doesn't work with simulations that have multiple state types/definitions
    # if ( len(np.array(tmp_states).shape) == 2):
    if ("perform_multiagent_training" in settings):
        pass
        ### This will be a little complex because agents can have different state dimensions
    elif (not ("use_dual_state_representations" in settings
            and (settings["use_dual_state_representations"] == True))
        ):
        ### consistency checks
        assert np.array(tmp_states).shape == (i_ * len(states[0]), len(model.getStateBounds()[0])), "np.array(tmp_states).shape: " + str(np.array(tmp_states).shape) + " == " + str((i_ * len(states[0]), len(model.getStateBounds()[0])))
        assert np.array(tmp_states).shape == np.array(tmp_res_states).shape, "np.array(tmp_states).shape == np.array(tmp_res_states).shape: " + str(np.array(tmp_states).shape) + " == " + str(np.array(tmp_res_states).shape)
        assert np.array(tmp_rewards).shape == (i_ * len(states[0]), 1), "np.array(tmp_rewards).shape: " + str(np.array(tmp_rewards).shape) + " == " + str((i_ * len(states[0]), 1))
        assert np.array(tmp_rewards).shape == np.array(tmp_falls).shape, "np.array(tmp_rewards).shape == np.array(tmp_falls).shape: " + str(np.array(tmp_rewards).shape) + " == " + str(np.array(tmp_falls).shape)
        assert np.array(tmp_falls).shape == np.array(tmp_G_ts).shape, "np.array(tmp_falls).shape == np.array(tmp_G_ts).shape: " + str(np.array(tmp_falls).shape) + " == " + str(np.array(tmp_G_ts).shape)
        assert np.array(tmp_G_ts).shape == np.array(tmp_advantage).shape, "np.array(tmp_G_ts).shape == np.array(tmp_advantage).shape: " + str(np.array(tmp_G_ts).shape) + " == " + str(np.array(tmp_advantage).shape)
        assert np.array(tmp_advantage).shape == np.array(tmp_exp_actions).shape, "np.array(tmp_advantage).shape == np.array(tmp_exp_actions).shape: " + str(np.array(tmp_advantage).shape) + " == " + str(np.array(tmp_exp_actions).shape)
        assert np.array(tmp_advantage).shape == np.array(tmp_baselines_).shape, "np.array(tmp_advantage).shape == np.array(tmp_baselines_).shape: " + str(np.array(tmp_advantage).shape) + " == " + str(np.array(tmp_baselines_).shape)
        assert np.array(tmp_baselines_).shape == np.array(tmp_discounted_sum).shape, "np.array(tmp_baselines_).shape == np.array(tmp_discounted_sum).shape: " + str(np.array(tmp_baselines_).shape) + " == " + str(np.array(tmp_discounted_sum).shape)
    elif  ("fd_use_multimodal_state" in settings
            and (settings["fd_use_multimodal_state"] == True)):
        pass
    elif  ("append_camera_velocity_state" in settings
            and (settings["append_camera_velocity_state"] == True)):
        pass
    elif  ("use_viz_for_policy" in settings
            and (settings["use_viz_for_policy"] == True)):
        pass
        """
        assert np.array(tmp_states)[:,0,:].shape == (i_ * len(states[0]), len(model.getStateBounds()[0])), "np.array(tmp_states).shape: " + str(np.array(tmp_states).shape) + " == " + str((i_ * len(states[0]), len(model.getStateBounds()[0])))
        assert np.array(tmp_states)[:,0,:].shape == np.array(tmp_res_states)[:,0,:].shape, "np.array(tmp_states).shape == np.array(tmp_res_states).shape: " + str(np.array(tmp_states).shape) + " == " + str(np.array(tmp_res_states).shape)
        assert np.array(tmp_rewards).shape == (i_ * len(states[0]), 1), "np.array(tmp_rewards).shape: " + str(np.array(tmp_rewards).shape) + " == " + str((i_ * len(states[0]), 1))
        assert np.array(tmp_rewards).shape == np.array(tmp_falls).shape, "np.array(tmp_rewards).shape == np.array(tmp_falls).shape: " + str(np.array(tmp_rewards).shape) + " == " + str(np.array(tmp_falls).shape)
        assert np.array(tmp_falls).shape == np.array(tmp_G_ts).shape, "np.array(tmp_falls).shape == np.array(tmp_G_ts).shape: " + str(np.array(tmp_falls).shape) + " == " + str(np.array(tmp_G_ts).shape)
        assert np.array(tmp_G_ts).shape == np.array(tmp_advantage).shape, "np.array(tmp_G_ts).shape == np.array(tmp_advantage).shape: " + str(np.array(tmp_G_ts).shape) + " == " + str(np.array(tmp_advantage).shape)
        assert np.array(tmp_advantage).shape == np.array(tmp_exp_actions).shape, "np.array(tmp_advantage).shape == np.array(tmp_exp_actions).shape: " + str(np.array(tmp_advantage).shape) + " == " + str(np.array(tmp_exp_actions).shape)
        assert np.array(tmp_advantage).shape == np.array(tmp_baselines_).shape, "np.array(tmp_advantage).shape == np.array(tmp_baselines_).shape: " + str(np.array(tmp_advantage).shape) + " == " + str(np.array(tmp_baselines_).shape)
        assert np.array(tmp_baselines_).shape == np.array(tmp_discounted_sum).shape, "np.array(tmp_baselines_).shape == np.array(tmp_discounted_sum).shape: " + str(np.array(tmp_baselines_).shape) + " == " + str(np.array(tmp_discounted_sum).shape)
        """
    else:
        ### dual or multiple state training
        pass
        """
        assert np.array(tmp_states)[:,0].shape == (i_ * len(states[0]), len(model.getStateBounds()[0])), "np.array(tmp_states).shape: " + str(np.array(tmp_states).shape) + " == " + str((i_ * len(states[0]), len(model.getStateBounds()[0])))
        assert np.array(tmp_states)[:,0].shape == np.array(tmp_res_states)[:,0].shape, "np.array(tmp_states).shape == np.array(tmp_res_states).shape: " + str(np.array(tmp_states).shape) + " == " + str(np.array(tmp_res_states).shape)
        assert np.array(tmp_rewards).shape == (i_ * len(states[0]), 1), "np.array(tmp_rewards).shape: " + str(np.array(tmp_rewards).shape) + " == " + str((i_ * len(states[0]), 1))
        assert np.array(tmp_rewards).shape == np.array(tmp_falls).shape, "np.array(tmp_rewards).shape == np.array(tmp_falls).shape: " + str(np.array(tmp_rewards).shape) + " == " + str(np.array(tmp_falls).shape)
        assert np.array(tmp_falls).shape == np.array(tmp_G_ts).shape, "np.array(tmp_falls).shape == np.array(tmp_G_ts).shape: " + str(np.array(tmp_falls).shape) + " == " + str(np.array(tmp_G_ts).shape)
        assert np.array(tmp_G_ts).shape == np.array(tmp_advantage).shape, "np.array(tmp_G_ts).shape == np.array(tmp_advantage).shape: " + str(np.array(tmp_G_ts).shape) + " == " + str(np.array(tmp_advantage).shape)
        assert np.array(tmp_advantage).shape == np.array(tmp_exp_actions).shape, "np.array(tmp_advantage).shape == np.array(tmp_exp_actions).shape: " + str(np.array(tmp_advantage).shape) + " == " + str(np.array(tmp_exp_actions).shape)
        assert np.array(tmp_advantage).shape == np.array(tmp_baselines_).shape, "np.array(tmp_advantage).shape == np.array(tmp_baselines_).shape: " + str(np.array(tmp_advantage).shape) + " == " + str(np.array(tmp_baselines_).shape)
        assert np.array(tmp_baselines_).shape == np.array(tmp_discounted_sum).shape, "np.array(tmp_baselines_).shape == np.array(tmp_discounted_sum).shape: " + str(np.array(tmp_baselines_).shape) + " == " + str(np.array(tmp_discounted_sum).shape)
        """
    
    # print("***** Sim Actions std:  ", np.std((actions), axis=0) )
    # print("***** Sim State mean:  ", np.mean((states), axis=0) )
    # print("***** Sim Next State mean:  ", np.mean((result_states___), axis=0) )

    return (tuples, tmp_discounted_sum, tmp_baselines_, evalData)

# @profile(precision=5)
def simModelParrallel(sw_message_queues, eval_episode_data_queue, model, settings, anchors=None, type=None, p=1):
    import numpy as np
    if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
        print ("Simulating epochs in Parallel:")
    j=0
    timeout_ = 60 * 10 ### 10 min timeout
    if ("simulation_timeout" in settings):
        timeout_ = settings["simulation_timeout"]
        
    if ( 'value_function_batch_size' in settings):
        batch_size=settings["value_function_batch_size"]
    else:
        batch_size=settings["batch_size"]
        
    discounted_values = []
    bellman_errors = []
    reward_over_epocs = []
    mean_discount_error = []
    std_discount_error = []
    
    epoch_=0
    states = []
    actions = []
    result_states = []
    rewards = []
    falls = []
    G_ts = []
    advantage = [] 
    exp_actions = []
    values = []
    evalDatas = []
    i = 0 
    
    if ("num_on_policy_rollouts" in settings):
        min_samples = settings["num_on_policy_rollouts"] * settings["max_epoch_length"]
    else:
        min_samples = settings["epochs"] * settings["max_epoch_length"]
    
    if (   ("anneal_exploration" in settings) 
         and (settings['anneal_exploration'] != False)
         # and (r < (max(float(settings['anneal_exploration']), epsilon * p))) ) 
        ):
        p_ = max(float(settings['anneal_exploration']), settings['epsilon'] * p)
        min_samples = min_samples * (1.0/p_)
    
        if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
            print("Updated min sample from collection is: ", min_samples)
    samples__ = 0
    while ( (samples__ < (min_samples))
            ):
        
        j = 0
        # print("j: ", j)
        # while (j < abs(settings['num_available_threads'])) and ( (i + j) < anchors):
        while (j < abs(settings['num_available_threads'])):
            episodeData = {}
            episodeData['data'] = i
            if ( (type is None) ):
                episodeData['type'] = 'sim_on_policy'
            elif( type == 'eval'):
                episodeData['type'] = 'eval'
            elif ( type == "keep_alive"):
                episodeData['type'] = 'keep_alive'
            elif ( type == "Get_Net_Params"):
                episodeData['type'] = 'Get_Net_Params'
            else:
                episodeData['type'] = 'bootstrapping'
            # sw_message_queues[j].put(episodeData, timeout=timeout_)
            if (settings['on_policy']):
                # print ("sw_message_queues[j].maxsize: ", sw_message_queues[j].qsize() )
                sw_message_queues[j].put(episodeData, timeout=timeout_)
            else:
                sw_message_queues.put(episodeData, timeout=timeout_)
            j += 1
            
        j = 0
        # while (j < abs(settings['num_available_threads'])) and ( (i + j) < anchors):
        datas__ = []
        while (j < abs(settings['num_available_threads'])):
            j += 1
            if ( type == "keep_alive"
                 or type == "Get_Net_Params"):
                dat =  eval_episode_data_queue.get(timeout=timeout_)
                datas__.append(dat)
                continue
            (tuples, discounted_sum_, value_, evalData_) =  eval_episode_data_queue.get(timeout=timeout_)
            discounted_values.append(discounted_sum_)
            values.append(value_)
            evalDatas.append(evalData_)
            """
            simEpoch(actor, exp, 
                    model, discount_factor, anchors=anchs, action_space_continuous=action_space_continuous, 
                    settings=settings, print_data=print_data, p=0.0, validation=True, epoch=epoch_, evaluation=evaluation,
                    visualizeEvaluation=visualizeEvaluation)
            """
            epoch_ = epoch_ + 1
            (states_, actions_, result_states_, rewards_, falls_, G_ts_, advantage_, exp_actions_) = tuples
            samples__ = samples__ + len(states_)
            states.append(states_)
            actions.append(actions_)
            result_states.append(result_states_)
            rewards.append(rewards_)
            falls.append(falls_)
            G_ts.append(G_ts_)
            advantage.append(advantage_)
            exp_actions.append(exp_actions_)
            if( type == 'eval'):
            
                if model.samples() >= batch_size:
                    error = model.bellman_error()
                    # print("Episode bellman error: ", error)
                else :
                    error = [[0]]
                    print ("Error: not enough samples in experience to check bellman error: ", model.samples(), " needed " , batch_size)
                error = np.mean(np.fabs(error))
                # This works better because epochs can terminate early, which is bad.
                # print ("rewards: ", np.array(rewards_).shape)
                rewards__=[]
                discounted_sum__=[]
                value__=[]
                discount_error__ = []
                for agent_ in range(len(model.getAgents())): 
                    rewards__.append(np.array(rewards_).flatten()[agent_::len(model.getAgents())])
                    discounted_sum__.append(np.array(discounted_sum_).flatten()[agent_::len(model.getAgents())])
                    value__.append(np.array(value_).flatten()[agent_::len(model.getAgents())])
                    discount_error__.append(discounted_sum__[agent_] - value__[agent_])
                reward_over_epocs.append(np.mean(np.array(rewards__), axis=1))
                bellman_errors.append(error)
                mean_discount_error.append(np.mean(np.fabs(discount_error__), axis=1))
                std_discount_error.append(np.std(discount_error__, axis=1))
        i += j
        if ( type == "keep_alive"
             or type == "Get_Net_Params"):
            break
        # print("samples collected so far: ", len(states))
    
    if( type == 'eval'):
        
        if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
            print ("Reward for best epoch: " + str(np.argmax(reward_over_epocs)) + " is " + str(np.max(reward_over_epocs)))
            print ("reward_over_epocs" + str(reward_over_epocs))
        if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['debug']):
            print ("Discounted sum: ", np.array(discounted_values))
            print ("Initial values: ", np.array(values))
            for i in range(len(discounted_values)):
                print ("len(discounted_values[",i,"]): ", np.array(discounted_values[i]).shape, " len(values[",i,"]): ", 
                       np.array(values[i]).shape)
            
        mean_reward = np.mean(reward_over_epocs, axis=0)
        std_reward = np.std(reward_over_epocs, axis=0)
        mean_bellman_error = np.mean(bellman_errors)
        std_bellman_error = np.std(bellman_errors)
        mean_discount_error = np.mean(mean_discount_error, axis=0)
        std_discount_error = np.std(std_discount_error, axis=0)
        mean_eval = np.mean(evalDatas)
        std_eval = np.std(evalDatas)
        return (mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error,
            mean_eval, std_eval)
        
    if ( type == "keep_alive"
         or type == "Get_Net_Params"):
        return datas__
    tuples = (states, actions, result_states, rewards, falls, G_ts, advantage, exp_actions)
    return (tuples, discounted_values, values, evalDatas)

# @profile(precision=5)
def simModelMoreParrallel(sw_message_queues, eval_episode_data_queue, model, settings, anchors=None, type=None, p=1):
    import numpy as np
    if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
        print ("Simulating epochs in Parallel:")
        
    if ( 'value_function_batch_size' in settings):
        batch_size=settings["value_function_batch_size"]
    else:
        batch_size=settings["batch_size"]
        
    j=0
    timeout_ = 60 * 10 ### 5 min timeout
    if ("simulation_timeout" in settings):
        timeout_ = settings["simulation_timeout"]
    discounted_values = []
    bellman_errors = []
    reward_over_epocs = []
    
    epoch_=0
    states = []
    actions = []
    result_states = []
    rewards = []
    falls = []
    G_ts = []
    advantage = [] 
    exp_actions = []
    values = []
    evalDatas = []
    i = 0 
    
    if ("num_on_policy_rollouts" in settings):
        min_samples = settings["num_on_policy_rollouts"] * settings["max_epoch_length"]
    else:
        min_samples = settings["epochs"] * settings["max_epoch_length"]
    
    if (   ("anneal_exploration" in settings) 
         and (settings['anneal_exploration'] != False)
         # and (r < (max(float(settings['anneal_exploration']), epsilon * p))) ) 
        ):
        p_ = max(float(settings['anneal_exploration']), settings['epsilon'] * p)
        min_samples = min_samples * (1.0/p_)
    

        anchors
        if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
            print("Updated min sample from collection is: ", min_samples)
    
    if( type == 'eval'): ### for number of eval epochs
        min_samples = anchors * settings["max_epoch_length"]
            
    samples__ = 0
    j=0
    while (j < abs(settings['num_available_threads'])):
        episodeData = {}
        episodeData['data'] = i
        if ( (type is None) ):
            episodeData['type'] = 'sim_on_policy'
        elif( type == 'eval'):
            episodeData['type'] = 'eval'
        elif ( type == "Get_Net_Params"):
            episodeData['type'] = 'Get_Net_Params'
        elif ( type == "keep_alive"):
            episodeData['type'] = 'keep_alive'
        else:
            episodeData['type'] = 'bootstrapping'
        # sw_message_queues[j].put(episodeData, timeout=timeout_)
        if (settings['on_policy'] == True):
            print ("sw_message_queues[j].maxsize: ", sw_message_queues[j].qsize() )
            sw_message_queues[j].put(episodeData, timeout=timeout_)
        else:
            sw_message_queues.put(episodeData, timeout=timeout_)
        j += 1
        # print("j: ", j)
        
    datas__ = []
    while ( (samples__ < (min_samples) or  (j > 0))
            ):
        
        # while (j < abs(settings['num_available_threads'])):
        j = j - 1
        # print("j2: ", j)
        if ( type == "keep_alive"
             or type == "Get_Net_Params"):
            dat =  eval_episode_data_queue.get(timeout=timeout_)
            datas__.append(dat)
            if ( j == 0 ):
                break
            else:
                continue
        (tuples, discounted_sum_, value_, evalData_) =  eval_episode_data_queue.get(timeout=timeout_)
        
        discounted_values.append(discounted_sum_)
        values.append(value_)
        evalDatas.append(evalData_)

        epoch_ = epoch_ + 1
        (states_, actions_, result_states_, rewards_, falls_, G_ts_, advantage_, exp_actions_) = tuples
        samples__ = samples__ + len(states_)
        states.append(states_)
        actions.append(actions_)
        result_states.append(result_states_)
        rewards.append(rewards_)
        falls.append(falls_)
        G_ts.append(G_ts_)
        advantage.append(advantage_)
        exp_actions.append(exp_actions_)
        
        
        if (samples__ < (min_samples)):
            ### If we still need more samples generate another trajectory
            episodeData = {}
            episodeData['data'] = i
            if ( (type is None) ):
                episodeData['type'] = 'sim_on_policy'
            elif( type == 'eval'):
                episodeData['type'] = 'eval'
            elif ( type == "keep_alive"):
                episodeData['type'] = 'keep_alive'
            else:
                episodeData['type'] = 'bootstrapping'
            # sw_message_queues[j].put(episodeData, timeout=timeout_)
            if (settings['on_policy'] == True):
                # print ("sw_message_queues[j].maxsize: ", sw_message_queues[j].qsize() )
                sw_message_queues[j].put(episodeData, timeout=timeout_)
            else:
                sw_message_queues.put(episodeData, timeout=timeout_)
            j = j + 1
            
        # print("j: ", j)
            
        if( type == 'eval'):
            
            if model.samples() >= batch_size:
                error = model.bellman_error()
                # print("Episode bellman error: ", error)
            else :
                error = [[0]]
                print ("Error: not enough samples in experience to check bellman error: ", model.samples(), " needed " , batch_size)
            error = np.mean(np.fabs(error))
            # This works better because epochs can terminate early, which is bad.
            # print ("rewards: ", np.array(rewards_).shape)
            reward_over_epocs.append(np.mean(np.array(rewards_)))
            bellman_errors.append(error)
        
            
        # print("samples collected so far: ", len(states))
    
    assert (j == 0)
    
    
    tuples = (states, actions, result_states, rewards, falls, G_ts, advantage, exp_actions)
    if( type == 'eval'):
        
        if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
            print ("Reward for best epoch: " + str(np.argmax(reward_over_epocs)) + " is " + str(np.max(reward_over_epocs)))
            print ("reward_over_epocs" + str(reward_over_epocs))
        if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['debug']):
            print ("Discounted sum: ", np.array(discounted_values))
            print ("Initial values: ", np.array(values))
            for i in range(len(discounted_values)):
                print ("len(discounted_values[",i,"]): ", np.array(discounted_values[i]).shape, " len(values[",i,"]): ", 
                       np.array(values[i]).shape)
            
        mean_reward = np.mean(reward_over_epocs)
        std_reward = np.std(reward_over_epocs)
        mean_bellman_error = np.mean(bellman_errors)
        std_bellman_error = np.std(bellman_errors)
        mean_discount_error = 0
        std_discount_error = 0
        for d in range(len(discounted_values)):
            mean_discount_error = mean_discount_error + np.mean(np.array(discounted_values[d]) - np.array(values[d]))
            std_discount_error =  std_discount_error + np.std(np.array(discounted_values[d]) - np.array(values[d]))
        mean_discount_error = mean_discount_error / float(len(discounted_values))
        std_discount_error = std_discount_error / float(len(discounted_values))
        mean_eval = np.mean(evalDatas)
        std_eval = np.std(evalDatas)
        return (mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error,
            mean_eval, std_eval)
    elif ( type == "keep_alive"
         or type == "Get_Net_Params"):
        return datas__
    else:
        return (tuples, discounted_values, values, evalDatas)
