

# import cPickle
import dill
import sys
import gc
# from theano.compile.io import Out
sys.setrecursionlimit(50000)
# from sim.PendulumEnvState import PendulumEnvState
# from sim.PendulumEnv import PendulumEnv
import logging
from multiprocessing import Process, Queue
# from pathos.multiprocessing import Pool
import threading
import time
import copy
from model.ModelUtil import *
from util.utils import checkSetting, checkSettingExists
# import memory_profiler
# import resources

log = logging.getLogger(__file__)
    
    
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
    if (settings["action_space_continuous"]):
        model.setNoise(action_bounds[0] * 0.0)
        
    if ( (bootstrapping == True) and 
         (settings["exploration_method"] == "sampling") ):
        settings = copy.deepcopy(settings)
        settings["exploration_method"] = "gaussian_network"
        settings["evalaute_with_MBRL"] = False
        
    if (movieWriter is not None):
        exp.setMovieWriter(movieWriter)
    if (isinstance(settings["discrete_actions"], list)):
        action_selection = range(len(settings["discrete_actions"]))
    else:
        action_selection = range(settings["discrete_actions"])
    # print ("Action selection: " + str(action_selection))   
    reward_bounds = np.array(settings['reward_bounds'] )
    pa=None
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
    """
    if ("llc_index" in settings):
        ### Bad hack for now to use llc in env
        if (settings["environment_type"] == "Multiworld"
            or settings["environment_type"] == "MultiworldHRL"):
            exp.setLLC(model.getAgents()[settings["llc_index"]])
        else:
            exp.getEnvironment().getEnv().setLLC(model.getAgents()[settings["llc_index"]])
    if ("hlc_index" in settings):
        ### Bad hack for now to use hlp in env
        exp.getEnvironment().getEnv().setHLP(model.getAgents()[settings["hlc_index"]])
    """
    if ("replace_entropy_state_with_vae" in settings 
        and (settings["replace_entropy_state_with_vae"])):
        actor.setEncoder(model.getForwardDynamics())
    ### reset the environments
    exp.initEpoch()
    actor.initEpoch()

    state_ = exp.getState()
    viz_q_values_ = []
    
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
    agent_ids = []
    task_ids = []
    result_states___ = []
    exp_actions = []
    evalDatas=[]
    stds=[]
    infos = []
    bad_sim_state = False
    entropy_ = 0
    state_dicts = []
    
    i_ = 0
    if ("use_max_T_annealing" in settings 
        and (settings["use_max_T_annealing"])
        and not evaluation):
        max_length= min(settings['max_epoch_length'], max(int(settings['max_epoch_length']*(1-p)), 4))
    else:
        max_length= settings['max_epoch_length']
#     print ("max_length: ", max_length)
    while (i_ < max_length):
        
#         print ("state_: ", state_)
        action=None
        evaluation__ = evaluation
        if ("use_stochastic_evaluation" in settings):
            evaluation__=False
        (action, exp_action, entropy_, state_) = model.sample(state_, p=p, sim_index=worker_id, bootstrapping=bootstrapping,
                                                epsilon=epsilon, sampling=sampling, time_step=i_, evaluation_=evaluation__)
        outside_bounds=False
        action_=None
        if (("clamp_actions_to_stay_inside_bounds" in settings and settings["clamp_actions_to_stay_inside_bounds"]) or 
        ("penalize_actions_outside_bounds" in settings and (settings['penalize_actions_outside_bounds']))):
            (action_, outside_bounds) = clampActionWarn(action, action_bounds)
            if (settings['clamp_actions_to_stay_inside_bounds']):
                action = action_
        if ("visualize_forward_dynamics" in settings and settings["visualize_forward_dynamics"] and 
            "train_forward_dynamics" in settings and settings['train_forward_dynamics']):
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
                
            
        # print("exp_action: ", exp_action, " action", action)
        if ( "use_hrl_logic" in settings ### Might need to add HLP action to LLP state
        and (settings["use_hrl_logic"]) == "full" ):
            observation, reward_, done, info = actor.step(exp,[action[settings["llc_index"]]])
            # observation, reward_, done, info = actor.step(exp,[action[settings["hlc_index"]]])
        else:
            observation, reward_, done, info = actor.step(exp,action)

#         print("done: ", done)
        infos.append(info)
        if ( "use_hrl_logic" in settings ### Might need to add HLP action to LLP state
        and (settings["use_hrl_logic"]) == "full" ):
            reward_ = model.addHRLReward(state_, observation, reward_, done, info)
#         print("reward_: ", reward_)
        a = 0
        """
        # support for mixing rewards across levels
        if ("hlc_index" in settings
                and "llc_index" in settings
                and "hlc_intrinsic_weight" in settings):
            a = reward_[settings["llc_index"]][0] * settings["hlc_intrinsic_weight"]
        b = 0
        if ("hlc_index" in settings
                and "llc_index" in settings
                and "llc_task_weight" in settings):
            b = reward_[settings["hlc_index"]][0] * settings["llc_task_weight"]
        if ("hlc_index" in settings
                and "llc_index" in settings):
            reward_[settings["hlc_index"]][0] += a
            reward_[settings["llc_index"]][0] += b
        """
        agent_not_fell = actor.hasNotFallen(exp)
        if (outside_bounds and settings['penalize_actions_outside_bounds']):
            ### TODO: this penalty should really be a function of the distance the action was outside the bounds
            reward_ = reward_ + settings['reward_lower_bound']  

        resultState_ = exp.getState()
        # if ( "use_hrl_logic" in settings ### Might need to add HLP action to LLP state
        #      and (settings["use_hrl_logic"]) ):
        #     resultState_ = resultState_.tolist()
        #     resultState_[1] = np.concatenate([resultState_[1], action[0]], axis=-1)
        if (i_ > 0):
#             if ("replace_next_state_with_imitation_viz_state" in settings
#                 and (settings["replace_next_state_with_imitation_viz_state"] == True)):
            result_states___.append(state_)
        states.append(state_)
        states__.extend(state_)
        
        # print ("state_: ", state_)
        # print ("resultState_: ", resultState_)
        
        if ("save_eval_video" in settings  
            and (settings['save_eval_video'] == True)):
            ### If the sim does not have it's own writing support
            from skimage.transform import rescale, resize, downscale_local_mean
            vizData = exp.getEnvironment().render(mode="rgb_array")
#             print ("vizData: ", vizData.shape)
            image_ = np.zeros((vizData.shape))
#             for row in range(len(vizData)):
#                 image_[row] = vizData[len(vizData)-row - 1]
            image_ = np.array(vizData, dtype="uint8")
# #             image_ = resize(image_, (64, 64, 3),
#                            anti_aliasing=True)
            info["rendering"] = image_
            
        if (movieWriter is not None
            and (not exp.movieWriterSupport())):
            ### If the sim does not have it's own writing support
            vizData = exp.getEnvironment().render()
            image_ = np.zeros((vizData.shape))
            for row in range(len(vizData)):
                image_[row] = vizData[len(vizData)-row - 1]
            image_ = np.array(image_, dtype="uint8")
            
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
                reward_ = rewmodel.predict([state_[0][0]], [resultState_[0][0]])[0][0]

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
                    ## How does this work without passing the state??
                    reward__0 = exp.computeImitationReward(rewmodel.predict)
                    reward__1 = exp.computeImitationReward(rewmodel.predict_reward)
                    if "anneal_dual_reward" in settings:
                        reward__ = ((reward__0 * 0.5) + (reward__1 * 0.5 * p))
                    else:
                        reward__ = ((reward__0 * 0.5) + (reward__1 * 0.5))
#                     reward__ = reward__0
#                     print ("reward__: ", reward__, " reward__spatial: ", reward__0, " reward__time: ", reward__1)
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
            
        G_t.append(np.array([[0]])) # *(1.0-discount_factor)))
        for i in range(len(G_t)):
            if isinstance(reward_, (list, tuple, np.ndarray)):
                assert len(np.array(reward_).shape) == 2, "reward shape is " + str(np.array(reward_).shape) + str(reward_) 
            else:
                reward_ = [[reward_]]
        
        ## For testing remove later
        if (checkSettings(settings,"use_back_on_track_forcing") and (not evaluation)):
            exp.getControllerBackOnTrack()
        # print ("reward_: ", reward_)
        if print_data:
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
                if ("MultiworldHRL" == settings["environment_type"] or
                        "MultiworldGoal" == settings["environment_type"]):
                    a = exp._goal
                    b = exp.observation_space.low
                    c = exp.observation_space.high
                    x = (a[0] - b[0]) / (c[0] - b[0])
                    y = (a[1] - b[1]) / (c[1] - b[1])
                    z = (a[2] - b[2]) / (c[2] - b[2])
                    x_size = 500
                    y_size = 500
                    x_var = 1.0
                    y_var = 1.0
                    np.zeros([y_size, x_size, 3])
                    x = x * x_size
                    y = y * y_size
                    z = z * 3
                    y = y_size - y
                    x_coords = np.arange(x_size)
                    y_coords = np.arange(y_size)
                    z_coords = np.arange(3)
                    x_coords, y_coords, z_coords = np.meshgrid(x_coords, y_coords, z_coords)
                    image = np.exp(-1.0 * ((y_coords - y)**2 / y_size / y_var +
                                           (x_coords - x)**2 / x_size / x_var +
                                           (z_coords - z)**2))

                    try:
                        import cv2
                        cv2.imshow("Goal", image)
                        cv2.waitKey(10)
                    except:
                        pass

                if "fd_algorithm" in settings and settings["fd_algorithm"] == "algorithm.VAE.VAE":

                    fd = model.getForwardDynamics()
                    xinputs = state_[0][1]
                    print(xinputs.shape)
                    __x = fd._get_reconstructed_image([[xinputs]])[0]
                    __z = fd._sample_image_from_prior([])[0]
                    __x = __x.reshape(settings["fd_terrain_shape"]) * 0.5 + 0.5
                    __z = __z.reshape(settings["fd_terrain_shape"]) * 0.5 + 0.5
                    __y = xinputs.reshape(settings["fd_terrain_shape"])

                    try:
                        import cv2
                        __x = np.flip(__x, axis=0)
                        __x = np.flip(__x, axis=2)
                        __y = np.flip(__y, axis=0)
                        __y = np.flip(__y, axis=2)
                        __z = np.flip(__z, axis=0)
                        __z = np.flip(__z, axis=2)
                        cv2.imshow("Decoded VAE Image", __x)
                        cv2.imshow("Target VAE Image", __y)
                        cv2.imshow("Prior VAE Image", __z)
                        cv2.waitKey(10)
                    except Exception as e:
                        print(e)
            
        if ("max_ent_rl" in settings
            and (settings["max_ent_rl"] == True)):
            # print ("entropy: ", entropy_)
            maxEnt_w = 0.2
            reward_ = reward_ + (np.array(entropy_)[..., np.newaxis] * maxEnt_w)
        ### I can't just unpack the vector of states here in a multi char sim because the 
        ### Order needs to be preserved for computing the advantage.
        actions.append(action)
        rewards.append(reward_)
        if (worker_id is not None):
            # Pushing working id as fall value for multi task training
            if ("ask_env_for_multitask_id" in settings 
                and (settings["ask_env_for_multitask_id"])):
                worker_id = exp.getTaskID()
                # print ("Task ID: ", worker_id)
                task_ids.append([[worker_id]] * len(state_))
            else:
                task_ids.append([[worker_id]] * len(state_))
        # elif ("perform_multiagent_training" in settings
        #  and (settings["perform_multiagent_training"] > 0)):
        ## Saving agent index for data
        agents_ = []
        for f in range(len(state_)):
            ### This needs to work for multi agent and single policy MultiAgent stuff
            if ("perform_multiagent_training" in settings):
                f = min(f, settings["perform_multiagent_training"]-1)
            agents_.append([f])
        agent_ids.append(agents_)
        # falls.append(falls_)
        # else:
#         print("Pushing actual fall value before : ", agent_not_fell)
            # print("Pushing actual fall value: ", [agent_not_fell] * np.array(state_).shape[0])
        if type(agent_not_fell) is list:
            falls.append(agent_not_fell)
        else:
            falls.append([[agent_not_fell]] * len(state_))
            # falls.append([[agent_not_fell]])
                
        exp_act = exp_action
        exp_actions.append(exp_act)
        # print("exp_actions: ", exp_actions)
        # print("falls: ", falls)
        if ((_output_queue != None) and (not evaluation) and (not bootstrapping)): # for multi-threading
            for state__, act__, res__, rew__, fall__, exp__ in zip (states[-1], actions[-1], result_states___[-1], rewards[-1],  falls[-1], exp_actions[-1]):
                _output_queue.put(([state__], [act__], [res__], [rew__],  [fall__], [[0]], [exp__]), timeout=timeout_)
        
        
        if (not (visualizeEvaluation == None)):
            viz_q_values_.append(model.q_value(state_)[0][0])
            if (len(viz_q_values_)>50):
                 viz_q_values_.pop(0)
            visualizeEvaluation.updateLoss(viz_q_values_, np.zeros(len(viz_q_values_)))
            visualizeEvaluation.redraw()
            # print ("viz_value")
            
        state_num += 1
        pa = None
        i_ += 1
        state_ = copy.deepcopy(resultState_)
#         state_ = resultState_
        
        # print ("reward_: ", reward_)
        if ("replace_next_state_with_imitation_viz_state" in settings
            and (settings["replace_next_state_with_imitation_viz_state"] == True)):
            ### This only works properly in the dual state rep case.
            if ("replace_next_state_with_pose_state" in settings and
                  (settings["replace_next_state_with_pose_state"] == True)):
                ob = np.asarray(exp.getEnvironment().getImitationState())
                ob = ob.flatten()
                resultState_[0][1] = ob
            elif ("use_dual_viz_state_representations" in settings
                  and (settings["use_dual_viz_state_representations"] == True)):
                ### Need agent data for simease net
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
                state_dict = {"pose_agent": state_[0][0],
                              "image_agent": state_[0][1],
                              "image_char": resultState_[0][1],
                              "state_char": copy.deepcopy(resultState_)}
                state_dicts.append(state_dict)
#                 import matplotlib
#                 matplotlib.use('Agg')
#                 import matplotlib.pyplot as plt
#                 # img_ = viewData
#                 img_ = np.reshape(state_[0][1][:2304], (48,48))
#                 img__ = np.reshape(resultState_[0][1][:2304], (48, 48))
#                 fig1 = plt.figure(1)
#                 img__ = np.concatenate((img_, img__), axis=1)
#                 plt.imshow(img__, origin='lower')
#                 fig1.savefig("char_viz_imitation_states_"+str(i_)+".png")
        ### Don't reset during evaluation...
        if (((exp.endOfEpoch() and settings['reset_on_fall'] and ((not evaluation)))
             and (reset_prop_tmp <= reset_prop) ) ### Allow option to collect some full trajectories  
            # or ((reward_ < settings['reward_lower_bound']) and (not evaluation))
                ):
            break

    ### This logic is not perfect yet, It should all sample() again to check if the goal should have been updated after the last action.
    if ( "use_hrl_logic" in settings ### Might need to add HLP action to LLP state
         and (settings["use_hrl_logic"]) == "full"):
        if isinstance(resultState_, np.ndarray):
            resultState_ = resultState_.tolist()
        resultState_ = model.addHRLData(resultState_)
        # print ("resultState_ obs :", resultState_)
        # resultState_[1] = np.concatenate([resultState_[1], action[0]], axis=-1)
    if ( "use_hrl_logic" in settings ### Might need to add HLP action to LLP state
         and (settings["use_hrl_logic"]) ):
        if isinstance(resultState_, np.ndarray):
            resultState_ = resultState_.tolist()
        resultState_[1] = np.concatenate([resultState_[1], action[0]], axis=-1)
    result_states___.append(resultState_)
    
    evalDatas.append(actor.getEvaluationData()/float(settings['max_epoch_length']))
    evalData = [np.mean(evalDatas)]
    G_ts.extend(copy.deepcopy(discounted_rewards(np.array(rewards), discount_factor)))
    discounted_sum = G_ts
    
    ## replace fall with termination flag.
    falls[-1] = ([[False]] * len(state_))
    if print_data:
        print ("Evaluation: ", str(evalData))
        print ("Eval Datas: ", evalDatas, falls) 
    ### Reset before predicting values for trajectory
    model.reset()
    for a in range(len(states[0])):
        path = {}
        ### timestep, agent, state
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
        path['agent_id'] = np.array(np.array(agent_ids[last_epoch_end:])[:,a,:])
        if (worker_id is not None):
            path['task_id'] = np.array(np.array(task_ids[last_epoch_end:])[:,a,:])
        path["terminated"] = False
        ## Append so that we can preserve the paths/trajectory structure.
        if (
            len(rewards[last_epoch_end:]) > 0 and 
             ("force_use_mod_state_for_critic" in settings and 
                 (settings["force_use_mod_state_for_critic"] == True)
                 ) 
             ):
#             paths = compute_advantage_(model, [path], discount_factor, settings['GAE_lambda'])
            adv__ = path["reward"]
            baselines_.append(np.array(path["reward"] * 0))
            advantage.append(np.array(adv__))
        elif (len(rewards[last_epoch_end:]) > 0):
            paths = compute_advantage_(model, [path], discount_factor, settings['GAE_lambda'])
            adv__ = paths["advantage"]
            baselines_.append(np.array(paths["baseline"]))
            advantage.append(np.array(adv__))

#     if ( ('print_level' in settings) and (settings["print_level"]== 'debug') ):
#         adv_r = [ [x, y] for x,y in zip(advantage, G_t_rewards)]
#         R_r = [ [x_r, y_r, z_r] for x_r,y_r,z_r in zip(path['reward'], rewards[last_epoch_end:], G_t)]
#         A_r = [ [x_r, y_r, z_r] for x_r,y_r,z_r in zip(advantage, discounted_rewards(np.array(rewards[last_epoch_end:]), discount_factor), baseline)]
#         print ("last_epoch_end: ", last_epoch_end, " i_ ", i_)
#         print ("Advantage, R: ", adv_r)
#         print ("Rewards: ", R_r)
#         print ("Advantage, discounted Reward, baseline: ", np.array(A_r))

    
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
    otherData = {"agent_id": []}
    for key in info:
        otherData[key] = []
    if (worker_id is not None):
        otherData["task_id"] = []
        
    ### data is in format (state, agent), this "extend" does not work well for multi-agent simulation
    for s in range(len(states)):
        tmp_states.extend(states[s])
        tmp_actions.extend(actions[s])
        if ("replace_next_state_with_imitation_viz_state" in settings
            and (settings["replace_next_state_with_imitation_viz_state"] == True)):
            # print (state_dicts[s])
            tmp_res_states.extend(state_dicts[s]['state_char'])
        else:
            tmp_res_states.extend(result_states___[s])
       
        # import matplotlib
        # matplotlib.use('Agg')
        # import matplotlib.pyplot as plt
        # # img_ = viewData
        # img_ = np.reshape(tmp_states[s][1][:2304], (48,48))
        # img__ = np.reshape(tmp_res_states[s][1][:2304], (48, 48))
        # fig1 = plt.figure(1)
        # img__ = np.concatenate((img_, img__), axis=1)
        # plt.imshow(img__, origin='lower')
        # fig1.savefig("char_viz_imitation_states_end"+str(s)+".png")
        
        tmp_rewards.extend(rewards[s])
        tmp_discounted_sum.extend(discounted_sum[s])
        tmp_G_ts.extend(G_ts[s])
        tmp_falls.extend(falls[s])
        tmp_exp_actions.extend(exp_actions[s])
        otherData["agent_id"].extend(agent_ids[s])
        if (worker_id is not None):
            otherData["task_id"].extend(task_ids[s])
        for key in info:
            try: ## Sometimes there are some weird random keys in the info
                if (type(infos[s][key]) is list):
                    otherData[key].extend(infos[s][key])
                else:
                    otherData[key].extend([infos[s][key]])
            except:
                pass
        ### Advantage is in a different format (agent , state)
        adv__ = []
        base__ = []
        for a_ in range(len(advantage)):
            adv__.append(advantage[a_][s])
            base__.append(baselines_[a_][s])
        tmp_baselines_.extend(base__)
        tmp_advantage.extend(adv__)
    tmp_advantage = np.array(tmp_advantage)

    # print ("otherData:", otherData)
    tuples = (tmp_states, tmp_actions, tmp_res_states, tmp_rewards, tmp_falls, tmp_G_ts, tmp_advantage, tmp_exp_actions, otherData)
    
    ### Doesn't work with simulations that have multiple state types/definitions
    if ("perform_multiagent_training" in settings):
        pass
        ### This will be a little complex because agents can have different state dimensions
    elif (not ("use_dual_state_representations" in settings
            and (settings["use_dual_state_representations"] == True))
        ):
        ### consistency checks
        """
        assert np.array(tmp_states).shape == (i_ * len(states[0]), len(model.getStateBounds()[0])), "np.array(tmp_states).shape: " + str(np.array(tmp_states).shape) + " == " + str((i_ * len(states[0]), len(model.getStateBounds()[0])))
        assert np.array(tmp_states).shape == np.array(tmp_res_states).shape, "np.array(tmp_states).shape == np.array(tmp_res_states).shape: " + str(np.array(tmp_states).shape) + " == " + str(np.array(tmp_res_states).shape)
        assert np.array(tmp_rewards).shape == (i_ * len(states[0]), 1), "np.array(tmp_rewards).shape: " + str(np.array(tmp_rewards).shape) + " == " + str((i_ * len(states[0]), 1))
        assert np.array(tmp_rewards).shape == np.array(tmp_falls).shape, "np.array(tmp_rewards).shape == np.array(tmp_falls).shape: " + str(np.array(tmp_rewards).shape) + " == " + str(np.array(tmp_falls).shape)
        assert np.array(tmp_falls).shape == np.array(tmp_G_ts).shape, "np.array(tmp_falls).shape == np.array(tmp_G_ts).shape: " + str(np.array(tmp_falls).shape) + " == " + str(np.array(tmp_G_ts).shape)
        assert np.array(tmp_G_ts).shape == np.array(tmp_advantage).shape, "np.array(tmp_G_ts).shape == np.array(tmp_advantage).shape: " + str(np.array(tmp_G_ts).shape) + " == " + str(np.array(tmp_advantage).shape)
        assert np.array(tmp_advantage).shape == np.array(tmp_exp_actions).shape, "np.array(tmp_advantage).shape == np.array(tmp_exp_actions).shape: " + str(np.array(tmp_advantage).shape) + " == " + str(np.array(tmp_exp_actions).shape)
        assert np.array(tmp_advantage).shape == np.array(tmp_baselines_).shape, "np.array(tmp_advantage).shape == np.array(tmp_baselines_).shape: " + str(np.array(tmp_advantage).shape) + " == " + str(np.array(tmp_baselines_).shape)
        assert np.array(tmp_baselines_).shape == np.array(tmp_discounted_sum).shape, "np.array(tmp_baselines_).shape == np.array(tmp_discounted_sum).shape: " + str(np.array(tmp_baselines_).shape) + " == " + str(np.array(tmp_discounted_sum).shape)
        """
        pass
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
    
    return (tuples, tmp_discounted_sum, tmp_baselines_, evalData)

# @profile(precision=5)
def simModelParrallel(sw_message_queues, eval_episode_data_queue, model, settings, anchors=None, type=None, p=1):
    import numpy as np
    if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
        log.info("Simulating epochs in Parallel:")
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
    data = []
    i = 0 
    
    if ("num_on_policy_rollouts" in settings):
        min_samples = settings["num_on_policy_rollouts"] * settings["max_epoch_length"]
    else:
        min_samples = settings["epochs"] * settings["max_epoch_length"]
        
    if ("perform_multiagent_training" in settings):
        min_samples = min_samples * settings["perform_multiagent_training"]
    
    if (   ("anneal_exploration" in settings) 
         and (settings['anneal_exploration'] != False)
         # and (r < (max(float(settings['anneal_exploration']), epsilon * p))) ) 
        ):
        p_ = max(float(settings['anneal_exploration']), settings['epsilon'] * p)
        min_samples = min_samples * (1.0/p_)
    
        log.info("Updated min sample from collection is: " + str(min_samples))
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
            (states_, actions_, result_states_, rewards_, falls_, G_ts_, advantage_, exp_actions_, data_) = tuples
            samples__ = samples__ + len(states_)
            states.append(states_)
            actions.append(actions_)
            result_states.append(result_states_)
            rewards.append(rewards_)
            falls.append(falls_)
            G_ts.append(G_ts_)
            advantage.append(advantage_)
            exp_actions.append(exp_actions_)
            data_['falls'] = falls_
            data.append(data_)
                
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
            mean_eval, std_eval, data)
        
    if ( type == "keep_alive"
         or type == "Get_Net_Params"):
        return datas__
    tuples = (states, actions, result_states, rewards, falls, G_ts, advantage, exp_actions, data)
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
    data = []
    i = 0 
    
    if ("num_on_policy_rollouts" in settings):
        min_samples = settings["num_on_policy_rollouts"] * settings["max_epoch_length"]
    else:
        min_samples = settings["epochs"] * settings["max_epoch_length"]
        
    if ("perform_multiagent_training" in settings):
        min_samples = min_samples * settings["perform_multiagent_training"]
    
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
        (states_, actions_, result_states_, rewards_, falls_, G_ts_, advantage_, exp_actions_, data_) = tuples
        samples__ = samples__ + len(states_)
        states.append(states_)
        actions.append(actions_)
        result_states.append(result_states_)
        rewards.append(rewards_)
        falls.append(falls_)
        G_ts.append(G_ts_)
        advantage.append(advantage_)
        exp_actions.append(exp_actions_)
        data_['falls'] = falls_
        data.append(data_)
            
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
        
            
        # print("samples collected so far: ", len(states))
    
    assert (j == 0)
    
    
    tuples = (states, actions, result_states, rewards, falls, G_ts, advantage, exp_actions, data)
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
            mean_eval, std_eval, data)
    elif ( type == "keep_alive"
         or type == "Get_Net_Params"):
        return datas__
    else:
        return (tuples, discounted_values, values, evalDatas)
