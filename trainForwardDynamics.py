
"""
theano.config.device='gpu'
theano.config.mode='FAST_RUN'
theano.config.floatX='float32'
"""

import logging
import numpy as np
import sys
import json
from dill.settings import settings
sys.path.append('../')
import dill
import datetime
import os
from util.SimulationUtil import getAgentName

log = logging.getLogger(os.path.basename(__file__))

def trainForwardDynamics(settings):
    """
    State is the input state and Action is the desired output (y).
    """
    # from model.ModelUtil import *
    
    np.random.seed(23)
    from util.SimulationUtil import setupEnvironmentVariable, setupLearningBackend
    setupEnvironmentVariable(settings)
    setupLearningBackend(settings)
    
    # import theano
    # from theano import tensor as T
    # import lasagne
    from util.SimulationUtil import validateSettings
    from util.SimulationUtil import getDataDirectory
    from util.SimulationUtil import createForwardDynamicsModel, createRLAgent, createEnvironment
    from model.NeuralNetwork import NeuralNetwork
    from util.ExperienceMemory import ExperienceMemory
    import matplotlib.pyplot as plt
    import math
    # from ModelEvaluation import *
    # from util.SimulationUtil import *
    import time  
    
    settings = validateSettings(settings)
    
    # anchor_data_file = open(settings["anchor_file"])
    # _anchors = getAnchors(anchor_data_file)
    # print ("Length of anchors epochs: ", str(len(_anchors)))
    # anchor_data_file.close()
    train_forward_dynamics=True
    model_type= settings["model_type"]
    directory= getDataDirectory(settings)
        
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    if "." in settings['forward_dynamics_model_type']:
        ### convert . to / and copy file over
        file_name = settings['forward_dynamics_model_type']
        k = file_name.rfind(".")
        file_name = file_name[:k]
        file_name_read = file_name.replace(".", "/")
        file_name_read = file_name_read + ".py"
        print ("model file name:", file_name)
        print ("os.path.basename(file_name): ", os.path.basename(file_name))
        file = open(file_name_read, 'r')
        out_file = open(directory+file_name+".py", 'w')
        out_file.write(file.read())
        file.close()
        out_file.close()
            
    discrete_actions = np.array(settings['discrete_actions'])
    num_actions= discrete_actions.shape[0] # number of rows
    rounds = settings["rounds"]
    epochs = settings["epochs"]
    # num_states=settings["num_states"]
    epsilon = settings["epsilon"]
    discount_factor=settings["discount_factor"]
    # max_reward=settings["max_reward"]
    reward_bounds = np.array(settings['reward_bounds'])
    batch_size=settings["batch_size"]
    if ("value_function_batch_size" in settings):
        batch_size = settings['value_function_batch_size']
    train_on_validation_set=settings["train_on_validation_set"]
    state_bounds = np.array(settings['state_bounds'])
    discrete_actions = np.array(settings['discrete_actions'])
    # c = characterSim.Configuration(str(settings["sim_config_file"]))
    # c = characterSim.Configuration("../data/epsilon0Config.ini")
    action_space_continuous=settings['action_space_continuous']
    # states2 = np.transpose(np.repeat([states], 2, axis=0))
    # print states2
    if action_space_continuous:
        action_bounds = np.array(settings["action_bounds"], dtype=float)
    print ("Sim config file name: ", str(settings["sim_config_file"]))
    
    if ((state_bounds == "ask_env")):
        exp_val = createEnvironment(settings["sim_config_file"], settings['environment_type'], settings, render=settings['shouldRender'], index=0)
        # exp_val.setActor(actor)
        exp_val.getActor().init()
        exp_val.init()
        print ("Getting state bounds from environment")
        s_min = exp_val.getEnvironment().observation_space.getMinimum()
        s_max = exp_val.getEnvironment().observation_space.getMaximum()
        print (exp_val.getEnvironment().observation_space.getMinimum())
        settings['state_bounds'] = [s_min,s_max]
        state_bounds = settings['state_bounds']
        print ("Removing extra environment.")
        exp_val.finish()
    
    print ("state_bounds: ", state_bounds)
    print ("action_bounds: ", action_bounds)
    
    if action_space_continuous:
        experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), settings['experience_length'], continuous_actions=True, settings=settings)
    else:
        experience = ExperienceMemory(len(state_bounds[0]), 1, settings['experience_length'])
    experience.setSettings(settings)
    if ("keep_seperate_fd_exp_buffer" in settings
        and (settings["keep_seperate_fd_exp_buffer"] == True)):
        file_name=directory+getAgentName()+"_FD_expBufferInit.hdf5"
    else:
        file_name=directory+getAgentName()+"_expBufferInit.hdf5"
        
    # experience.saveToFile(file_name)
    experience.loadFromFile(file_name)
    state_bounds = experience._state_bounds
    print ("Samples in experience: ", experience.samples())
    
    res_state_bounds__ = state_bounds
    if ("use_dual_state_representations" in settings
        and (settings["use_dual_state_representations"] == True)):
        """
        res_state_bounds__ = np.array([[-1] * settings["dense_state_size"], 
                             [1] * settings["dense_state_size"]])
        experience.setResultStateBounds(res_state_bounds__)
        """
        # _states, _actions, _result_states, _rewards, _falls, _G_ts, exp_actions__, _advantage = experience.get_batch(min(experience.samples(), settings["experience_length"]))
        
        ### Usually the state and next state are the same size, not in this case...
        # s_mean_ = np.mean(_states, axis=0)
        # s_std_ = np.std(_states, axis=0) + 1.1 ### hack to avoid zeros
        # s_state_bounds__ = np.array([s_mean_ - s_std_, 
        #                      s_mean_ + s_std_])
        # print("s_state_bounds__: ", s_state_bounds__)

        # experience.setStateBounds(s_state_bounds__)
        print ("state bounds: ", experience.getStateBounds())
        
        # res_mean_ = np.mean(_result_states, axis=0)
        # res_std_ = np.std(_result_states, axis=0) + 0.1 ### hack to avoid zeros
        # res_state_bounds__ = np.array([res_mean_ - res_std_, 
        #                      res_mean_ + res_std_])
        # print ("result state_bounds: ", res_state_bounds__)
        # experience.setResultStateBounds(res_state_bounds__)
        
        # print("res_state_bounds__: ", np.array(res_state_bounds__).shape)
        
    
    if ( settings['forward_dynamics_model_type'] == "SingleNet"):
        print ("Creating forward dynamics network: Using single network model")
        # model = createRLAgent(settings['agent_name'], state_bounds, discrete_actions, reward_bounds, settings)
        model = createRLAgent(settings['agent_name'], state_bounds, discrete_actions, reward_bounds, settings, print_info=True)
        forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, None, None, agentModel=model,
                                                          reward_bounds=reward_bounds)
        # forwardDynamicsModel.setResultStateBounds(res_state_bounds__)
        # forwardDynamicsModel = model
    else:
        print ("Creating forward dynamics network")
        # forwardDynamicsModel = ForwardDynamicsNetwork(state_length=len(state_bounds[0]),action_length=len(action_bounds[0]), state_bounds=state_bounds, action_bounds=action_bounds, settings_=settings)
        forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, None, None, agentModel=None,
                                                          reward_bounds=reward_bounds)
    if settings['visualize_learning']:
        from NNVisualize import NNVisualize
        title = file_name = settings['forward_dynamics_model_type']
        k = title.rfind(".") + 1
        if (k > len(title)): ## name does not contain a .
            k = 0 
        file_name = file_name[k:]
        nlv = NNVisualize(title=str("Forward Dynamics Model") + " with " + str(file_name))
        nlv.setInteractive()
        nlv.init()
    if (settings['train_reward_predictor']):
        if settings['visualize_learning']:
            rewardlv = NNVisualize(title=str("Reward Model") + " with " + str(settings["model_type"]), settings=settings)
            rewardlv.setInteractive()
            rewardlv.init()
    
    
    forwardDynamicsModel.setStateBounds(experience.getStateBounds())
    forwardDynamicsModel.setResultStateBounds(res_state_bounds__)
    
    print ("FD state bounds2: ", forwardDynamicsModel.getStateBounds())
    print ("FD state bounds2 shape: ", np.array(forwardDynamicsModel.getStateBounds()).shape)
    # experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), experience_length, continuous_actions=True)
    """
    for i in range(experience_length):
        action_ = np.array([actions[i]])
        state_ = np.array([states[i]])
        # print "Action: " + str([actions[i]])
        experience.insert(norm_state(state_, state_bounds), norm_action(action_, action_bounds),
                           norm_state(state_, state_bounds), norm_reward(np.array([0]), reward_bounds))
    """
    trainData = {}
    trainData["mean_reward"]=[]
    trainData["std_reward"]=[]
    trainData["mean_bellman_error"]=[]
    trainData["std_bellman_error"]=[]
    trainData["mean_discount_error"]=[]
    trainData["std_discount_error"]=[]
    trainData["mean_forward_dynamics_loss"]=[]
    trainData["std_forward_dynamics_loss"]=[]
    trainData["mean_forward_dynamics_reward_loss"]=[]
    trainData["std_forward_dynamics_reward_loss"]=[]
    trainData["mean_eval"]=[]
    trainData["std_eval"]=[]
    lstm_batch_size_fd=4
    if ("lstm_batch_size" in settings):
        lstm_batch_size_fd=settings["lstm_batch_size"][0]
    # dynamicsLosses=[]
    best_dynamicsLosses=1000000
    _states, _actions, _result_states, _rewards, _falls, _G_ts, exp_actions__, _advantage = experience.get_batch(batch_size)
    # states, actions, result_states, rewards, falls, G_ts, exp_actions, advantage
    """
    _states = theano.shared(np.array(_states, dtype=theano.config.floatX))
    _actions = theano.shared(np.array(_actions, dtype=theano.config.floatX))
    _result_states = theano.shared(np.array(_result_states, dtype=theano.config.floatX))
    _rewards = theano.shared(np.array(_rewards, dtype=theano.config.floatX))
    """
    for round_ in range(rounds):
        t0 = time.time()
        for epoch in range(epochs):
            if ( "model_perform_batch_training" in settings
                 and (settings["model_perform_batch_training"] == True)):
                samps = min(experience.samples(), settings["experience_length"])
                print ("samps: ", samps)
                _states, _actions, _result_states, _rewards, _falls, _G_ts, exp_actions__, _advantage = experience.get_batch(samps)
                dynamicsLoss = forwardDynamicsModel.train(_states, _actions, _result_states, _rewards, updates=1, batch_size=settings["batch_size"])
            else:
                if ((("train_LSTM_FD" in settings)
                    and (settings["train_LSTM_FD"] == True))
                    or
                    (("train_LSTM_Reward" in settings)
                    and (settings["train_LSTM_Reward"] == True))
                    ):
                    state_, action_, resultState_, reward_, fall_, G_ts_, exp_actions, advantage_ = experience.get_multitask_trajectory_batch(batch_size=lstm_batch_size_fd)
                    dynamicsLoss = forwardDynamicsModel.train(states=state_, actions=action_, result_states=resultState_, rewards=reward_)
                    if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
                        print ("Forward Dynamics Loss: ", dynamicsLoss)
                    
                    if (type(settings["sim_config_file"]) == list):
                        if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
                                print ("Additional Multi-task training: ")
                        state_, action_, resultState_, reward_, fall_, G_ts_, exp_actions, advantage_ = experience.get_multitask_trajectory_batch(batch_size=lstm_batch_size_fd)
                        dynamicsLoss = forwardDynamicsModel.train(states=state_, actions=action_, result_states=resultState_, rewards=reward_, falls=fall_)
                        if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
                            print ("Forward Dynamics Loss: ", dynamicsLoss)
                    
                    if (("train_LSTM_FD" in settings)
                        and (settings["train_LSTM_FD"] == False)):   
                        for k in range(0):  
                            _states, _actions, _result_states, _rewards, _falls, _G_ts, exp_actions__, _advantage = experience.get_batch(batch_size)
                            # print("result state shape: ", np.asarray(_result_states).shape)
                            fdloss = forwardDynamicsModel.train(_states, _actions, _result_states, _rewards, lstm=False)
                            if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
                                print ("FD Loss: ", fdloss)
                    
                else:
                    _states, _actions, _result_states, _rewards, _falls, _G_ts, exp_actions__, _advantage = experience.get_batch(batch_size)
                    # print("result state shape: ", np.asarray(_result_states).shape)
                    dynamicsLoss = forwardDynamicsModel.train(_states, _actions, _result_states, _rewards)
                    # print ("dynamicsLoss: ", dynamicsLoss)
                    # reg_loss = forwardDynamicsModel._get_fd_regularization([])
                    # print("regularization_loss: ", reg_loss)
                if (False):
                    import matplotlib
                    # matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    # img_ = np.reshape(viewData, (150,158,3))
                    img_ = _states[0]
                    img_ = np.reshape(img_[:1024], newshape=(32, 32))
                    noise = np.random.normal(loc=0, scale=0.02, size=img_.shape)
                    
                    img_ = img_ + noise
                    print("img_ shape", img_.shape, " sum: ", np.sum(img_))
                    fig1 = plt.figure(1)
                    plt.imshow(img_, origin='lower')
                    plt.title("visual Data: ")
                    # fig1.savefig("viz_state_"+str(i)+".svg")
            # dynamicsLoss = forwardDynamicsModel._train()
        t1 = time.time()
        if (round_ % settings['plotting_update_freq_num_rounds']) == 0:
            if (("train_LSTM_FD" in settings)
                and (settings["train_LSTM_FD"] == True)):
                state_, action_, resultState_, reward_, fall_, G_ts_, exp_actions, advantage_ = experience.get_multitask_trajectory_batch(batch_size=min(experience.samplesTrajectory(),lstm_batch_size_fd))
                dynamicsLoss_ = forwardDynamicsModel.bellman_error(state_, action_, resultState_, reward_)
            else:
                # print ("_states shape: ", _states.shape)
                dynamicsLoss_ = forwardDynamicsModel.bellman_error(_states, _actions, _result_states, _rewards)
                # print ("dynamicsLoss_: ", dynamicsLoss_)
            # dynamicsLoss_ = forwardDynamicsModel.bellman_error((_states), (_actions), (_result_states))
            if ( "use_stochastic_forward_dynamics" in settings 
                 and (settings['use_stochastic_forward_dynamics'] == True)):
                dynamicsLoss = np.mean(dynamicsLoss_)
            else:
                dynamicsLoss = np.mean(np.fabs(dynamicsLoss_))
            if (settings['train_reward_predictor']):
                if (("train_LSTM_Reward" in settings)
                    and (settings["train_LSTM_Reward"] == True)):
                    state_, action_, resultState_, reward_, fall_, G_ts_, exp_actions, advantage_ = experience.get_multitask_trajectory_batch(batch_size=min(experience.samplesTrajectory(),lstm_batch_size_fd))
                    dynamicsRewardLoss_ = forwardDynamicsModel.reward_error(state_, action_, resultState_, reward_)
                else:
                    dynamicsRewardLoss_ = forwardDynamicsModel.reward_error(_states, _actions, _result_states, _rewards)
                dynamicsRewardLoss = np.mean(np.fabs(dynamicsRewardLoss_))
                # dynamicsRewardLosses.append(dynamicsRewardLoss)
                dynamicsRewardLosses = dynamicsRewardLoss
                
            # dynamicsLosses.append(dynamicsLoss)
            mean_dynamicsLosses = dynamicsLoss
            std_dynamicsLosses = np.std((dynamicsLoss_))
            trainData["mean_forward_dynamics_loss"].append(mean_dynamicsLosses)
            trainData["std_forward_dynamics_loss"].append(std_dynamicsLosses)
            if (settings['train_reward_predictor']):
                log.info("\n\nRound: {}, Epoch: {}, ForwardPredictionLoss: {}, Reward Prediction Loss: {}. in {:.1f} seconds\n===\n".format(
                    round_, epoch, dynamicsLoss, dynamicsRewardLosses, datetime.timedelta(seconds=(t1-t0))))
            else:
                log.info("\n\nRound: {}, Epoch: {}, ForwardPredictionLoss: {}, in {:.1f} seconds\n===\n".format(
                    round_, epoch, dynamicsLoss, datetime.timedelta(seconds=(t1-t0))))
            # print ("State Bounds: ", forwardDynamicsModel.getStateBounds(), " exp: ", experience.getStateBounds())
            # print ("Action Bounds: ", forwardDynamicsModel.getActionBounds(), " exp: ", experience.getActionBounds())
            # print (str(datetime.timedelta(seconds=(t1-t0))))
            if (settings['visualize_learning']):
                nlv.updateLoss(np.array(trainData["mean_forward_dynamics_loss"]), np.array(trainData["std_forward_dynamics_loss"]))
                nlv.redraw()
                nlv.setInteractiveOff()
                nlv.saveVisual(directory+"trainingGraphNN")
                nlv.setInteractive()
            if (settings['train_reward_predictor']):
                mean_dynamicsRewardLosses = np.mean(dynamicsRewardLoss)
                std_dynamicsRewardLosses = np.std(dynamicsRewardLoss_)
                dynamicsRewardLosses = []
                trainData["mean_forward_dynamics_reward_loss"].append(mean_dynamicsRewardLosses)
                trainData["std_forward_dynamics_reward_loss"].append(std_dynamicsRewardLosses)
            if (settings['train_reward_predictor'] and settings['visualize_learning']):
                rewardlv.updateLoss(np.array(trainData["mean_forward_dynamics_reward_loss"]), np.array(trainData["std_forward_dynamics_reward_loss"]))
                rewardlv.redraw()
                rewardlv.setInteractiveOff()
                rewardlv.saveVisual(directory+"rewardTrainingGraph")
                rewardlv.setInteractive()
            save_embedding = True
            if (save_embedding == True):
                encoding = {}
                encoding['class'] = []
                encoding['class2'] = []
                encoding['code'] = []
                encoding['code2'] = []
                state_, action_, resultState_, reward_, fall_, G_ts_, exp_actions, advantage_ = experience.get_trajectory_batch(batch_size=min(experience.history_size_Trajectory(), experience.samplesTrajectory()), cast=False)
                """
                for l in range(len(state_)):
                    forwardDynamicsModel.reset()
                    h_a = forwardDynamicsModel.predict_reward_encoding(state_[l])
                    forwardDynamicsModel.reset()
                    h_a_2 = forwardDynamicsModel.predict_reward_encoding(resultState_[l])
                    # print ("state: ", state_[l])
                    clas = settings["worker_to_task_mapping"][fall_[l][0][0]]
                    # print ("Encoding ", l, ": ", h_a)
                    # print ("class: ", clas)
                    encoding['class'].append(clas)
                    encoding['class2'].append(clas)
                    encoding['code'].append([float(i) for i in h_a[0]])
                    encoding['code2'].append([float(i) for i in h_a_2[0]])
                
                tsne_data = open("tsne_data.json", "w")
                json.dump(encoding, tsne_data)
                tsne_data.close()
                """
                
        if (round_ % settings['saving_update_freq_num_rounds']) == 0:
            if mean_dynamicsLosses < best_dynamicsLosses:
                best_dynamicsLosses = mean_dynamicsLosses
                print ("Saving BEST current forward dynamics model: " + str(best_dynamicsLosses))
                file_name_dynamics=directory+"forward_dynamics_Best_pretrain"
                forwardDynamicsModel.saveTo(file_name_dynamics)
            print ("Saving current forward dynamics model: " + str(mean_dynamicsLosses))
            file_name_dynamics=directory+"forward_dynamics_pretrain"
            forwardDynamicsModel.saveTo(file_name_dynamics)
            if settings['save_trainData']:
                fp = open(directory+"FD_trainingData_" + str(settings['agent_name']) + ".json", 'w')
                # print ("Train data: ", trainData)
                ## because json does not serialize np.float32 
                for key in trainData:
                    trainData[key] = [float(i) for i in trainData[key]]
                json.dump(trainData, fp)
                fp.close()
                # draw data
        # print "Error: " + str(error)
    return trainData

if __name__ == '__main__':
    
    settingsFileName = sys.argv[1]

    from util.simOptions import getOptions
    
    options = getOptions(sys.argv)
    options = vars(options)
    
    file = open(options['configFile'])
    settings = json.load(file)
    file.close()

        
    for option in options:
        if ( not (options[option] is None) ):
            print ("Updateing option: ", option, " = ", options[option])
            settings[option] = options[option]
            if ( options[option] == 'true'):
                settings[option] = True
            elif ( options[option] == 'false'):
                settings[option] = False
    # settings['num_available_threads'] = options['num_available_threads']

    
    trainForwardDynamics(settings)
    
