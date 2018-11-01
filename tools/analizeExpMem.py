
"""
theano.config.device='gpu'
theano.config.mode='FAST_RUN'
theano.config.floatX='float32'
"""

import numpy as np
import sys
import json
sys.path.append('../')
import dill
import datetime
import os
from util.SimulationUtil import getAgentName

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
        experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), settings['expereince_length'], continuous_actions=True, settings=settings)
    else:
        experience = ExperienceMemory(len(state_bounds[0]), 1, settings['expereince_length'])
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
        _states, _actions, _result_states, _rewards, _falls, _G_ts, exp_actions__, _advantage = experience.get_batch(min(experience.samples(), settings["expereince_length"]))
        
        ### Usually the state and next state are the same size, not in this case...
        # s_mean_ = np.mean(_states, axis=0)
        # s_std_ = np.std(_states, axis=0) + 1.1 ### hack to avoid zeros
        # s_state_bounds__ = np.array([s_mean_ - s_std_, 
        #                      s_mean_ + s_std_])
        # print("s_state_bounds__: ", s_state_bounds__)

        # experience.setStateBounds(s_state_bounds__)
        print ("state bounds: ", experience.getStateBounds())
        
        res_mean_ = np.mean(_result_states, axis=0)
        res_std_ = np.std(_result_states, axis=0) + 0.1 ### hack to avoid zeros
        res_state_bounds__ = np.array([res_mean_ - res_std_, 
                             res_mean_ + res_std_])
        # print ("result state_bounds: ", res_state_bounds__)
        experience.setResultStateBounds(res_state_bounds__)
        
        print("res_state_bounds__: ", np.array(res_state_bounds__).shape)
        
    
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
    lstm_batch_size=4
    if ("lstm_batch_size" in settings):
        lstm_batch_size=settings["lstm_batch_size"][0]
    # dynamicsLosses=[]
    best_dynamicsLosses=1000000
    _states, _actions, _result_states, _rewards, _falls, _G_ts, exp_actions__, _advantage = experience.get_batch(batch_size)
    # states, actions, result_states, rewards, falls, G_ts, exp_actions, advantage

    print ("States: ", _states)
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
    