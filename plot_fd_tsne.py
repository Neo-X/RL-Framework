
"""
theano.config.device='gpu'
theano.config.mode='FAST_RUN'
theano.config.floatX='float32'
"""

import numpy as np
import sys
import json
from dill.settings import settings
sys.path.append('../')
import dill
import datetime
import os
from util.SimulationUtil import getAgentName

def trainForwardDynamics(settings):
    """
    State is the input state and Action is the desired output (y).
    """
    print ("settings: ", settings)
    # from model.ModelUtil import *
    
    np.random.seed(23)
    from util.SimulationUtil import setupEnvironmentVariable, setupLearningBackend
    setupEnvironmentVariable(settings)
    setupLearningBackend(settings)
    
    # import theano
    # from theano import tensor as T
    # import lasagne
    from util.SimulationUtil import validateSettings, getFDStateSize
    from util.SimulationUtil import getDataDirectory
    from util.SimulationUtil import createForwardDynamicsModel, createRLAgent, createEnvironment
    from util.SimulationUtil import createNewFDModel
    from model.NeuralNetwork import NeuralNetwork
    from util.ExperienceMemory import ExperienceMemory
    import matplotlib.pyplot as plt
    import math
    # from ModelEvaluation import *
    # from util.SimulationUtil import *
    import time  
    
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
        
    state_bounds = getFDStateSize(settings)
    
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
        
    print ("state_bounds: ", state_bounds)
    runLastModel = True
    settings["load_saved_model"] = True
    forwardDynamicsModel = createNewFDModel(settings, None, None)

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
    encoding = {}
    encoding['class'] = []
    encoding['class2'] = []
    encoding['code'] = []
    encoding['code2'] = []
    ### Get all the trajectories
    state_, action_, resultState_, reward_, fall_, G_ts_, exp_actions, advantage_ = experience.get_trajectory_batch(batch_size=min(experience.history_size_Trajectory(), experience.samplesTrajectory()), cast=False)
    for l in range(len(state_)):
        forwardDynamicsModel.reset()
        h_a = forwardDynamicsModel.predict_reward_encoding(state_[l])
        forwardDynamicsModel.reset()
        h_a_2 = forwardDynamicsModel.predict_reward_encoding(resultState_[l])
        # print ("state: ", state_[l])
        #### Lazy hack, the number of types is greater than the number of threads, now.
        if (  "ask_env_for_multitask_id" in settings
              and (settings["ask_env_for_multitask_id"])):
            clas = fall_[l][0][0]
        else:
            clas = settings["worker_to_task_mapping"][fall_[l][0][0]]
        # print ("fall_[l][0][0]: ", fall_[l][0][0])
        # print ("Encoding ", l, ": ", h_a)
        print ("class: ", clas)
        # encoding['class'].append(0)
        # encoding['class2'].append(1)
        encoding['class'].append(int(clas))
        encoding['class2'].append(int(clas))
        encoding['code'].append([float(i) for i in h_a[0]])
        encoding['code2'].append([float(i) for i in h_a_2[0]])
    
    tsne_data = open("tsne_data.json", "w")
    json.dump(encoding, tsne_data)
    tsne_data.close()
                

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
            print ("Updating option: ", option, " = ", options[option])
            settings[option] = options[option]
            if ( options[option] == 'true'):
                settings[option] = True
            elif ( options[option] == 'false'):
                settings[option] = False
    # settings['num_available_threads'] = options['num_available_threads']

    
    trainForwardDynamics(settings)
    