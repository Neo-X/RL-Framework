import random
import numpy as np
import math
import dill
import json
import os
import sys


def modelSampling(settings):
    # make a color map of fixed colors
    #try: 
        # Normalization constants for data
        
        from model.ModelUtil import getSettings
        # settings = getSettings(settings_file_name)
        # settings['shouldRender'] = True
        import os    
        os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device="+settings['training_processor_type']+",floatX="+settings['float_type']
        
        ## Theano needs to be imported after the flags are set.
        # from ModelEvaluation import *
        # from model.ModelUtil import *
        from ModelEvaluation import SimWorker, evalModelParrallel, collectExperience, evalModel
        # from model.ModelUtil import validBounds
        from model.LearningAgent import LearningAgent, LearningWorker
        from util.SimulationUtil import validateSettings, createEnvironment, createRLAgent, createActor
        from util.SimulationUtil import getDataDirectory, createForwardDynamicsModel, createSampler
        
        
        from util.ExperienceMemory import ExperienceMemory
        from RLVisualize import RLVisualize
        from NNVisualize import NNVisualize
                
        print "Sim config file name: " + str(settings["sim_config_file"])
        # c = characterSim.Configuration("../data/epsilon0Config.ini")
        action_space_continuous=settings['action_space_continuous']
        state_bounds = np.array(settings['state_bounds'])
        discrete_actions = np.array(settings['discrete_actions'])
        print ("Sim config file name: ", str(settings["sim_config_file"]))
        # c = characterSim.Configuration(str(settings["sim_config_file"]))
        # c = characterSim.Configuration("../data/epsilon0Config.ini")
        action_space_continuous=settings['action_space_continuous']
        if action_space_continuous:
            action_bounds = np.array(settings["action_bounds"], dtype=float)
            
        if action_space_continuous:
            experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), settings['expereince_length'], continuous_actions=True, settings=settings)
        else:
            experience = ExperienceMemory(len(state_bounds[0]), 1, settings['expereince_length'])
        ### Using a wrapper for the type of actor now
        actor = createActor(str(settings['environment_type']),settings, experience)
        # this is the process that selects which game to play
        exp = None
        
        data_folder = getDataDirectory(settings)
        
        sampler = createSampler(settings, exp)
            
        # if (settings['use_actor_policy_action_suggestion']):
        # file_name=data_folder+"pendulum_agent_"+str(settings['agent_name'])+"_Best.pkl"
        # model = dill.load(open(file_name))
        
        if (settings['train_forward_dynamics']):
            file_name_dynamics=directory+"forward_dynamics_"+str(settings['agent_name'])+"_Best.pkl"
            # file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+".pkl"
            f = open(file_name_dynamics, 'r')
            forwardDynamicsModel = dill.load(f)
            f.close()
            sampler.setForwardDynamics(forwardDynamicsModel)
        else:
            
            forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, actor, exp)
            forwardDynamicsModel.initEpoch()
            sampler.setForwardDynamics(forwardDynamicsModel)
            
        
        # sampler.setPolicy(model)
        sampler.setSettings(settings)
        
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
            
        exp = createEnvironment(str(settings["sim_config_file"]), str(settings['environment_type']), settings, render=True)
        sampler.setEnvironment(exp)
        exp.getActor().init()   
        exp.getEnvironment().init()

        mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error = evalModel(actor, exp, sampler, settings["discount_factor"], 
                                                anchors=settings['eval_epochs'], action_space_continuous=action_space_continuous, settings=settings, print_data=True, 
                                                bootstrapping=True)

        print "Average Reward: " + str(mean_reward)
    #except Exception, e:
    #    print "Error: " + str(e)
    #    raise e
    
if __name__ == "__main__":
    
    file = open(sys.argv[1])
    settings = json.load(file)
    file.close()
    modelSampling(settings)
    