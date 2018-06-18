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
        if ("learning_backend" in settings):
            # KERAS_BACKEND=tensorflow
            os.environ['KERAS_BACKEND'] = settings['learning_backend']
        import keras
        import theano
        keras.backend.set_floatx(settings['float_type'])
        print ("K.floatx()", keras.backend.floatx())
        print ("theano.config.floatX", theano.config.floatX)
        
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
                
        directory= getDataDirectory(settings)
        print ("Sim config file name: ", str(settings["sim_config_file"]))
        # c = characterSim.Configuration("../data/epsilon0Config.ini")
        action_space_continuous=settings['action_space_continuous']
        state_bounds = np.array(settings['state_bounds'])
        discrete_actions = np.array(settings['discrete_actions'])
        print ("Sim config file name: ", str(settings["sim_config_file"]))
        reward_bounds=np.array(settings["reward_bounds"])
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
        
        agent = LearningAgent(settings_=settings)
            
        agent.setSettings(settings)
        
            
        # if (settings['use_actor_policy_action_suggestion']):
        # file_name=data_folder+getAgentName()+"_Best.pkl"
        # model = dill.load(open(file_name))
        settings["load_saved_model"] = True
        # settings["load_saved_model"] = "network_and_scales"
        model = createRLAgent(settings['agent_name'], state_bounds, discrete_actions, reward_bounds, settings)
        settings["load_saved_model"] = False
        print ("State Length: ", len(model.getStateBounds()[0]) )
        agent.setPolicy(model)
            
        
        # sampler.setPolicy(model)
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
            
        exp = createEnvironment(str(settings["sim_config_file"]), str(settings['environment_type']), settings, render=True)
        agent.setEnvironment(exp)
        exp.getActor().init()   
        exp.init()
        
        if (settings['train_forward_dynamics']):
            file_name_dynamics=directory+"forward_dynamics"+"_Best.pkl"
            # file_name=directory+getAgentName()+".pkl"
            f = open(file_name_dynamics, 'r')
            forwardDynamicsModel = dill.load(f)
            f.close()
            agent.setForwardDynamics(forwardDynamicsModel)
            """
        else:
            
            forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, actor, exp, agentModel=None, print_info=True)
            # forwardDynamicsModel.initEpoch(exp)
            agent.setForwardDynamics(forwardDynamicsModel)
        """
        if ( settings['use_simulation_sampling'] ):
            
            sampler = createSampler(settings, exp)
            ## This should be some kind of copy of the simulator not a network
            forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, actor, exp, agentModel=None, print_info=True)
            sampler.setForwardDynamics(forwardDynamicsModel)
            # sampler.setPolicy(model)
            agent.setSampler(sampler)
            sampler.setEnvironment(exp)
            # forwardDynamicsModel.initEpoch()
            print ("thread together exp: ", sampler._exp)

        expected_value_viz=None
        if (settings['visualize_expected_value']):
            expected_value_viz = NNVisualize(title=str("Expected Value") + " with " + str(settings["model_type"]), settings=settings)
            expected_value_viz.setInteractive()
            expected_value_viz.init()
            criticLosses = []
        
        agent.setSettings(settings)
        agent.setExperience(experience)
        agent.setPolicy(model)
    
        mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error, mean_eval, std_eval = evalModel(actor, exp, agent, settings["discount_factor"], 
                                                anchors=settings['eval_epochs'], action_space_continuous=action_space_continuous, settings=settings, print_data=True, 
                                                bootstrapping=True, visualizeEvaluation=None, p=10.0, sampling=True)

        print ("Average Reward: " + str(mean_reward))
        
        exp.finish()
        agent.finish()
        
    #except Exception, e:
    #    print "Error: " + str(e)
    #    raise e
    
if __name__ == "__main__":
    
    file = open(sys.argv[1])
    settings = json.load(file)
    file.close()
    modelSampling(settings)
    