import random
import numpy as np
import math
import dill
import json
import os
import sys


# Networks
from model.RLDeepNet import RLDeepNet 
from model.DeepCACLA import DeepCACLA
# from model.DeepDPG import DeepDPG 
from model.ForwardDynamicsNetwork import ForwardDynamicsNetwork
from model.ForwardDynamicsSimulator import ForwardDynamicsSimulator
from model.ForwardDynamicsSimulatorParallel import ForwardDynamicsSimulatorParallel
from model.Sampler import Sampler
from model.BruteForceSampler import BruteForceSampler
from model.SequentialMCSampler import SequentialMCSampler
from model.ForwardPlanner import ForwardPlanner

# Games
from sim.PendulumEnvState import PendulumEnvState
from sim.PendulumEnv import PendulumEnv

from actor.ActorInterface import ActorInterface

from model.ModelUtil import *
from ModelEvaluation import *
from util.SimulationUtil import *

from RLVisualize import RLVisualize
from NNVisualize import NNVisualize
from util.ExperienceMemory import ExperienceMemory


def modelSampling(settings):
    # make a color map of fixed colors
    #try: 
        # Normalization constants for data
        
                
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
            
    ### Using a wrapper for the type of actor now
        if action_space_continuous:
            experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), settings['expereince_length'], continuous_actions=True, settings=settings)
        else:
            experience = ExperienceMemory(len(state_bounds[0]), 1, settings['expereince_length'])
        ### Using a wrapper for the type of actor now
        actor = createActor(str(settings['environment_type']),settings, experience)
        # this is the process that selects which game to play
        exp = createEnvironment(str(settings["sim_config_file"]), str(settings['environment_type']), settings)
        
        data_folder = getDataDirectory(settings)
        
        sampler = createSampler(settings, exp)
            
        # if (settings['use_actor_policy_action_suggestion']):
        file_name=data_folder+"pendulum_agent_"+str(settings['agent_name'])+"_Best.pkl"
        model = dill.load(open(file_name))
        
        if settings["forward_dynamics_predictor"] == "simulator":
            print "Using forward dynamics method: " + str(settings["forward_dynamics_predictor"])
            forwardDynamicsModel = ForwardDynamicsSimulator(len(model._state_bounds[0]), len(model._action_bounds[0]), 
                                                            model._state_bounds, model._action_bounds, actor, exp, settings)
            sampler.setForwardDynamics(forwardDynamicsModel)
        elif settings["forward_dynamics_predictor"] == "simulator_parallel":
            print "Using forward dynamics method: " + str(settings["forward_dynamics_predictor"])
            forwardDynamicsModel = ForwardDynamicsSimulatorParallel(len(model._state_bounds[0]), len(model._action_bounds[0]), 
                                                            model._state_bounds, model._action_bounds, actor, exp, settings)
            sampler.setForwardDynamics(forwardDynamicsModel)
            
        if (settings['train_forward_dynamics']):
            file_name_dynamics=directory+"forward_dynamics_"+str(settings['agent_name'])+"_Best.pkl"
            # file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+".pkl"
            f = open(file_name_dynamics, 'r')
            forwardDynamicsModel = dill.load(f)
            f.close()
            sampler.setForwardDynamics(forwardDynamicsModel)
                
        sampler.setPolicy(model)
        sampler.setSettings(settings)
        
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error = evalModel(actor, exp, sampler, settings["discount_factor"], 
                                                anchors=settings['eval_epochs'], action_space_continuous=action_space_continuous, settings=settings, print_data=True)

        print "Average Reward: " + str(mean_reward)
    #except Exception, e:
    #    print "Error: " + str(e)
    #    raise e
    
if __name__ == "__main__":
    
    file = open(sys.argv[1])
    settings = json.load(file)
    file.close()
    modelSampling(settings)
    