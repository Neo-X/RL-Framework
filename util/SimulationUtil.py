import copy
import sys
sys.setrecursionlimit(50000)
import os
import json
from numpy import dtype
sys.path.append("../")
sys.path.append("../env")
sys.path.append("../characterSimAdapter/")
sys.path.append("../simbiconAdapter/")
sys.path.append("../simAdapter/")
import math
import numpy as np

from env.BallGame2D import BallGame2D
from env.BallGame1D import BallGame1D
from ModelEvaluation import *

from model.RLDeepNet import RLDeepNet
from model.DeepRLNet3 import DeepRLNet3  
from model.DeepCACLA import DeepCACLA
from model.DeepCACLADQ import DeepCACLADQ
from model.DeepCACLADV import DeepCACLADV
from model.DeepCACLADropout import DeepCACLADropout
from model.DeepDPG import DeepDPG
from model.DeepDPG2 import DeepDPG2
from model.DeepDPGDQ import DeepDPGDQ
from model.LearningAgent import *

from model.DeepDropout import DeepDropout
from model.DeepNN import DeepNN
from model.DeepNNSingleNet import DeepNNSingleNet
from model.DeepCNN import DeepCNN
from model.DeepCNNDropout import DeepCNNDropout

from algorithm.DeepQNetwork import DeepQNetwork
from algorithm.DoubleDeepQNetwork import DoubleDeepQNetwork
from algorithm.CACLA import CACLA
from algorithm.CACLADV import CACLADV
from algorithm.CACLADVTarget import CACLADVTarget
from algorithm.ForwardDynamics import ForwardDynamics

from util.ExperienceMemory import ExperienceMemory
from RLVisualize import RLVisualize
from NNVisualize import NNVisualize

from actor.ActorInterface import ActorInterface
from actor.BallGame2DActor import BallGame2DActor
from actor.BallGame1DActor import BallGame1DActor
from actor.SimbiconActor import SimbiconActor

from sim.PendulumEnvState import PendulumEnvState
from sim.PendulumEnv import PendulumEnv
from sim.SimbiconEnv import SimbiconEnv
from sim.ParkourEnv import ParkourEnv
from sim.BallGame2DEnv import BallGame2DEnv
from sim.BallGame1DEnv import BallGame1DEnv

#Sampler types
from model.ForwardDynamicsNetwork import ForwardDynamicsNetwork
from model.ForwardDynamicsCNN import ForwardDynamicsCNN
from model.ForwardDynamicsCNN2 import ForwardDynamicsCNN2
from model.ForwardDynamicsCNN3 import ForwardDynamicsCNN3
from model.ForwardDynamicsCNNTile import ForwardDynamicsCNNTile
from model.ForwardDynamicsSimulator import ForwardDynamicsSimulator
from model.ForwardDynamicsSimulatorParallel import ForwardDynamicsSimulatorParallel
from model.Sampler import Sampler
from model.BruteForceSampler import BruteForceSampler
from model.SequentialMCSampler import SequentialMCSampler
from model.ForwardPlanner import ForwardPlanner


import random
# import cPickle
import dill
import dill as pickle
import dill as cPickle

import cProfile, pstats, io
# import memory_profiler
# import psutil
import gc
# from guppy import hpy; h=hpy()
# from memprof import memprof

# import pathos.multiprocessing
import multiprocessing

from model.ModelUtil import *
import lasagne

def getDataDirectory(settings):
    return settings["environment_type"]+"/"+settings["agent_name"]+"/"+settings["data_folder"]+"/"+settings["model_type"]+"/"

def getTaskDataDirectory(settings):
    return settings["environment_type"]+"/"+settings["agent_name"]+"/"+settings["task_data_folder"]+"/"+settings["model_type"]+"/"


def validateSettings(settings):
    """
        This method is used to check and overwrite any settings that are not going to work properly
        for example, geck is there is a display screen
    """
    
    if ( not ( "DISPLAY" in os.environ)): # No screen on this computer
        settings['visulaize_forward_dynamics'] = False
        settings['visualize_learning'] = False
    
    return settings

def createNetworkModel(model_type, state_bounds, action_bounds, reward_bounds, settings):
    if settings['action_space_continuous']:
        n_out_ = len(action_bounds[0])
    else:
        n_out_ = len(action_bounds)
    if (settings['load_saved_model'] == True):
        return None
    elif (model_type == "Deep_Dropout" ):
        model = DeepDropout(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (model_type == "Deep_NN" ):
        model = DeepNN(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (model_type == "Deep_CNN" ):
        model = DeepCNN(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (model_type == "Deep_CNN_Dropout" ):
        model = DeepCNNDropout(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (model_type == "Deep_NN_SingleNet" ):
        model = DeepNNSingleNet(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    else:
        print ("Unknown network model type: ", str(model_type), " I hope you know what you are doing....")
        # sys.exit(2)
        return
    
    print ("Number of Critic network parameters", lasagne.layers.count_params(model.getCriticNetwork()))
    print ("Number of Actor network parameters", lasagne.layers.count_params(model.getActorNetwork()))
    return model

def createRLAgent(algorihtm_type, state_bounds, action_bounds, reward_bounds, settings):
    
    networkModel = createNetworkModel(settings["model_type"], state_bounds, action_bounds, reward_bounds, settings)
    
    if (settings['load_saved_model'] == True):
        print ("Loading pre compiled network")
        file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+"_Best.pkl"
        f = open(file_name, 'rb')
        model = dill.load(f)
        f.close()
    elif ( "Deep_NN2" == algorihtm_type):
        model = RLDeepNet(n_in=len(state_bounds[0]), n_out=num_actions, state_bounds=state_bounds, 
                          action_bounds=None, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "Deep_NN3" ):
        model = DeepRLNet3(n_in=len(state_bounds[0]), n_out=num_actions, state_bounds=state_bounds, 
                          action_bounds=None, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "Deep_CACLA" ):
        model = DeepCACLA(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "Deep_CACLA2" ):
        model = DeepCACLA2(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "Deep_CACLA_Dropout" ):
        model = DeepCACLADropout(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "Deep_CACLA_DQ" ):
        model = DeepCACLADQ(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "DeepCACLADV" ):
        model = DeepCACLADV(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "Deep_DPG" ):
        model = DeepDPG(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "Deep_DPG_DQ" ):
        model = DeepDPGDQ(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "Deep_DPG_2" ):
        model = DeepDPG2(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "CACLA" ):
        model = CACLA(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "CACLADV" ):
        model = CACLADV(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "CACLADVTarget" ):
        model = CACLADVTarget(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "DeepQNetwork" ):
        print ("Using model type ", algorihtm_type , " with ", len(action_bounds), " actions")
        model = DeepQNetwork(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "DoubleDeepQNetwork" ):
        print ("Using model type ", algorihtm_type , " with ", len(action_bounds), " actions")
        model = DoubleDeepQNetwork(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    else:
        print ("Unknown learning algorithm type: " + str(algorihtm_type))
        sys.exit(2)
    
    return model


def createEnvironment(config_file, env_type):
    
    if env_type == 'ballgame_2d':
        file = open(config_file)
        conf = json.load(file)
        # print ("Settings: " + str(json.dumps(conf)))
        file.close()
        exp = BallGame2D(conf)
        exp = BallGame2DEnv(exp)
        return exp
    elif env_type == 'ballgame_1d':
        file = open(config_file)
        conf = json.load(file)
        # print ("Settings: " + str(json.dumps(conf)))
        file.close()
        exp = BallGame1D(conf)
        exp = BallGame1DEnv(exp)
        return exp
    elif env_type == 'simbiconBiped2D':
        import simbiconAdapter
        c = simbiconAdapter.Configuration(config_file)
        print ("Num state: ", c._NUMBER_OF_STATES)
        sim = simbiconAdapter.SimbiconWrapper(c)
        print ("Using Environment Type: " + str(env_type))
        exp = SimbiconEnv(sim)
        exp._conf = c # OMFG HACK so that python does not garbage collect the configuration and F everything up!
        return exp
    elif env_type == 'parkourBiped2D':
        import terrainRLAdapter
        sim = terrainRLAdapter.cSimAdapter()
        sim.init(['train', '-arg_file=', config_file])
        # print ("Num state: ", c._NUMBER_OF_STATES)
        # sim = simbiconAdapter.SimbiconWrapper(c)
        print ("Using Environment Type: " + str(env_type))
        exp = ParkourEnv(sim)
        # exp._conf = c # OMFG HACK so that python does not garbage collect the configuration and F everything up!
        return exp
    
    import characterSim
    c = characterSim.Configuration(config_file)
    # print ("Num state: ", c._NUMBER_OF_STATES)
    exp = characterSim.Experiment(c)
    # print ("Num state: ", exp._config._NUMBER_OF_STATES)
    if env_type == 'pendulum_env_state':
        print ("Using Environment Type: " + str(env_type))
        exp = PendulumEnvState(exp)
    elif env_type == 'pendulum_env':
        print ("Using Environment Type: " + str(env_type))
        exp = PendulumEnv(exp)
    elif env_type == 'pendulum3D_env':
        print ("Using Environment Type: " + str(env_type))
        exp = PendulumEnv(exp)
    elif env_type == 'paperGibbon':
        print ("Using Environment Type: " + str(env_type))
        exp = PendulumEnv(exp)
    else:
        print ("Invalid environment type: " + str(env_type))
        sys.exit()
    
    exp._conf = c # OMFG HACK so that python does not garbage collect the configuration and F everything up!    
    return exp

def createActor(env_type, settings, experience):
    actor=None
    if env_type == 'ballgame_2d':
        actor = BallGame2DActor(settings)
    elif env_type == 'ballgame_1d':
        actor = BallGame1DActor(settings, experience)
    elif env_type == 'simbiconBiped2D':
        actor = SimbiconActor(settings, experience)
    else:
        actor = ActorInterface(settings, experience)
    
    return actor

def createSampler(settings, exp):
    actor=None
    if (settings['sampling_method'] == 'simple'):
        print ("Using Sampling Method: " + str(settings['sampling_method']))
        sampler = Sampler(settings)
    elif (settings['sampling_method'] == 'bruteForce'):
        print ("Using Sampling Method: " + str(settings['sampling_method']))
        sampler = BruteForceSampler()
    elif (settings['sampling_method'] == 'SequentialMC'):
        print ("Using Sampling Method: " + str(settings['sampling_method']))
        sampler = SequentialMCSampler(exp, settings['look_ahead_planning_steps'], settings)
    elif (settings['sampling_method'] == 'ForwardPlanner'):
        print ("Using Sampling Method: " + str(settings['sampling_method']))
        sampler = ForwardPlanner(exp, settings['look_ahead_planning_steps'])
    else:
        print ("Sampler method not supported: " + str(settings['sampling_method']) )
        sys.exit()
    
    return sampler

def createForwardDynamicsModel(settings, state_bounds, action_bounds, actor, exp):
    
    if settings["forward_dynamics_predictor"] == "simulator":
        print ("Using forward dynamics method: " + str(settings["forward_dynamics_predictor"]))
        forwardDynamicsModel = ForwardDynamicsSimulator(len(state_bounds[0]), len(action_bounds[0]), 
                                                        state_bounds, action_bounds, actor, exp, settings)
    elif settings["forward_dynamics_predictor"] == "simulator_parallel":
        print ("Using forward dynamics method: " + str(settings["forward_dynamics_predictor"]))
        forwardDynamicsModel = ForwardDynamicsSimulatorParallel(len(state_bounds[0]), len(action_bounds[0]), 
                                                        state_bounds, action_bounds, actor, exp, settings)
        
    elif settings["forward_dynamics_predictor"] == "saved_network":
        print ("Using forward dynamics method: " + str(settings["forward_dynamics_predictor"]))
        file_name_dynamics=data_folder+"forward_dynamics_"+str(settings['agent_name'])+"_Best.pkl"
        forwardDynamicsModel = dill.load(open(file_name_dynamics))
    elif settings["forward_dynamics_predictor"] == "network":
        print ("Using forward dynamics method: " + str(settings["forward_dynamics_predictor"]))
        fd_net = createForwardDynamicsNetwork(state_bounds, action_bounds, settings)
        forwardDynamicsModel = ForwardDynamics(fd_net, state_length=len(state_bounds[0]), action_length=len(action_bounds[0]), 
                                               state_bounds=state_bounds, action_bounds=action_bounds, settings_=settings)
    else:
        print ("Unrecognized forward dynamics method: " + str(settings["forward_dynamics_predictor"]))
        sys.exit()
        
    return forwardDynamicsModel

def createForwardDynamicsNetwork(state_bounds, action_bounds, settings):
    
    if settings["forward_dynamics_model_type"] == "Deep_NN":
        print ("Using forward dynamics network type: " + str(settings["forward_dynamics_model_type"]))
        forwardDynamicsNetwork = ForwardDynamicsNetwork(len(state_bounds[0]), len(action_bounds[0]), 
                                                        state_bounds, action_bounds, settings)
    elif settings["forward_dynamics_model_type"] == "Deep_CNN":
        print ("Using forward dynamics network type: " + str(settings["forward_dynamics_model_type"]))
        forwardDynamicsNetwork = ForwardDynamicsCNN(len(state_bounds[0]), len(action_bounds[0]), 
                                                        state_bounds, action_bounds, settings)
    elif settings["forward_dynamics_model_type"] == "Deep_CNN_Tile":
        print ("Using forward dynamics network type: " + str(settings["forward_dynamics_model_type"]))
        forwardDynamicsNetwork = ForwardDynamicsCNNTile(len(state_bounds[0]), len(action_bounds[0]), 
                                                        state_bounds, action_bounds, settings)
    elif settings["forward_dynamics_model_type"] == "Deep_CNN2":
        print ("Using forward dynamics network type: " + str(settings["forward_dynamics_model_type"]))
        forwardDynamicsNetwork = ForwardDynamicsCNN2(len(state_bounds[0]), len(action_bounds[0]), 
                                                        state_bounds, action_bounds, settings)
        
    elif settings["forward_dynamics_model_type"] == "Deep_CNN3":
        print ("Using forward dynamics network type: " + str(settings["forward_dynamics_model_type"]))
        forwardDynamicsNetwork = ForwardDynamicsCNN3(len(state_bounds[0]), len(action_bounds[0]), 
                                                        state_bounds, action_bounds, settings)       
        
    else:
        print ("Unrecognized forward dynamics network type: " + str(settings["forward_dynamics_model_type"]))
        sys.exit()
        
    return forwardDynamicsNetwork

