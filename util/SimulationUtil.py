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

from util.ExperienceMemory import ExperienceMemory
# from ModelEvaluation import *
# from RLVisualize import RLVisualize
# from NNVisualize import NNVisualize



import random
# import cPickle
import dill
import dill as pickle
import dill as cPickle

# import cProfile, pstats, io
# import memory_profiler
# import psutil
import gc
# from guppy import hpy; h=hpy()
# from memprof import memprof

# import pathos.multiprocessing
# import multiprocessing

from model.ModelUtil import scale_action

def getDataDirectory(settings):
    return settings["environment_type"]+"/"+settings["agent_name"]+"/"+settings["data_folder"]+"/"+settings["model_type"]+"/"

def getAgentName(settings):
    return 'pendulum_agent'

def getTaskDataDirectory(settings):
    return settings["environment_type"]+"/"+settings["agent_name"]+"/"+settings["task_data_folder"]+"/"+settings["model_type"]+"/"


def validateSettings(settings):
    """
        This method is used to check and overwrite any settings that are not going to work properly
        for example, check if there is a display screen
    """
    """
    ## This doesn't work as well as I was hoping...
    if ( not ( "DISPLAY" in os.environ)): # No screen on this computer
        settings['visulaize_forward_dynamics'] = False
        settings['visualize_learning'] = False
    """
    return settings

def createNetworkModel(model_type, state_bounds, action_bounds, reward_bounds, settings):
    if settings['action_space_continuous']:
        n_out_ = len(action_bounds[0])
    else:
        n_out_ = len(action_bounds)
    if (settings['load_saved_model'] == True):
        return None
    elif (model_type == "Deep_Dropout" ):
        from model.DeepDropout import DeepDropout
        model = DeepDropout(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (model_type == "Deep_NN" ):
        from model.DeepNN import DeepNN
        model = DeepNN(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (model_type == "Deep_CNN" ):
        from model.DeepCNN import DeepCNN
        model = DeepCNN(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (model_type == "Deep_CNN_Dropout" ):
        from model.DeepCNNDropout import DeepCNNDropout
        model = DeepCNNDropout(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (model_type == "Deep_NN_Dropout" ):
        from model.DeepNNDropout import DeepNNDropout
        model = DeepNNDropout(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (model_type == "Deep_NN_SingleNet" ):
        from model.DeepNNSingleNet import DeepNNSingleNet
        model = DeepNNSingleNet(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (model_type == "Deep_CNN_SingleNet" ):
        from model.DeepCNNSingleNet import DeepCNNSingleNet
        model = DeepCNNSingleNet(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)   
    elif (model_type == "Deep_NN_SingleNet_Dropout" ):
        from model.DeepNNSingleNetDropout import DeepNNSingleNetDropout
        model = DeepNNSingleNetDropout(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (model_type == "Deep_NN_Wide" ):
        from model.DeepNNWide import DeepNNWide
        model = DeepNNWide(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (model_type == "Deep_CNN_KERAS" ):
        from model.DeepCNNKeras import DeepCNNKeras
        print("Creating network model: ", model_type)
        model = DeepCNNKeras(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
        return model  
    elif (model_type == "Deep_NN_KERAS" ):
        from model.DeepNNKeras import DeepNNKeras
        print("Creating network model: ", model_type)
        model = DeepNNKeras(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
        return model
    elif (model_type == "Deep_NN_Dropout_Critic" ):
        from model.DeepNNDropoutCritic import DeepNNDropoutCritic
        print("Creating network model: ", model_type)
        model = DeepNNDropoutCritic(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
        return model  
    elif (model_type == "Deep_CNN_Dropout_Critic" ):
        from model.DeepCNNDropoutCritic import DeepCNNDropoutCritic
        print("Creating network model: ", model_type)
        model = DeepCNNDropoutCritic(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
        return model  
    elif (model_type == "Deep_NN_Wide_Dropout_Critic" ):
        from model.DeepNNWideDropoutCritic import DeepNNWideDropoutCritic
        print("Creating network model: ", model_type)
        model = DeepNNWideDropoutCritic(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
        return model  
    elif (model_type == "Deep_NN_TanH" ):
        from model.DeepNNTanH import DeepNNTanH
        print("Creating network model: ", model_type)
        model = DeepNNTanH(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
        return model  
    elif (model_type == "Deep_NN_TanH_SingleNet" ):
        from model.DeepNNTanHSingleNet import DeepNNTanHSingleNet
        print("Creating network model: ", model_type)
        model = DeepNNTanHSingleNet(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
        return model 
    elif (model_type == "Deep_CNN_TanH_SingleNet" ):
        from model.DeepCNNTanHSingleNet import DeepCNNTanHSingleNet
        print("Creating network model: ", model_type)
        model = DeepCNNTanHSingleNet(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
        return model 
    
    
    elif (model_type == "DumbModel" ):
        model = DumbModel(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    else:
        print ("Unknown network model type: ", str(model_type), " I hope you know what you are doing....")
        # sys.exit(2)
        return
    import lasagne
    print (" network type: ", model_type, " : ", model)
    print ("Number of Critic network parameters", lasagne.layers.count_params(model.getCriticNetwork()))
    print ("Number of Actor network parameters", lasagne.layers.count_params(model.getActorNetwork()))
    
    if (settings['train_forward_dynamics'] and (settings['forward_dynamics_model_type'] == 'SingleNet')):
        print ("Number of Forward Dynamics network parameters", lasagne.layers.count_params(model.getForwardDynamicsNetwork()))
        print ("Number of Reward predictor network parameters", lasagne.layers.count_params(model.getRewardNetwork()))

    return model

def createRLAgent(algorihtm_type, state_bounds, discrete_actions, reward_bounds, settings):
    
    action_bounds = np.array(settings['action_bounds'])
    networkModel = createNetworkModel(settings["model_type"], state_bounds, action_bounds, reward_bounds, settings)
    num_actions= discrete_actions.shape[0] # number of rows
    if settings['action_space_continuous']:
            action_bounds = np.array(settings["action_bounds"], dtype=float)
            num_actions = action_bounds.shape[1]
            
    if (settings['load_saved_model'] == True):
        directory= getDataDirectory(settings)
        print ("Loading pre compiled network")
        file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+"_Best.pkl"
        f = open(file_name, 'r')
        model = dill.load(f)
        model.setSettings(settings)
        f.close()
    elif ( "Deep_NN2" == algorihtm_type):
        from model.RLDeepNet import RLDeepNet
        model = RLDeepNet(n_in=len(state_bounds[0]), n_out=num_actions, state_bounds=state_bounds, 
                          action_bounds=None, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "Deep_NN3" ):
        from model.DeepRLNet3 import DeepRLNet3
        model = DeepRLNet3(n_in=len(state_bounds[0]), n_out=num_actions, state_bounds=state_bounds, 
                          action_bounds=None, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "Deep_CACLA" ):
        from model.DeepCACLA import DeepCACLA
        model = DeepCACLA(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "Deep_CACLA2" ):
        from model.DeepCACLA2 import DeepCACLA2
        model = DeepCACLA2(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "Deep_CACLA_Dropout" ):
        from model.DeepCACLADropout import DeepCACLADropout
        model = DeepCACLADropout(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "Deep_CACLA_DQ" ):
        from model.DeepCACLADQ import DeepCACLADQ
        model = DeepCACLADQ(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "DeepCACLADV" ):
        from model.DeepCACLADV import DeepCACLADV
        model = DeepCACLADV(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "Deep_DPG" ):
        from model.DeepDPG import DeepDPG
        model = DeepDPG(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "Deep_DPG_DQ" ):
        from model.DeepDPGDQ import DeepDPGDQ
        model = DeepDPGDQ(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "Deep_DPG_2" ):
        from model.DeepDPG2 import DeepDPG2
        model = DeepDPG2(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "CACLA" ):
        from algorithm.CACLA import CACLA
        model = CACLA(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "CACLA2" ):
        from algorithm.CACLA import CACLA
        model = CACLA2(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "CACLADV" ):
        from algorithm.CACLADV import CACLADV
        model = CACLADV(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "CACLADVTarget" ):
        from algorithm.CACLADVTarget import CACLADVTarget
        model = CACLADVTarget(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "DeepQNetwork" ):
        from algorithm.DeepQNetwork import DeepQNetwork
        print ("Using model type ", algorihtm_type , " with ", len(action_bounds), " actions")
        model = DeepQNetwork(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "DoubleDeepQNetwork" ):
        from algorithm.DoubleDeepQNetwork import DoubleDeepQNetwork
        print ("Using model type ", algorihtm_type , " with ", len(action_bounds), " actions")
        model = DoubleDeepQNetwork(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "A_CACLA" ):
        from algorithm.A_CACLA import A_CACLA
        model = A_CACLA(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "A3C2" ):
        from algorithm.A3C2 import A3C2
        model = A3C2(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "TRPO" ):
        from algorithm.TRPO import TRPO
        model = TRPO(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "PPO" ):
        from algorithm.PPO import PPO
        model = PPO(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "AP_CACLA" ):
        from algorithm.AP_CACLA import AP_CACLA
        model = AP_CACLA(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)    
    elif (algorihtm_type == "PPO_Critic" ):
        from algorithm.PPOCritic import PPOCritic
        model = PPOCritic(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "PPO_Critic_2" ):
        from algorithm.PPOCritic2 import PPOCritic2
        model = PPOCritic2(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "TRPO_Critic" ):
        from algorithm.TRPOCritic import TRPOCritic
        model = TRPOCritic(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "CACLA_KERAS" ):
        from algorithm.CACLA_KERAS import CACLA_KERAS
        model = CACLA_KERAS(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    elif (algorihtm_type == "CACLA_Entropy" ):
        from algorithm.CACLAEntropy import CACLAEntropy
        model = CACLAEntropy(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    else:
        print ("Unknown learning algorithm type: " + str(algorihtm_type))
        raise ValueError("Unknown learning algorithm type: " + str(algorihtm_type))
        # sys.exit(2)
        
    print ("Using model type ", algorihtm_type , " : ", model)
    
    return model


def createEnvironment(config_file, env_type, settings, render=False):
    print("Creating sim Type: ", env_type)
    if env_type == 'ballgame_2d':
        from env.BallGame2D import BallGame2D
        from sim.BallGame2DEnv import BallGame2DEnv
        file = open(config_file)
        conf = json.load(file)
        # print ("Settings: " + str(json.dumps(conf)))
        file.close()
        conf['render'] = render
        exp = BallGame2D(conf)
        exp = BallGame2DEnv(exp, settings)
        return exp
    elif env_type == 'ballgame_1d':
        from env.BallGame1D import BallGame1D
        from sim.BallGame1DEnv import BallGame1DEnv
        file = open(config_file)
        conf = json.load(file)
        # print ("Settings: " + str(json.dumps(conf)))
        file.close()
        conf['render'] = render
        exp = BallGame1D(conf)
        exp = BallGame1DEnv(exp, settings)
        return exp
    elif env_type == 'gapgame_1d':
        from env.GapGame1D import GapGame1D
        from sim.GapGame1DEnv import GapGame1DEnv
        file = open(config_file)
        conf = json.load(file)
        # print ("Settings: " + str(json.dumps(conf)))
        file.close()
        conf['render'] = render
        exp = GapGame1D(conf)
        exp = GapGame1DEnv(exp, settings)
        return exp
    elif env_type == 'gapgame_2d':
        from env.GapGame2D import GapGame2D
        from sim.GapGame2DEnv import GapGame2DEnv
        file = open(config_file)
        conf = json.load(file)
        # print ("Settings: " + str(json.dumps(conf)))
        file.close()
        conf['render'] = render
        exp = GapGame2D(conf)
        exp = GapGame2DEnv(exp, settings)
        return exp
    elif env_type == 'nav_Game':
        from env.NavGame import NavGame
        from sim.NavGameEnv import NavGameEnv
        # file = open(config_file)
        # conf = json.load(file)
        conf = copy.deepcopy(settings)
        # print ("Settings: " + str(json.dumps(conf)))
        # file.close()
        conf['render'] = render
        exp = NavGame(conf)
        exp = NavGameEnv(exp, settings)
        return exp
    
    elif env_type == 'open_AI_Gym':
        import gym
        from gym import wrappers
        from gym import envs
        from sim.OpenAIGymEnv import OpenAIGymEnv
        # print(envs.registry.all())
        
        # env = gym.make('CartPole-v0')
        env_name = config_file
        env = gym.make(env_name)
        # file = open(config_file)
        # conf = json.load(file)
        
        conf = copy.deepcopy(settings)
        conf['render'] = render
        exp = OpenAIGymEnv(env, conf)
        exp = exp
        return exp
    
    elif ((env_type == 'simbiconBiped2D') or (env_type == 'simbiconBiped3D') or (env_type == 'Imitate3D') or 
          (env_type == 'simbiconBiped2DTerrain') or (env_type == 'hopper_2D')):
        import simbiconAdapter
        from sim.SimbiconEnv import SimbiconEnv
        c = simbiconAdapter.Configuration(config_file)
        print ("Num state: ", c._NUMBER_OF_STATES)
        c._RENDER = render
        sim = simbiconAdapter.SimbiconWrapper(c)
        print ("Using Environment Type: " + str(env_type))
        exp = SimbiconEnv(sim, settings)
        exp._conf = c # OMFG HACK so that python does not garbage collect the configuration and F everything up!
        return exp
    elif ((env_type == 'mocapImitation2D') or (env_type == 'mocapImitation3D')):
        import simbiconAdapter
        from sim.MocapImitationEnv import MocapImitationEnv
        c = simbiconAdapter.Configuration(config_file)
        print ("Num state: ", c._NUMBER_OF_STATES)
        c._RENDER = render
        sim = simbiconAdapter.SimbiconWrapper(c)
        print ("Using Environment Type: " + str(env_type))
        exp = MocapImitationEnv(sim, settings)
        exp._conf = c # OMFG HACK so that python does not garbage collect the configuration and F everything up!
        return exp
    elif env_type == 'terrainRLBiped2D':
        import terrainRLAdapter
        from sim.TerrainRLEnv import TerrainRLEnv
        sim = terrainRLAdapter.cSimAdapter(['train', '-arg_file=', config_file])
        sim.setRender(render)
        # sim.init(['train', '-arg_file=', config_file])
        # print ("Num state: ", c._NUMBER_OF_STATES)
        # sim = simbiconAdapter.SimbiconWrapper(c)
        print ("Using Environment Type: " + str(env_type))
        exp = TerrainRLEnv(sim, settings)
        # exp._conf = c # OMFG HACK so that python does not garbage collect the configuration and F everything up!
        return exp
    elif env_type == 'terrainRLFlatBiped2D':
        import terrainRLAdapter
        from sim.TerrainRLFlatEnv import TerrainRLFlatEnv
        sim = terrainRLAdapter.cSimAdapter(['train', '-arg_file=', config_file])
        sim.setRender(render)
        # sim.init(['train', '-arg_file=', config_file])
        # print ("Num state: ", c._NUMBER_OF_STATES)
        # sim = simbiconAdapter.SimbiconWrapper(c)
        print ("Using Environment Type: " + str(env_type))
        exp = TerrainRLFlatEnv(sim, settings)
        # exp._conf = c # OMFG HACK so that python does not garbage collect the configuration and F everything up!
        return exp
    elif env_type == 'terrainRLImitateBiped2D':
        import terrainRLAdapter
        from sim.TerrainRLImitateEnv import TerrainRLImitateEnv
        sim = terrainRLAdapter.cSimAdapter(['train', '-arg_file=', config_file])
        sim.setRender(render)
        # sim.init(['train', '-arg_file=', config_file])
        # print ("Num state: ", c._NUMBER_OF_STATES)
        # sim = simbiconAdapter.SimbiconWrapper(c)
        print ("Using Environment Type: " + str(env_type))
        exp = TerrainRLImitateEnv(sim, settings)
        # exp._conf = c # OMFG HACK so that python does not garbage collect the configuration and F everything up!
        return exp
    
    import characterSim
    c = characterSim.Configuration(config_file)
    # print ("Num state: ", c._NUMBER_OF_STATES)
    c._RENDER = render
    exp = characterSim.Experiment(c)
    # print ("Num state: ", exp._config._NUMBER_OF_STATES)
    if env_type == 'pendulum_env_state':
        from sim.PendulumEnvState import PendulumEnvState
        print ("Using Environment Type: " + str(env_type))
        exp = PendulumEnvState(exp, settings)
    elif env_type == 'pendulum_env':
        from sim.PendulumEnv import PendulumEnv
        print ("Using Environment Type: " + str(env_type))
        exp = PendulumEnv(exp, settings)
    elif env_type == 'pendulum3D_env':
        from sim.PendulumEnv import PendulumEnv
        print ("Using Environment Type: " + str(env_type))
        exp = PendulumEnv(exp, settings)
    elif env_type == 'pendulum_3D_env':
        from sim.PendulumEnv import PendulumEnv
        print ("Using Environment Type: " + str(env_type))
        exp = PendulumEnv(exp, settings)
    elif env_type == 'paperGibbon_env':
        from sim.PaperGibbonEnv import PaperGibbonEnv
        print ("Using Environment Type: " + str(env_type))
        exp = PaperGibbonEnv(exp, settings)
    else:
        print ("Invalid environment type: " + str(env_type))
        raise ValueError("Invalid environment type: " + str(env_type))
        # sys.exit()
    
    exp._conf = c # OMFG HACK so that python does not garbage collect the configuration and F everything up!    
    return exp

def createActor(env_type, settings, experience):
    actor=None
    if env_type == 'ballgame_2d':
        from actor.BallGame2DActor import BallGame2DActor
        actor = BallGame2DActor(settings, experience)
    elif env_type == 'ballgame_1d':
        from actor.BallGame1DActor import BallGame1DActor
        actor = BallGame1DActor(settings, experience)
    elif env_type == 'gapgame_1d':
        from actor.GapGame1DActor import GapGame1DActor
        actor = GapGame1DActor(settings, experience)
    elif env_type == 'gapgame_2d':
        from actor.GapGame2DActor import GapGame2DActor
        actor = GapGame2DActor(settings, experience)
    elif (env_type == 'nav_Game'):
        from actor.NavGameActor import NavGameActor
        actor = NavGameActor(settings, experience)
    elif ((env_type == 'simbiconBiped2D') or (env_type == 'simbiconBiped3D') or
          (env_type == 'simbiconBiped2DTerrain')):
        from actor.SimbiconActor import SimbiconActor
        actor = SimbiconActor(settings, experience)
    elif ((env_type == 'mocapImitation2D') or (env_type == 'mocapImitation3D')):
        from actor.MocapImitationActor import MocapImitationActor
        actor = MocapImitationActor(settings, experience)
    elif ((env_type == 'hopper_2D')):
        from actor.Hopper2DActor import Hopper2DActor
        actor = Hopper2DActor(settings, experience)
    elif (env_type == 'Imitate3D') :
        from actor.ImitationActor import ImitationActor
        actor = ImitationActor(settings, experience)
    elif env_type == 'terrainRLBiped2D' or (env_type == 'terrainRLFlatBiped2D'):
        from actor.TerrainRLActor import TerrainRLActor
        actor = TerrainRLActor(settings, experience)
    elif (env_type == 'terrainRLImitateBiped2D'):
        from actor.TerrainRLImitationActor import TerrainRLImitationActor
        actor = TerrainRLImitationActor(settings, experience)
    elif (env_type == 'paperGibbon_env'):
        from actor.PaperGibbonAgent import PaperGibbonAgent
        actor = PaperGibbonAgent(settings, experience)
    elif (env_type == 'pendulum'):
        from actor.ActorInterface import ActorInterface
        actor = ActorInterface(settings, experience)
    elif (env_type == 'open_AI_Gym'):
        from actor.OpenAIGymActor import OpenAIGymActor
        actor = OpenAIGymActor(settings, experience)
    else:
        print("Error actor type unknown: ", env_type)
        raise ValueError("Error actor type unknown: ", env_type)
        # sys.exit()
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

def createForwardDynamicsModel(settings, state_bounds, action_bounds, actor, exp, agentModel):
    
    if settings["forward_dynamics_predictor"] == "simulator":
        from model.ForwardDynamicsSimulator import ForwardDynamicsSimulator
        print ("Using forward dynamics method: " + str(settings["forward_dynamics_predictor"]))
        forwardDynamicsModel = ForwardDynamicsSimulator(len(state_bounds[0]), len(action_bounds[0]), 
                                                        state_bounds, action_bounds, actor, exp, settings)
    elif settings["forward_dynamics_predictor"] == "simulator_parallel":
        from model.ForwardDynamicsSimulatorParallel import ForwardDynamicsSimulatorParallel
        print ("Using forward dynamics method: " + str(settings["forward_dynamics_predictor"]))
        forwardDynamicsModel = ForwardDynamicsSimulatorParallel(len(state_bounds[0]), len(action_bounds[0]), 
                                                        state_bounds, action_bounds, actor, exp, settings)
        forwardDynamicsModel.init(len(state_bounds[0]), len(action_bounds[0]), state_bounds, action_bounds, actor, exp, settings)
    elif settings["forward_dynamics_predictor"] == "saved_network":
        # from model.ForwardDynamicsNetwork import ForwardDynamicsNetwork
        print ("Using forward dynamics method: " + str(settings["forward_dynamics_predictor"]))
        file_name_dynamics=data_folder+"forward_dynamics_"+str(settings['agent_name'])+"_Best.pkl"
        forwardDynamicsModel = dill.load(open(file_name_dynamics))
    elif settings["forward_dynamics_predictor"] == "network":
        print ("Using forward dynamics method: " + str(settings["forward_dynamics_predictor"]))
        if (settings['load_saved_model'] == True):
            print ("Loading pre compiled network")
            directory= getDataDirectory(settings)
            file_name_dynamics=directory+"forward_dynamics_"+str(settings['agent_name'])+"_Best.pkl"
            f = open(file_name, 'rb')
            forwardDynamicsModel = dill.load(f)
            f.close()
        else:
            if ( settings['forward_dynamics_model_type'] == "SingleNet"):
                ## Hopefully this will allow for parameter sharing across both models...
                fd_net = agentModel.getModel()
            else:
                fd_net = createForwardDynamicsNetwork(state_bounds, action_bounds, settings)
            from algorithm.ForwardDynamics import ForwardDynamics
            forwardDynamicsModel = ForwardDynamics(fd_net, state_length=len(state_bounds[0]), action_length=len(action_bounds[0]), 
                                                   state_bounds=state_bounds, action_bounds=action_bounds, settings_=settings)
    else:
        print ("Unrecognized forward dynamics method: " + str(settings["forward_dynamics_predictor"]))
        raise ValueError("Unrecognized forward dynamics method: " + str(settings["forward_dynamics_predictor"]))
        # sys.exit()
        
    return forwardDynamicsModel

def createForwardDynamicsNetwork(state_bounds, action_bounds, settings):
    
    if settings["forward_dynamics_model_type"] == "Deep_NN":
        from model.ForwardDynamicsNetwork import ForwardDynamicsNetwork
        print ("Using forward dynamics network type: " + str(settings["forward_dynamics_model_type"]))
        forwardDynamicsNetwork = ForwardDynamicsNetwork(len(state_bounds[0]), len(action_bounds[0]), 
                                                        state_bounds, action_bounds, settings)
    elif settings["forward_dynamics_model_type"] == "Deep_NN_Dropout":
        from model.ForwardDynamicsNNDropout import ForwardDynamicsNNDropout
        print ("Using forward dynamics network type: " + str(settings["forward_dynamics_model_type"]))
        forwardDynamicsNetwork = ForwardDynamicsNNDropout(len(state_bounds[0]), len(action_bounds[0]), 
                                                        state_bounds, action_bounds, settings)
    elif settings["forward_dynamics_model_type"] == "Deep_CNN":
        from model.ForwardDynamicsCNN import ForwardDynamicsCNN
        print ("Using forward dynamics network type: " + str(settings["forward_dynamics_model_type"]))
        forwardDynamicsNetwork = ForwardDynamicsCNN(len(state_bounds[0]), len(action_bounds[0]), 
                                                        state_bounds, action_bounds, settings)
    elif settings["forward_dynamics_model_type"] == "Deep_CNN_Tile":
        from model.ForwardDynamicsCNNTile import ForwardDynamicsCNNTile
        print ("Using forward dynamics network type: " + str(settings["forward_dynamics_model_type"]))
        forwardDynamicsNetwork = ForwardDynamicsCNNTile(len(state_bounds[0]), len(action_bounds[0]), 
                                                        state_bounds, action_bounds, settings)
    elif settings["forward_dynamics_model_type"] == "Deep_CNN2":
        from model.ForwardDynamicsCNN2 import ForwardDynamicsCNN2
        print ("Using forward dynamics network type: " + str(settings["forward_dynamics_model_type"]))
        forwardDynamicsNetwork = ForwardDynamicsCNN2(len(state_bounds[0]), len(action_bounds[0]), 
                                                        state_bounds, action_bounds, settings)
        
    elif settings["forward_dynamics_model_type"] == "Deep_CNN3":
        from model.ForwardDynamicsCNN3 import ForwardDynamicsCNN3
        print ("Using forward dynamics network type: " + str(settings["forward_dynamics_model_type"]))
        forwardDynamicsNetwork = ForwardDynamicsCNN3(len(state_bounds[0]), len(action_bounds[0]), 
                                                        state_bounds, action_bounds, settings)       
    elif settings["forward_dynamics_model_type"] == "Deep_CNN_Dropout":
        from model.ForwardDynamicsCNNDropout import ForwardDynamicsCNNDropout
        print ("Using forward dynamics network type: " + str(settings["forward_dynamics_model_type"]))
        forwardDynamicsNetwork = ForwardDynamicsCNNDropout(len(state_bounds[0]), len(action_bounds[0]), 
                                                        state_bounds, action_bounds, settings)   
    elif settings["forward_dynamics_model_type"] == "Deep_Dense_NN_Dropout":
        from model.ForwardDynamicsDenseNetworkDropout import ForwardDynamicsDenseNetworkDropout
        print ("Using forward dynamics network type: " + str(settings["forward_dynamics_model_type"]))
        forwardDynamicsNetwork = ForwardDynamicsDenseNetworkDropout(len(state_bounds[0]), len(action_bounds[0]), 
                                                        state_bounds, action_bounds, settings)   
            
    else:
        print ("Unrecognized forward dynamics network type: " + str(settings["forward_dynamics_model_type"]))
        raise ValueError("Unrecognized forward dynamics network type: " + str(settings["forward_dynamics_model_type"]))
        # sys.exit()
    import lasagne
    print ("Number of Forward Dynamics network parameters", lasagne.layers.count_params(forwardDynamicsNetwork.getForwardDynamicsNetwork()))
    print ("Number of Reward predictor network parameters", lasagne.layers.count_params(forwardDynamicsNetwork.getRewardNetwork()))
    return forwardDynamicsNetwork

