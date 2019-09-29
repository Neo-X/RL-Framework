import copy
import sys
import traceback
import logging
from dill.settings import settings
sys.setrecursionlimit(50000)
import os
import json
sys.path.append("../")
import cProfile, pstats, io
# import memory_profiler
# import psutil
import gc
# from guppy import hpy; h=hpy()
# from memprof import memprof

import multiprocessing

sim_processes = []
learning_processes = []
_input_anchor_queue = None
_output_experience_queue = None
_eval_episode_data_queue = None
_sim_work_queues = []

from trainModel import collectEmailData, pretrainCritic, pretrainFD

def createSimWorkers(settings, input_anchor_queue, output_experience_queue, eval_episode_data_queue, model, forwardDynamicsModel, exp_val, default_sim_id=None):
    """
        Creates a number of simulation workers and the message queues that
        are used to tell them what to simulate.
    """
    
    from model.LearningMultiAgent import LearningMultiAgent
    from simulation.SimWorker import SimWorker
    from util.SimulationUtil import createActor, getAgentName, createSampler, createForwardDynamicsModel
    
    
    sim_workers = []
    sim_work_queues = []
    for process in range(abs(settings['num_available_threads'])):
        # this is the process that selects which game to play
        exp_=None
        
        if (int(settings["num_available_threads"]) == -1): # This is okay if there is one thread only...
            print ("Assigning same EXP")
            exp_ = exp_val # This should not work properly for many simulations running at the same time. It could try and evalModel a simulation while it is still running samples 
        print ("original exp: ", exp_)
            # sys.exit()
        ### Using a wrapper for the type of actor now
        actor = createActor(settings['environment_type'], settings, None)
        
        agent = LearningMultiAgent(settings_=settings)
        agent.setSettings(settings)
        agent.setPolicy(model)
        if (settings['train_forward_dynamics']):
            agent.setForwardDynamics(forwardDynamicsModel)
        
        elif ( settings['use_simulation_sampling'] ):
            
            sampler = createSampler(settings, exp_)
            ## This should be some kind of copy of the simulator not a network
            forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, actor, exp_, agentModel=None, print_info=True)
            sampler.setForwardDynamics(forwardDynamicsModel)
            # sampler.setPolicy(model)
            agent.setSampler(sampler)
            print ("thread together exp: ", sampler._exp)
        
        ### Check if this is to be a multi-task simulation
        if type(settings['sim_config_file']) is list:
            if (default_sim_id != None):
                print("Setting sim_id to default id")
                sim_id = default_sim_id
            else:
                print("Setting sim_id to process number")
                sim_id = process
        else:
            print("Not Multi task simulation")
            sim_id = None
            
        print("Setting sim_id to:" , sim_id)
        if (settings['on_policy']):
            message_queue = multiprocessing.Queue(1)
        else:
            message_queue = multiprocessing.Queue(settings['num_available_threads'])
        sim_work_queues.append(message_queue)
        w = SimWorker(input_anchor_queue, output_experience_queue, actor, exp_, agent, settings["discount_factor"], action_space_continuous=settings['action_space_continuous'], 
                settings=settings, print_data=False, p=0.0, validation=True, eval_episode_data_queue=eval_episode_data_queue, process_random_seed=settings['random_seed']+process + 1,
                message_que=message_queue, worker_id=sim_id )
        # w.start()
        sim_workers.append(w)

    return (sim_workers, sim_work_queues)

def createLearningAgent(settings, output_experience_queue, print_info=False):
    """
        Create the Learning Agent to be used
    """
    from model.LearningAgent import LearningWorker
    from model.LearningMultiAgent import LearningMultiAgent
    
    learning_workers = []
    for process in range(1):
        agent = LearningMultiAgent(settings_=settings)
        
        agent.setSettings(settings)
        
        lw = LearningWorker(output_experience_queue, agent, random_seed_=settings['random_seed']+process + 1)
        learning_workers.append(lw)  
    masterAgent = agent
    return (agent, learning_workers)

# python -m memory_profiler example.py
# @profile(precision=5)
# def trainModelParallel(settingsFileName, settings):
def trainModelParallel(inputData):
    # (sys.argv[1], settings)
    settings = inputData[1]
    if ("perform_multiagent_training" not in settings):
        settings["perform_multiagent_training"] = 1
        settings["state_bounds"] = [settings["state_bounds"]]
        settings["action_bounds"] = [settings["action_bounds"]]
        settings["reward_bounds"] = [settings["reward_bounds"]]
        settings["exploration_rate"] = [settings["exploration_rate"]]
        settings["experience_length"] = [settings["experience_length"]]
        settings["critic_network_layer_sizes"] = [settings["critic_network_layer_sizes"]]
        settings["policy_network_layer_sizes"] = [settings["policy_network_layer_sizes"]]
    from util.SimulationUtil import setupEnvironmentVariable, setupLearningBackend
    from simulation.LoggingWorker import LoggingWorker
    from util.SimulationUtil import validateSettings, getFDStateSize
    if (not validateSettings(settings)):
        return False
    exp_logger = setupEnvironmentVariable(settings)
    settingsFileName = inputData[0]
    settings['sample_single_trajectories'] = True
    # settings['shouldRender'] = True
    # pr = cProfile.Profile()
    # pr.enable()
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
    trainData["mean_critic_loss"]=[]
    trainData["std_critic_loss"]=[]
    trainData["mean_critic_regularization_cost"]=[]
    trainData["std_critic_regularization_cost"]=[]
    trainData["mean_actor_loss"]=[]
    trainData["std_actor_loss"]=[]
    trainData["mean_actor_regularization_cost"]=[]
    trainData["std_actor_regularization_cost"]=[]
    trainData["anneal_p"]=[]
    trainData["round"]=0
    try:
            
        rounds = settings["rounds"]
        epochs = settings["epochs"]
        # settings["num_available_threads"] = int(settings["num_available_threads"])
        # num_states=settings["num_states"]
        epsilon = settings["epsilon"]
        discount_factor=settings["discount_factor"]
        reward_bounds=settings["reward_bounds"]
        # reward_bounds = np.array([[-10.1],[0.0]])
        if ( 'value_function_batch_size' in settings):
            batch_size=settings["value_function_batch_size"]
        else:
            batch_size=settings["batch_size"]
        train_on_validation_set=settings["train_on_validation_set"]
        state_bounds = settings['state_bounds']
        discrete_actions = settings['discrete_actions']
        num_actions= len(discrete_actions) # number of rows
        print ("Sim config file name: " + str(settings["sim_config_file"]))
        # c = characterSim.Configuration(str(settings["sim_config_file"]))
        # c = characterSim.Configuration("../data/epsilon0Config.ini")
        action_space_continuous=settings['action_space_continuous']

        sim_work_queues = []
        
        action_space_continuous=settings['action_space_continuous']
        if action_space_continuous:
            action_bounds = settings["action_bounds"]
            
            
        if (settings['num_available_threads'] == -1):
            input_anchor_queue = multiprocessing.Queue(settings['queue_size_limit'])
            input_anchor_queue_eval = multiprocessing.Queue(settings['queue_size_limit'])
            output_experience_queue = multiprocessing.Queue(settings['queue_size_limit'])
            eval_episode_data_queue = multiprocessing.Queue(settings['queue_size_limit'])
        else:
            input_anchor_queue = multiprocessing.Queue(settings['num_available_threads'])
            input_anchor_queue_eval = multiprocessing.Queue(settings['num_available_threads'])
            output_experience_queue = multiprocessing.Queue(settings['num_available_threads'])
            eval_episode_data_queue = multiprocessing.Queue(settings['num_available_threads'])
            
        if (settings['on_policy']): ## So that off-policy agent does not learn
            output_experience_queue = None
            
        exp_val = None
        timeout_ = 60 * 10 ### 10 min timeout
        if ("simulation_timeout" in settings):
            timeout_ = settings["simulation_timeout"]
        
        ### Try and load previous data
        if ( ((settings["load_saved_model"] == True)
              or (settings["load_saved_model"] == 'last')) and
            (settings["save_experience_memory"] == "continual")):
            
            ### load training data
            from util.SimulationUtil import getDataDirectory
            directory = getDataDirectory(settings)
            file_name_ = directory+"trainingData_" + str(settings['agent_name']) + ".json"
            print ("loading previous settings file: ", file_name_)
            if os.path.exists(file_name_):
                fp = open(file_name_, 'r')
                # print ("Train data: ", trainData)
                trainData = json.load(fp)
                fp.close()
                
                if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
                    print ("Loading training data")
                    print ("Round: ", trainData["round"])
                    # sys.exit()
            else:
                ### Actually this is the first run..
                settings["load_saved_model"] = False
                
        
        ### Keep forward models on the CPU
        if ( (("email_log_data_periodically" in settings)
            and (settings["email_log_data_periodically"] == True))
            or 
             ("save_video_to_file" in settings)):
            loggingWorkerQueue = multiprocessing.Queue(1)
            loggingWorker = LoggingWorker(settings, 
                                          collectEmailData,
                                           loggingWorkerQueue)
            loggingWorker.start()
            if ("test_movie_rendering" in settings
                and (settings["test_movie_rendering"] == True)):
                return
        
        ### These are the workers for training
        (sim_workers, sim_work_queues) = createSimWorkers(settings, input_anchor_queue, 
                                              output_experience_queue, eval_episode_data_queue, 
                                              [], [], exp_val)

        eval_sim_workers = sim_workers
        eval_sim_work_queues = sim_work_queues
        if ( 'override_sim_env_id' in settings and (settings['override_sim_env_id'] != False)):
            (eval_sim_workers, eval_sim_work_queues) = createSimWorkers(settings, input_anchor_queue_eval, 
                                                            output_experience_queue, eval_episode_data_queue, 
                                                            None, forwardDynamicsModel, exp_val,
                                                            default_sim_id=settings['override_sim_env_id'])
        else:
            input_anchor_queue_eval = input_anchor_queue
        
        
        # paramSampler = exp_val.getActor().getParamSampler()
        best_eval =-100000000.0
        mean_eval = best_eval * 10
        best_dynamicsLosses = best_eval*-1.0
        mean_dynamicsLosses = best_dynamicsLosses * 10 
            
        values = []
        discounted_values = []
        bellman_error = []
        reward_over_epoc = []
        dynamicsLosses = []
        dynamicsRewardLosses = []
        
        """
        for lw in learning_workers:
            print ("Learning worker" )
            print (lw)
        """
        if (int(settings["num_available_threads"]) > 0):
            for sw in sim_workers:
                print ("Sim worker")
                print (sw)
                sw.start()
            if ( 'override_sim_env_id' in settings and (settings['override_sim_env_id'] != False)):
                for sw in eval_sim_workers:
                    print ("Sim worker")
                    print (sw)
                    sw.start()
        """
        if ("numpy" in sys.modules):
            print ("Numpy is already loaded")
        else:
            print ("Numpy is not loaded")
        sys.exit()
        """
        ## Theano and numpy needs to be imported after the flags are set.
        import numpy as np
        import math
        import random
        import time
        import datetime
        np.random.seed(int(settings['random_seed']))
        setupLearningBackend(settings)
        
        # print ( "theano.config.mode: ", theano.config.mode)
        from simulation.SimWorker import SimWorker
        from simulation.simEpoch import simEpoch, simModelParrallel, simModelMoreParrallel
        from simulation.evalModel import evalModelParrallel, evalModel, evalModelMoreParrallel
        from simulation.collectExperience import collectExperience
        from model.ModelUtil import validBounds, fixBounds, anneal_value, getLearningData
        # from model.LearningMultiAgent import LearningMultiAgent, LearningWorker
        # from model.LearningAgent import LearningMultiAgent, LearningWorker
        from util.SimulationUtil import createEnvironment, logExperimentData, saveData
        from util.SimulationUtil import createRLAgent, createNewFDModel, processBounds
        from util.SimulationUtil import createActor, getAgentName, updateSettings
        from util.SimulationUtil import getDataDirectory, createForwardDynamicsModel, createSampler
        
        from util.ExperienceMemory import ExperienceMemory
        
        from sim.PendulumEnvState import PendulumEnvState
        from sim.PendulumEnv import PendulumEnv
        from sim.BallGame2DEnv import BallGame2DEnv
        
        model_type= settings["model_type"]
        directory= getDataDirectory(settings)
        
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        if ("pretrained_data_folder" in settings):
            import shutil
            pretrain_file = open(settings["pretrained_data_folder"], "r")
            settings_pretrain = json.load(pretrain_file)
            pretrain_file.close()
            directory_pretrain = getDataDirectory(settings_pretrain)
            for i in range(settings["perform_multiagent_training"]):
                print ("copying over pretained files: ", directory_pretrain+getAgentName()+str(i)+"_Best_actor.h5" )
                shutil.copy2(directory_pretrain+getAgentName()+str(i)+"_Best_actor.h5", directory+getAgentName()+str(i)+"_Best_actor.h5" )
                shutil.copy2(directory_pretrain+getAgentName()+str(i)+"_Best_critic.h5", directory+getAgentName()+str(i)+"_Best_critic.h5" )
                shutil.copy2(directory_pretrain+getAgentName()+str(i)+"_Best_critic_T.h5", directory+getAgentName()+str(i)+"_Best_critic_T.h5" )
                shutil.copy2(directory_pretrain+getAgentName()+str(i)+"_Best_bounds.h5", directory+getAgentName()+str(i)+"_Best_bounds.h5" )
            # sys.exit()
            
        saveData(settings, settingsFileName)
            
        ### Using a wrapper for the type of actor now
        actor = createActor(settings['environment_type'], settings, None)
        exp_val = None
        for i in range(len(action_bounds)):
            if ((action_bounds[i] != "ask_env")
                and
                not validBounds(action_bounds[i])):
                # Check that the action bounds are specified correctly
                print("Action bounds invalid: ", action_bounds[i])
                sys.exit()
            if ( (state_bounds[i] != "ask_env") 
                 and not validBounds(state_bounds[i])):
                # Probably did not collect enough bootstrapping samples to get good state bounds.
                print("State bounds invalid: ", state_bounds[i])
                state_bounds[i] = fixBounds(np.array(state_bounds[i]))
                bound_fixed = validBounds(state_bounds[i])
                print("State bounds fixed: ", bound_fixed)
                # sys.exit()
            if (not validBounds(reward_bounds[i])):
                print("Reward bounds invalid: ", reward_bounds[i])
                sys.exit()
        
        """
        if settings['action_space_continuous']:
            experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), settings['experience_length'], continuous_actions=True, settings=settings)
        else:
            experience = ExperienceMemory(len(state_bounds[0]), 1, settings['experience_length'])
            
        experience.setSettings(settings)
        """

        # mgr = multiprocessing.Manager()
        # namespace = mgr.Namespace()
        ## This needs to be done after the simulation worker processes are created
        # exp_val = createEnvironment(str(settings["sim_config_file"]), settings['environment_type'], settings, render=settings['shouldRender'], )
        # if (int(settings["num_available_threads"]) == -1
        #     or (state_bounds == "ask_env")
        #     or (action_bounds == "ask_env")): # This is okay if there is one thread only...
        exp_val = createEnvironment(settings["sim_config_file"], settings['environment_type'], settings, render=settings['shouldRender'], index=0)
        exp_val.setActor(actor)
        exp_val.getActor().init()
        exp_val.init()
        
        (state_bounds, action_bounds, settings) = processBounds(state_bounds, action_bounds, settings, exp_val)
        
        ### This is for a single-threaded Synchronous sim only.
        if (int(settings["num_available_threads"]) == -1): # This is okay if there is one thread only...
            sim_workers[0].setEnvironment(exp_val)
            sim_workers[0].start()
            if ( 'override_sim_env_id' in settings and (settings['override_sim_env_id'] != False)):
                eval_sim_workers[0].setEnvironment(exp_val)
                eval_sim_workers[0].start()
        
        model = createRLAgent(settings['agent_name'], state_bounds, discrete_actions, reward_bounds, settings, print_info=True)
        forwardDynamicsModel = None
        if (settings['train_forward_dynamics']):
            forwardDynamicsModel = createNewFDModel(settings, exp_val, model)
            # forwardDynamicsModel.setActor(actor)
            # forwardDynamicsModel.setEnvironment(exp)
            # forwardDynamicsModel.init(len(state_bounds[0]), len(action_bounds[0]), state_bounds, action_bounds, actor, None, settings)
        
        if ("train_reward_distance_metric" in settings and
            (settings['train_reward_distance_metric'] == True )):
            print ("Creating reward distance model")
            settings_ = copy.deepcopy(settings)
            settings_ = updateSettings(settings_, settings_["reward_metric_settings"])
            rewardModel = createNewFDModel(settings_, exp_val, model)
            rewardModel.setActor(actor)
            rewardModel.init(len(state_bounds[0]), len(action_bounds[0]), state_bounds, action_bounds, actor, None, settings)
        
        exp_val.finish()
        (agent, learning_workers) = createLearningAgent(settings, output_experience_queue, print_info=True)
        masterAgent = agent
        
        if ((settings['visualize_learning'] == False) 
            and (settings['save_trainData'] == True) ):
            import matplotlib
            matplotlib.use('Agg')
            print("********Using non interactive matplotlib interface")
        
        
        masterAgent.setPolicy(model)
        if (settings['train_forward_dynamics']):
            masterAgent.setForwardDynamics(forwardDynamicsModel)
            
        if ("train_reward_distance_metric" in settings and
            (settings['train_reward_distance_metric'] == True )):
            masterAgent.setRewardModel(rewardModel)
        
        print ("masterAgent state bounds: ", masterAgent.getStateBounds())
        print ("state bounds: ", state_bounds)
        ### If the policy loaded state bounds use those
        state_bounds = masterAgent.getStateBounds()
        action_bounds = masterAgent.getActionBounds()
        reward_bounds = masterAgent.getRewardBounds()
        settings['state_bounds'] = masterAgent.getStateBounds()
        settings['action_bounds'] = masterAgent.getActionBounds()
        settings['reward_bounds'] = masterAgent.getRewardBounds()
        # sys.exit()
        
        
        tmp_p=1.0
        message={}
        if ( settings['load_saved_model'] ):
            tmp_p = settings['min_epsilon']
        data = getLearningData(masterAgent, settings, tmp_p)
        message['type'] = 'Update_Policy'
        message['data'] = data
        for m_q in sim_work_queues:
            print("trainModel: Sending current network parameters: ", m_q)
            m_q.put(message, timeout=timeout_)
        
        if ( int(settings["num_available_threads"]) ==  -1):
           experience, state_bounds, reward_bounds, action_bounds, (states, actions, resultStates, rewards_, falls_, G_ts_, exp_actions, advantage_), experiencefd = collectExperience(actor, exp_val, masterAgent, settings,
                           sim_work_queues=None, 
                           eval_episode_data_queue=None)
            
        else:
            if (settings['on_policy'] == True):
                
                experience, state_bounds, reward_bounds, action_bounds, (states, actions, resultStates, rewards_, falls_, G_ts_, exp_actions, advantage_, datas), experiencefd = collectExperience(actor, None, masterAgent, settings,
                           sim_work_queues=sim_work_queues, 
                           eval_episode_data_queue=eval_episode_data_queue)
            else:
                experience, state_bounds, reward_bounds, action_bounds, (states, actions, resultStates, rewards_, falls_, G_ts_, exp_actions, advantage_, datas), experiencefd = collectExperience(actor, None, masterAgent, settings,
                           sim_work_queues=input_anchor_queue, 
                           eval_episode_data_queue=eval_episode_data_queue)
            masterAgent.setExperience(experience)
        fd_epxerience_length = settings['experience_length']
        if ("fd_experience_length" in settings):
            fd_epxerience_length = settings["fd_experience_length"]
        if ( 'keep_seperate_fd_exp_buffer' in settings and (settings['keep_seperate_fd_exp_buffer'])):
            state_bounds__ = getFDStateSize(settings)
            ### Might be some memory expenditure here with a double copy
            masterAgent.setFDExperience(copy.deepcopy(experiencefd))
            masterAgent.setFDStateBounds(copy.deepcopy(state_bounds__))
            masterAgent.setFDActionBounds(copy.deepcopy(action_bounds))
            masterAgent.setFDRewardBounds(copy.deepcopy(reward_bounds))
            
        # print ("masterAgent.getFDExperience().getStateBounds() shape : ", masterAgent.getFDExperience().getStateBounds().shape)
        # sys.exit()
        
        if (settings["load_saved_model"] == True and
            (settings["save_experience_memory"] == "continual")):
            ### load exp mem
            if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
                print ("Loading Experience memory")
            file_name=directory+getAgentName()+"_expBufferInit.hdf5"
            masterAgent.getExperience().loadFromFile(file_name)
            if (settings['train_forward_dynamics']):
                if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
                    print ("Loading Experience FD memory")
                file_name=directory+getAgentName()+"_FD_expBufferInit.hdf5"
                masterAgent.getFDExperience().loadFromFile(file_name)
                # print ("****** state bounds mean: ", np.mean(masterAgent.getFDExperience().getStateBounds()))
                print ("****** fd exp mem insters ***: ", masterAgent.getFDExperience().inserts())
            
        if (not validBounds(action_bounds)):
            # Check that the action bounds are spcified correctly
            print("Action bounds invalid: ", action_bounds)
            sys.exit()
        if (not validBounds(state_bounds)):
            # Probably did not collect enough bootstrapping samples to get good state bounds.
            print("State bounds invalid: ", state_bounds)
            state_bounds = fixBounds(np.array(state_bounds))
            bound_fixed = validBounds(state_bounds)
            print("State bounds fixed: ", bound_fixed)
            model.setStateBounds(state_bounds)
            masterAgent.getExperience().setStateBounds(copy.deepcopy(model.getStateBounds()))
            # sys.exit()
        if (not validBounds(reward_bounds)):
            print("Reward bounds invalid: ", reward_bounds)
            sys.exit()
        
        """
        if action_space_continuous:
            model = createRLAgent(settings['agent_name'], state_bounds, action_bounds, reward_bounds, settings)
        else:
            model = createRLAgent(settings['agent_name'], state_bounds, discrete_actions, reward_bounds, settings)
        """
        if ( settings['load_saved_model'] or (settings['load_saved_model'] == 'network_and_scales') ): ## Transfer learning
            masterAgent.setStateBounds(state_bounds)
            masterAgent.setRewardBounds(reward_bounds)
            masterAgent.setActionBounds(action_bounds)
            masterAgent.setSettings(settings)
        else: ## Normal
            masterAgent.setStateBounds(state_bounds)
            masterAgent.setActionBounds(action_bounds)
            masterAgent.setRewardBounds(reward_bounds)
            
        if (settings["save_experience_memory"] == True):
            print ("Saving initial experience memory")
            file_name=directory+getAgentName()+"_expBufferInit.hdf5"
            masterAgent.getExperience().saveToFile(file_name)
            if (settings['train_forward_dynamics'] and
                settings['keep_seperate_fd_exp_buffer']):
                if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
                    print ("Saving Experience FD memory")
                file_name=directory+getAgentName()+"_FD_expBufferInit.hdf5"
                masterAgent.getFDExperience().saveToFile(file_name)
        
        masterAgent_message_queue = multiprocessing.Queue(settings['epochs'])
        
        ## Now everything related to the exp memory needs to be updated
        bellman_errors=[]
        masterAgent.setPolicy(model)
        print("Master agent state bounds: ",  repr(masterAgent.getStateBounds()))
        for sw in sim_workers: # Need to update parameter bounds for models
            print ("exp: ", sw._exp)
            print ("sw modle: ", sw._model.getPolicy()) 
            
            
        ## If not on policy
        if ( not settings['on_policy']):
            for lw in learning_workers:
                # lw._agent.setPolicy(copy.deepcopy(model))
                lw._agent.setPolicy(model)
                # lw.setLearningNamespace(learningNamespace)
                lw.setMasterAgentMessageQueue(masterAgent_message_queue)
                lw.updateExperience(experience)
                # lw.updateModel()
                print ("ls policy: ", lw._agent.getPolicy())
                
                lw.start()
                
            
        del model
        ## Give gloabl access to processes to they can be terminated when ctrl+c is pressed
        global sim_processes
        sim_processes = sim_workers
        global learning_processes
        learning_processes = learning_workers
        global _input_anchor_queue
        _input_anchor_queue = input_anchor_queue
        global _output_experience_queue
        _output_experience_queue = output_experience_queue
        global _eval_episode_data_queue
        _eval_episode_data_queue = eval_episode_data_queue
        global _sim_work_queues
        _sim_work_queues = sim_work_queues
        
        if ( settings['save_trainData'] or settings['visualize_learning']):
            from RLVisualize import RLVisualize
            if (settings['train_forward_dynamics']
                or settings['debug_critic']
                or settings['debug_actor']):
                from NNVisualize import NNVisualize
            
        if settings['visualize_learning']:
            title = settings['agent_name']
            k = title.rfind(".") + 1
            if (k > len(title)): ## name does not contain a .
                k = 0 
            title = str(settings['sim_config_file'])
            if (settings['environment_type'] == "open_AI_Gym"):
                settings['environment_type'] = settings['sim_config_file']
            rlv = RLVisualize(title=title + " agent on " + str(settings['environment_type']), settings=settings)
            rlv.setInteractive()
            rlv.init()
        if (settings['train_forward_dynamics']):
            if settings['visualize_learning']:
                title = settings['forward_dynamics_model_type']
                k = title.rfind(".") + 1
                if (k > len(title)): ## name does not contain a .
                    k = 0 
                title = title[k:]
                nlv = NNVisualize(title=str("Dynamics Model") + " with " + title, settings=settings)
                nlv.setInteractive()
                nlv.init()
            if (settings['train_reward_predictor']):
                if settings['visualize_learning']:
                    title = settings['forward_dynamics_model_type']
                    k = title.rfind(".") + 1
                    if (k > len(title)): ## name does not contain a .
                        k = 0 
                    
                    title = title[k:]
                    rewardlv = NNVisualize(title=str("Reward Model") + " with " + title, settings=settings)
                    rewardlv.setInteractive()
                    rewardlv.init()
                 
        if (settings['debug_critic']):
            criticLosses = []
            criticRegularizationCosts = [] 
            if (settings['visualize_learning']):
                title = settings['agent_name']
                k = title.rfind(".") + 1
                if (k > len(title)): ## name does not contain a .
                    k = 0 
                title = title[k:]
                critic_loss_viz = NNVisualize(title=str("Critic Loss") + " with " + title)
                critic_loss_viz.setInteractive()
                critic_loss_viz.init()
                critic_regularization_viz = NNVisualize(title=str("Critic Reg Cost") + " with " + title)
                critic_regularization_viz.setInteractive()
                critic_regularization_viz.init()
            
        if (settings['debug_actor']):
            actorLosses = []
            actorRegularizationCosts = []            
            if (settings['visualize_learning']):
                title = settings['agent_name']
                k = title.rfind(".") + 1
                if (k > len(title)): ## name does not contain a .
                    k = 0 
                title = title[k:]
                actor_loss_viz = NNVisualize(title=str("Actor Loss") + " with " + title)
                actor_loss_viz.setInteractive()
                actor_loss_viz.init()
                actor_regularization_viz = NNVisualize(title=str("Actor Reg Cost") + " with " + title)
                actor_regularization_viz.setInteractive()
                actor_regularization_viz.init()
                
        settings["logger_instance"] = exp_logger
        settings["round"] = int(trainData["round"])
        masterAgent.setSettings(settings)

        if (False ):
            print("State Bounds:", masterAgent.getStateBounds())
            print("Action Bounds:", masterAgent.getActionBounds())
            
            print("Exp State Bounds: ", masterAgent.getExperience().getStateBounds())
            print("Exp Action Bounds: ", masterAgent.getExperience().getActionBounds())
            
        if ("pretrain_critic" in settings and (settings["pretrain_critic"] > 0)
            and (trainData["round"] == 0)):
            pretrainCritic(masterAgent, states, actions, resultStates, rewards_, 
                           falls_, G_ts_, exp_actions, advantage_, sim_work_queues, 
                           eval_episode_data_queue)
            
        if ("pretrain_fd" in settings and (settings["pretrain_fd"] > 0)
            and (trainData["round"] == 0)):
            pretrainFD(masterAgent, states, actions, resultStates, rewards_, 
                           falls_, G_ts_, exp_actions, advantage_, sim_work_queues, 
                           eval_episode_data_queue)
        
        print ("Starting first round: ", trainData["round"])
        if (settings['on_policy']):
            sim_epochs_ = epochs
            # epochs = 1
        # for round_ in range(0,rounds):
        while (trainData["round"] <= rounds):
            trainData["round"] = int(trainData["round"])
            settings["round"] = int(trainData["round"])
            masterAgent.setSettings(settings)
            # p = math.fabs(settings['initial_temperature'] / (math.log(round_*round_) - round_) )
            # p = (settings['initial_temperature'] / (math.log(round_))) 
            # p = ((settings['initial_temperature']/math.log(round_))/math.log(rounds))
            if ( 'annealing_schedule' in settings and (settings['annealing_schedule'] != False)):
                p = anneal_value(float(trainData["round"]/rounds), settings_=settings)
            else:
                p = ((settings['initial_temperature']/math.log(trainData["round"]+2))) 
            # p = ((rounds - trainData["round"])/rounds) ** 2
            p = max(settings['min_epsilon'], min(settings['epsilon'], p)) # Keeps it between 1.0 and 0.2
            if ( settings['load_saved_model'] == True):
                p = settings['min_epsilon']
                
            # pr = cProfile.Profile()
            for epoch in range(epochs):
                if (settings['on_policy']):
                    
                    # if ( settings['num_available_threads'] > 0 ):  
                    if ("skip_rollouts" in settings and 
                        (settings["skip_rollouts"] == True)):
                        out = (([],[],[],[],[],[],[],[]), [], [], [])
                    
                    else:
                        if (settings['on_policy'] == "fast"):
                            out = simModelMoreParrallel( sw_message_queues=input_anchor_queue,
                                                       model=masterAgent, settings=settings, 
                                                       eval_episode_data_queue=eval_episode_data_queue, 
                                                       anchors=settings['num_on_policy_rollouts']
                                                       ,p=p)
                        else:
                            out = simModelParrallel( sw_message_queues=sim_work_queues,
                                                       model=masterAgent, settings=settings, 
                                                       eval_episode_data_queue=eval_episode_data_queue, 
                                                       anchors=settings['num_on_policy_rollouts']
                                                       ,p=p)
                    
                    (tuples, discounted_sum, q_value, evalData) = out
                    (__states, __actions, __result_states, __rewards, __falls, __G_ts, advantage__, exp_actions__, datas__) = tuples
                    if ( ('anneal_on_policy' in settings) and settings['anneal_on_policy']):  
                        p_tmp_ = p 
                    else:
                        p_tmp_ = 1.0
                    
                    for i in range(1):
                        masterAgent.train(_states=__states, _actions=__actions, _rewards=__rewards, _result_states=__result_states,
                                           _falls=__falls, _advantage=advantage__, _exp_actions=exp_actions__, _G_t=__G_ts, p=p_tmp_, datas=datas__)
                    masterAgent.reset()
                    
                    data = getLearningData(masterAgent, settings, p)
                    message['data'] = data
                    
                    if ("skip_rollouts" in settings and 
                        (settings["skip_rollouts"] == True)):
                        pass
                    else:
                        for m_q in sim_work_queues:
                            ## block on full queue
                            m_q.put(message, timeout=timeout_)
                        
                        if ( 'override_sim_env_id' in settings and (settings['override_sim_env_id'] != False)):
                            for m_q in eval_sim_work_queues:
                                ## block on full queue
                                m_q.put(message, timeout=timeout_)
                    
                    # states, actions, result_states, rewards, falls, G_ts, exp_actions = masterAgent.getExperience().get_batch(batch_size)
                    # print ("Batch size: " + str(batch_size))
                else:
                    episodeData = {}
                    episodeData['data'] = epoch
                    episodeData['type'] = 'sim'
                    input_anchor_queue.put(episodeData, timeout=timeout_)
                
                # pr.enable()
                # print ("Current Tuple: " + str(learningNamespace.experience.current()))
                # print ("masterAgent.getExperience().samples() >= batch_size: ", masterAgent.getExperience().samples(), " >= ", batch_size)
                error = 0
                rewards = 0
                if masterAgent.samples() >= batch_size:
                    states, actions, result_states, rewards, falls, G_ts, exp_actions, advantage = masterAgent.get_batch(batch_size, 0)
                    # print ("Batch size: " + str(batch_size))
                    masterAgent.reset()
                    error = masterAgent.bellman_error()
                    # error = np.mean(np.fabs(error), axis=1)
                    # print ("Error: ", error)
                    # bellman_errors.append(np.mean(np.fabs(error)))
                    bellman_errors.append(error)
                    if (settings['debug_critic']):
                        masterAgent.reset()
                        if ((("train_LSTM" in settings)
                        and (settings["train_LSTM"] == True))
                            or (("train_LSTM_Critic" in settings)
                            and (settings["train_LSTM_Critic"] == True))):
                            batch_size_lstm = 4
                            if ("lstm_batch_size" in settings):
                                batch_size_lstm = settings["lstm_batch_size"][1]
                            states_, actions_, result_states_, rewards_, falls_, G_ts_, exp_actions, advantage_ = masterAgent.getExperience().get_multitask_trajectory_batch(batch_size=min(batch_size_lstm, masterAgent.getExperience().samplesTrajectory()))
                            loss__ = masterAgent.getPolicy().get_critic_loss(states_, actions_, rewards_, result_states_)
                        else:
                            loss__ = masterAgent.getPolicy().get_critic_loss(states, actions, rewards, result_states)
                        criticLosses.append(loss__)
                        regularizationCost__ = masterAgent.getPolicy().get_critic_regularization()
                        criticRegularizationCosts.append(regularizationCost__)
                        
                    if (settings['debug_actor']):
                        """
                        print( "Advantage: ", masterAgent.getPolicy()._get_advantage())
                        print("Policy prob: ", masterAgent.getPolicy()._q_action())
                        print("Policy log prob: ", masterAgent.getPolicy()._get_log_prob())
                        print( "Actor loss: ", masterAgent.getPolicy()._get_action_diff())
                        """
                        masterAgent.reset()
                        # loss__ = [loss___ masterAgent.getPolicy().get_actor_loss(states, actions, rewards, result_states, advantage)
                        loss__ = [p_.getPolicy().get_actor_loss(states, actions, rewards, result_states, advantage) for p_ in masterAgent.getAgents() ]
                        actorLosses.append(np.mean(loss__))
                        regularizationCost__ = [p_.getPolicy().get_actor_regularization() for p_ in masterAgent.getAgents() ]
                        actorRegularizationCosts.append(np.mean(regularizationCost__))
                    
                    if not all(np.isfinite(np.mean(error, axis=0))):
                        print ("Bellman Error is Nan: " + str(error) + str(np.isfinite(error)))
                        # if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
                        print ("States: " + str(states) + " ResultsStates: " + str(result_states) + " Rewards: " + str(rewards) + " Actions: " + str(actions) + " Falls: ", str(falls))
                        sys.exit()
                    
                    error = np.mean(np.fabs(error), axis=1)
                    if np.mean(error) > 10000:
                        print ("Error to big: ")
                        if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
                            print (states, actions, rewards, result_states)
                        
                if (settings['train_forward_dynamics']):
                    if ( 'keep_seperate_fd_exp_buffer' in settings 
                         and (settings['keep_seperate_fd_exp_buffer'])):
                        states, actions, result_states, rewards, falls, G_ts, exp_actions, advantage = masterAgent.getFDBatch(batch_size)
                    masterAgent.reset()
                    if (("train_LSTM_FD" in settings)
                        and (settings["train_LSTM_FD"] == True)):
                        batch_size_lstm_fd = 4
                        if ("lstm_batch_size" in settings):
                            batch_size_lstm_fd = settings["lstm_batch_size"][0]
                        ### This can consume a lot of memory if trajectories are long...
                        state_, action_, resultState_, reward_, fall_, G_ts_, exp_actions, advantage_ = masterAgent.getFDExperience().get_multitask_trajectory_batch(batch_size=2)
                        dynamicsLoss = masterAgent.getForwardDynamics().bellman_error(state_, action_, resultState_, reward_)
                    else:
                        dynamicsLoss = masterAgent.getForwardDynamics().bellman_error(states, actions, result_states, rewards)
                    if (type(dynamicsLoss) == 'list'):
                        dynamicsLoss = np.mean([np.mean(np.fabs(dfl)) for dfl in dynamicsLoss])
                    else:
                        dynamicsLoss = np.mean(np.fabs(dynamicsLoss))
                    dynamicsLosses.append(dynamicsLoss)
                    if (settings['train_reward_predictor']):
                        masterAgent.reset()
                        if (("train_LSTM_Reward" in settings)
                            and (settings["train_LSTM_Reward"] == True)):
                            batch_size_lstm_fd = 4
                            if ("lstm_batch_size" in settings):
                                batch_size_lstm_fd = settings["lstm_batch_size"][0]
                            ### This can consume a lot of memory if trajectories are long...
                            state_, action_, resultState_, reward_, fall_, G_ts_, exp_actions, advantage_ = masterAgent.getFDExperience().get_multitask_trajectory_batch(batch_size=2)
                            dynamicsRewardLoss = masterAgent.getForwardDynamics().reward_error(state_, action_, resultState_, reward_)
                        else:
                            dynamicsRewardLoss = masterAgent.getForwardDynamics().reward_error(states, actions, result_states, rewards)
                        
                        if (type(dynamicsRewardLoss) == 'list'):
                            dynamicsRewardLoss = np.mean([np.mean(np.fabs(drl)) for drl in dynamicsRewardLoss])
                        else:
                            dynamicsRewardLoss = np.mean(np.fabs(dynamicsRewardLoss))

                        dynamicsRewardLosses.append(dynamicsRewardLoss)
                    if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
                        if (settings['train_forward_dynamics']):
                            print ("Round: " + str(trainData["round"]) + " of ", rounds,  ", Epoch: " + str(epoch) + " p: " + str(p) + " With mean reward: " + str(np.mean(rewards)) + " bellman error: " + str(error) + " ForwardPredictionLoss: " + str(dynamicsLoss))
                        else:
                            print ("Round: " + str(trainData["round"]) + " of ", rounds,  ", Epoch: " + str(epoch) + " p: " + str(p) + " With mean reward: " + str(np.mean(rewards)) + " bellman error: " + str(error))
                    # discounted_values.append(discounted_sum)
                    
                if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
                    print ("Master agent experience size: " + str(masterAgent.samples()))
                # print ("**** Master agent experience size: " + str(learning_workers[0]._agent._expBuff.samples()))
                
                if (settings['on_policy'] is False):
                    ## There could be stale policy parameters in here, use the last set put in the queue
                    data = None
                    while (not masterAgent_message_queue.empty()):
                        ## Don't block
                        try:
                            data = masterAgent_message_queue.get(False)
                        except Exception as inst:
                            # print ("training: In model parameter message queue empty: ", masterAgent_message_queue.qsize())
                            pass
                    if (not (data == None) ):
                        # print ("Data: ", data)
                        masterAgent.setExperience(data[0])
                        masterAgent.getPolicy().setNetworkParameters(data[1])
                        masterAgent.setStateBounds(masterAgent.getExperience().getStateBounds())
                        masterAgent.setActionBounds(masterAgent.getExperience().getActionBounds())
                        masterAgent.setRewardBounds(masterAgent.getExperience().getRewardBounds())
                        if (settings['train_forward_dynamics']):
                            masterAgent.getForwardDynamics().setNetworkParameters(data[2])
                            if ( 'keep_seperate_fd_exp_buffer' in settings and (settings['keep_seperate_fd_exp_buffer'])):
                                masterAgent.setFDExperience(data[3])
                        
                # experience = learningNamespace.experience
                # actor.setExperience(experience)
                """
                pr.disable()
                f = open('x.prof', 'a')
                pstats.Stats(pr, stream=f).sort_stats('time').print_stats()
                f.close()
                """
            
                # this->_actor->iterate();
            ## This will let me know which part of learning is going slower training updates or simulation
            if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
                print ("sim queue size: ", input_anchor_queue.qsize() )
                if ( output_experience_queue != None):
                    print ("exp tuple queue size: ", output_experience_queue.qsize())
            
            if (not settings['on_policy']):
                # masterAgent.getPolicy().setNetworkParameters(learningNamespace.agentPoly)
                # masterAgent.setExperience(learningNamespace.experience)
                masterAgent.reset()
                data = ('Update_Policy', p,
                        masterAgent.getStateBounds(),
                        masterAgent.getActionBounds(),
                        masterAgent.getRewardBounds(),
                        masterAgent.getPolicy().getNetworkParameters())
                if (settings['train_forward_dynamics']):
                    # masterAgent.getForwardDynamics().setNetworkParameters(learningNamespace.forwardNN)
                    data = ('Update_Policy', p, 
                            masterAgent.getStateBounds(),
                            masterAgent.getActionBounds(),
                            masterAgent.getRewardBounds(),
                            masterAgent.getPolicy().getNetworkParameters(),
                             masterAgent.getForwardDynamics().getNetworkParameters())
                message['type'] = 'Update_Policy'
                message['data'] = data
                for m_q in sim_work_queues:
                    ## Don't block on full queue
                    try:
                        m_q.put(message, False)
                    except: 
                        print ("SimWorker model parameter message queue full: ", m_q.qsize())
                if ( 'override_sim_env_id' in settings and (settings['override_sim_env_id'] != False)):
                    for m_q in eval_sim_work_queues:
                        ## Don't block on full queue
                        try:
                            m_q.put(message, False)
                        except: 
                            print ("SimWorker model parameter message queue full: ", m_q.qsize())
              
            if (trainData["round"] % settings['plotting_update_freq_num_rounds']) == 0:
                # Running less often helps speed learning up.
                # Sync up sim actors
                
                # if (settings['on_policy'] or ((settings['num_available_threads'] == 1))):
                #     mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error, mean_eval, std_eval = evalModel(actor, exp_val, masterAgent, discount_factor, 
                #                                         anchors=settings['eval_epochs'], action_space_continuous=action_space_continuous, settings=settings)
                # else:
                if ("skip_rollouts" in settings and 
                        (settings["skip_rollouts"] == True)):
                    mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error, mean_eval, std_eval = 0,0,0,0,0,0,0,0
                else:
                    if ( ( settings["eval_epochs"] == "stochastic")):
                        rewards__=[]
                        discounted_sum__=[]
                        value__=[]
                        discount_error__ = []
                        reward_over_epocs = []
                        for tr in range(len(__rewards)):
                            for agent_ in range(len(masterAgent.getAgents())): 
                                rewards__.append(np.array(__rewards[tr]).flatten()[agent_::len(masterAgent.getAgents())])
                                # discounted_sum__.append(np.array(discounted_sum).flatten()[agent_::len(masterAgent.getAgents())])
                                # value__.append(np.array(q_value).flatten()[agent_::len(masterAgent.getAgents())])
                                # discount_error__.append(discounted_sum__[agent_] - value__[agent_])
                            reward_over_epocs.append(np.mean(np.array(rewards__), axis=1))
                        # bellman_errors.append(error)
                        # mean_discount_error.append(np.mean(np.fabs(discount_error__), axis=1))
                        # std_discount_error.append(np.std(discount_error__, axis=1))
                        
                        mean_reward = np.mean(reward_over_epocs)
                        std_reward = np.std(reward_over_epocs)
                        mean_bellman_error = 0
                        std_bellman_error = 0
                        mean_discount_error = 0
                        std_discount_error = 0
                        mean_eval = np.mean(reward_over_epocs)
                        std_eval = np.std(reward_over_epocs)
                    elif (settings['on_policy'] == True ):
                        mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error, mean_eval, std_eval, otherMetrics = evalModelParrallel( input_anchor_queue=eval_sim_work_queues,
                                                                   model=masterAgent, settings=settings, eval_episode_data_queue=eval_episode_data_queue, anchors=settings['eval_epochs'])
                    elif (settings['on_policy'] == "fast"):
                        mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error, mean_eval, std_eval, otherMetrics = evalModelMoreParrallel( input_anchor_queue=input_anchor_queue_eval,
                                                                   model=masterAgent, settings=settings, eval_episode_data_queue=eval_episode_data_queue, anchors=settings['eval_epochs'])
                    else:
                        mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error, mean_eval, std_eval, otherMetrics = evalModelParrallel( input_anchor_queue=input_anchor_queue_eval,
                                                                model=masterAgent, settings=settings, eval_episode_data_queue=eval_episode_data_queue, anchors=settings['eval_epochs'])
                """
                for sm in sim_workers:
                    sm.setP(0.0)
                for lw in learning_workers:
                    output_experience_queue.put(None, timeout=timeout_)
                mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error = evalModelParrallel(input_anchor_queue, output_experience_queue, discount_factor, 
                                                    anchors=_anchors[:settings['eval_epochs']], action_space_continuous=action_space_continuous, settings=settings)
                                                    """
                print ("round_, p, mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error")
                print (trainData["round"], p, mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error)
                if np.mean(mean_bellman_error) > 10000:
                    print ("Error to big: ")
                else:
                    if (settings['train_forward_dynamics']):
                        mean_dynamicsLosses = np.mean(dynamicsLosses)
                        std_dynamicsLosses = np.std(dynamicsLosses)
                        dynamicsLosses = []
                        if (settings['train_reward_predictor']):
                            mean_dynamicsRewardLosses = np.mean(dynamicsRewardLosses)
                            std_dynamicsRewardLosses = np.std(dynamicsRewardLosses)
                            dynamicsRewardLosses = []
                        
                        
                    logExperimentData(trainData, "falls", np.mean(otherMetrics["falls"]), settings)
                    logExperimentData(trainData, "mean_reward", mean_reward, settings)
                    logExperimentData(trainData, "std_reward", std_reward, settings)
                    logExperimentData(trainData, "anneal_p", p, settings)
                    logExperimentData(trainData, "mean_bellman_error", np.array([np.mean(er_) for er_ in np.fabs(bellman_errors[0])]), settings)
                    logExperimentData(trainData, "std_bellman_error", np.array([np.std(er_) for er_ in bellman_errors[0]]), settings)
                    bellman_errors=[]
                    logExperimentData(trainData, "mean_discount_error", mean_discount_error, settings)
                    logExperimentData(trainData, "std_discount_error", std_discount_error, settings)
                    logExperimentData(trainData, "mean_eval", mean_eval, settings)
                    logExperimentData(trainData, "std_eval", std_eval, settings)
                    # error = np.mean(np.fabs(error), axis=1)
                    # trainData["std_bellman_error"].append(std_bellman_error)
                    if (settings['train_forward_dynamics']):
                        logExperimentData(trainData, "mean_forward_dynamics_loss", mean_dynamicsLosses, settings)
                        logExperimentData(trainData, "std_forward_dynamics_loss", std_dynamicsLosses, settings)
                        if (settings['train_reward_predictor']):
                            logExperimentData(trainData, "mean_forward_dynamics_reward_loss", mean_dynamicsRewardLosses, settings)
                            logExperimentData(trainData, "std_forward_dynamics_reward_loss", std_dynamicsRewardLosses, settings)
                            
                    ### Lets always save a figure for the learning...
                    if ( settings['save_trainData'] and (not settings['visualize_learning'])):
                        rlv_ = RLVisualize(title=str(settings['sim_config_file']) + " agent on " + str(settings['environment_type']), settings=settings)
                        rlv_.init()
                        rlv_.updateBellmanError(np.array(trainData["mean_bellman_error"]), np.array(trainData["std_bellman_error"]))
                        rlv_.updateReward(np.array(trainData["mean_reward"]), np.array(trainData["std_reward"]))
                        rlv_.updateDiscountError(np.fabs(trainData["mean_discount_error"]), np.array(trainData["std_discount_error"]))
                        rlv_.redraw()
                        rlv_.saveVisual(directory+getAgentName())
                        rlv_.finish()
                        del rlv_
                    if settings['visualize_learning']:
                        rlv.updateBellmanError(np.array(trainData["mean_bellman_error"]), np.array(trainData["std_bellman_error"]))
                        rlv.updateReward(np.array(trainData["mean_reward"]), np.array(trainData["std_reward"]))
                        rlv.updateDiscountError(np.fabs(trainData["mean_discount_error"]), np.array(trainData["std_discount_error"]))
                        rlv.redraw()
                        rlv.setInteractiveOff()
                        rlv.saveVisual(directory+getAgentName())
                        rlv.setInteractive()
                    
                    if (settings['train_forward_dynamics'] and settings['save_trainData']
                        and (not settings['visualize_learning'])):
                        nlv_ = NNVisualize(title=str("Dynamics Model") + " with " + str(settings['sim_config_file']), settings=settings)
                        nlv_.init()
                        nlv_.updateLoss(np.array(trainData["mean_forward_dynamics_loss"]), np.array(trainData["std_forward_dynamics_loss"]))
                        nlv_.redraw()
                        nlv_.saveVisual(directory+"trainingGraphNN")
                        nlv_.finish()
                        del nlv_
                        if (settings['train_reward_predictor']):
                            rewardlv_ = NNVisualize(title=str("Reward Model") + " with " + str(settings['sim_config_file']), settings=settings)
                            rewardlv_.init()
                            rewardlv_.updateLoss(np.array(trainData["mean_forward_dynamics_reward_loss"]), np.array(trainData["std_forward_dynamics_reward_loss"]))
                            rewardlv_.redraw()
                            rewardlv_.saveVisual(directory+"rewardTrainingGraph")
                            rewardlv_.finish()
                            del rewardlv_
                    if (settings['visualize_learning'] and settings['train_forward_dynamics']):
                        nlv.updateLoss(np.array(trainData["mean_forward_dynamics_loss"]), np.array(trainData["std_forward_dynamics_loss"]))
                        nlv.redraw()
                        nlv.setInteractiveOff()
                        nlv.saveVisual(directory+"trainingGraphNN")
                        nlv.setInteractive()
                        if (settings['train_reward_predictor']):
                            rewardlv.updateLoss(np.array(trainData["mean_forward_dynamics_reward_loss"]), np.array(trainData["std_forward_dynamics_reward_loss"]))
                            rewardlv.redraw()
                            rewardlv.setInteractiveOff()
                            rewardlv.saveVisual(directory+"rewardTrainingGraph")
                            rewardlv.setInteractive()
                    if (settings['debug_critic']):
                        
                        mean_criticLosses = np.mean([np.mean(cl) for cl in criticLosses])
                        std_criticLosses = np.mean([np.std(acl) for acl in criticLosses])
                        logExperimentData(trainData, "mean_critic_loss", mean_critic_loss, settings)
                        logExperimentData(trainData, "std_critic_loss", std_critic_loss, settings)
                        criticLosses = []
                        if (settings['visualize_learning']):
                            critic_loss_viz.updateLoss(np.array(trainData["mean_critic_loss"]), np.array(trainData["std_critic_loss"]))
                            critic_loss_viz.redraw()
                            critic_loss_viz.setInteractiveOff()
                            critic_loss_viz.saveVisual(directory+"criticLossGraph")
                            critic_loss_viz.setInteractive()
                        
                        mean_criticRegularizationCosts = np.mean(criticRegularizationCosts)
                        std_criticRegularizationCosts = np.std(criticRegularizationCosts)
                        logExperimentData(trainData, "mean_critic_regularization_cost", mean_critic_regularization_cost, settings)
                        logExperimentData(trainData, "std_critic_regularization_cost", std_critic_regularization_cost, settings)
                        criticRegularizationCosts = []
                        if (settings['visualize_learning']):
                            critic_regularization_viz.updateLoss(np.array(trainData["mean_critic_regularization_cost"]), np.array(trainData["std_critic_regularization_cost"]))
                            critic_regularization_viz.redraw()
                            critic_regularization_viz.setInteractiveOff()
                            critic_regularization_viz.saveVisual(directory+"criticRegularizationGraph")
                            critic_regularization_viz.setInteractive()
                        
                    if (settings['debug_actor']):
                        
                        mean_actorLosses = np.mean([np.mean(acL) for acL in actorLosses])
                        std_actorLosses = np.mean([np.std(acl) for acl in actorLosses])
                        logExperimentData(trainData, "mean_actor_loss", mean_actorLosses, settings)
                        logExperimentData(trainData, "std_actor_loss", std_actorLosses, settings)
                        actorLosses = []
                        if (settings['visualize_learning']):
                            actor_loss_viz.updateLoss(np.array(trainData["mean_actor_loss"]), np.array(trainData["std_actor_loss"]))
                            actor_loss_viz.redraw()
                            actor_loss_viz.setInteractiveOff()
                            actor_loss_viz.saveVisual(directory+"actorLossGraph")
                            actor_loss_viz.setInteractive()
                        
                        mean_actorRegularizationCosts = np.mean(actorRegularizationCosts)
                        std_actorRegularizationCosts = np.std(actorRegularizationCosts)
                        logExperimentData(trainData, "mean_actor_regularization_cost", mean_actorRegularizationCosts, settings)
                        logExperimentData(trainData, "std_actor_regularization_cost", std_actorRegularizationCosts, settings)
                        actorRegularizationCosts = []
                        if (settings['visualize_learning']):
                            actor_regularization_viz.updateLoss(np.array(trainData["mean_actor_regularization_cost"]), np.array(trainData["std_actor_regularization_cost"]))
                            actor_regularization_viz.redraw()
                            actor_regularization_viz.setInteractiveOff()
                            actor_regularization_viz.saveVisual(directory+"actorRegularizationGraph")
                            actor_regularization_viz.setInteractive()
                """for lw in learning_workers:
                    lw.start()
                   """     
                ## Visulaize some stuff if you want to
                if (int(settings["num_available_threads"]) == -1 
                    # or (int(settings["num_available_threads"]) == 1)
                    ): # This is okay if there is one thread only...
                    exp_val.updateViz(actor, masterAgent, directory, p=p)
                
                
            if (trainData["round"] % settings['saving_update_freq_num_rounds']) == 0:
            
                if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['hyper_train']):
                    print ("Saving current masterAgent")
                masterAgent.saveTo(directory)
                
                if ( settings['train_forward_dynamics'] and 
                     (mean_dynamicsLosses < best_dynamicsLosses)):
                    best_dynamicsLosses = mean_dynamicsLosses
                    if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['hyper_train']):
                        print ("Saving BEST current forward dynamics agent: " + str(best_dynamicsLosses))
                    masterAgent.saveTo(directory, bestFD=True)
                        
                if (mean_eval > best_eval):
                    best_eval = mean_eval
                    if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['hyper_train']):
                        print ("Saving BEST current agent: " + str(best_eval))
                    masterAgent.saveTo(directory, bestPolicy=True)
                    
                fp = open(directory+"trainingData_" + str(settings['agent_name']) + ".json", 'w')
                # print ("Train data: ", trainData)
                ## because json does not serialize np.float32 
                """
                for key in trainData:
                    # print ("trainData[",key,"]", trainData[key])
                    if (key == 'error'):
                        continue
                    elif (type(trainData[key]) is list):
                        trainData[key] = [float(i) for i in trainData[key]]
                    else:
                        trainData[key] = float(trainData[key])
                """
                from util.utils import NumpyEncoder 
                # print ("trainData: ", trainData)
                json.dump(trainData, fp, cls=NumpyEncoder)
                fp.close()
                # draw data
                """
                t0 = time.time()
                if (settings["save_experience_memory"] == "continual"
                    or(settings["save_experience_memory"] == "all")):
                    if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
                        print ("Saving Experience memory")
                    file_name=directory+getAgentName()+"_expBufferInit.hdf5"
                    masterAgent.getExperience().saveToFile(file_name)
                    if (settings['train_forward_dynamics']):
                        if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
                            print ("Saving Experience FD memory")
                        file_name=directory+getAgentName()+"_FD_expBufferInit.hdf5"
                        if ("keep_seperate_fd_exp_buffer" in settings
                            and (settings["keep_seperate_fd_exp_buffer"] == True)):
                            masterAgent.getFDExperience().saveToFile(file_name)
                t1 = time.time()
                sim_time_ = datetime.timedelta(seconds=(t1-t0))
                if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
                    print ("exp saving time complete in " + str(sim_time_) + " seconds")
                """
            # mean_reward = std_reward = mean_bellman_error = std_bellman_error = mean_discount_error = std_discount_error = None
            # if ( trainData["round"] % 10 ) == 0 :
            
            trainData["round"] = trainData["round"] + 1
                
            gc.collect()    
            # print (h.heap())
    except Exception as e:
        ### Nothing to really do, but can still send email of progress
        error = traceback.format_exc() 
        logging.error(error)
        print ("Caught error: ", error)
        trainData['error'] = error 
        # bellman_error = np.fabs(np.array(bellman_error))
        # print ("Mean Bellman error: " + str(np.mean(np.fabs(bellman_error))))
        # print ("STD Bellman error: " + str(np.std(np.fabs(bellman_error))))
        
        # discounted_values = np.array(discounted_values)
        # values = np.array(values)
        
        # print ("Discounted reward difference: " + str(discounted_values - values))
        # print ("Discounted reward difference Avg: " +  str(np.mean(np.fabs(discounted_values - values))))
        # print ("Discounted reward difference STD: " +  str(np.std(np.fabs(discounted_values - values))))
        # reward_over_epoc = np.array(reward_over_epoc)
    print ("Terminating Workers")
    if (settings['on_policy'] == True):
        for m_q in sim_work_queues:
            ## block on full queue
            m_q.put(None, timeout=timeout_)
        if ( 'override_sim_env_id' in settings and (settings['override_sim_env_id'] != False)):
            for m_q in eval_sim_work_queues:
                ## block on full queue
                m_q.put(None, timeout=timeout_)
        for sw in sim_workers: # Should update these more often
            sw.join()
        if ( 'override_sim_env_id' in settings and (settings['override_sim_env_id'] != False)):
            for sw in eval_sim_workers: # Should update these more often
                sw.join() 
    else:
        for sw in sim_workers: 
            input_anchor_queue.put(None, timeout=timeout_)
        if ( 'override_sim_env_id' in settings and (settings['override_sim_env_id'] != False)):
            for sw in eval_sim_workers: 
                input_anchor_queue_eval.put(None, timeout=timeout_)
        print ("Joining Workers"        )
        for sw in sim_workers: # Should update these more often
            sw.join()
        if ( 'override_sim_env_id' in settings and (settings['override_sim_env_id'] != False)):
            for sw in eval_sim_workers: # Should update these more often
                sw.join() 
    
    # input_anchor_queue.close()            
    # input_anchor_queue_eval.close()
    
    if (not settings['on_policy']):    
        print ("Terminating learners"        )
        if ( output_experience_queue != None):
            for lw in learning_workers: # Should update these more often
                output_experience_queue.put(None, timeout=timeout_)
                output_experience_queue.put(None, timeout=timeout_)
            output_experience_queue.close()
        print ("Joining learners"        )  
        """
        for m_q in sim_work_queues:  
            print(masterAgent_message_queue.get(False))
            # print(masterAgent_message_queue.get(False))
        while (not masterAgent_message_queue.empty()):
            ## Don't block
            try:
                data = masterAgent_message_queue.get(False)
            except Exception as inst:
                print ("training: In model parameter message queue empty: ", masterAgent_message_queue.qsize())
        """
        for i in range(len(learning_workers)): # Should update these more often
            print ("Joining learning worker ", i , " of ", len(learning_workers))
            learning_workers[i].join()
    
    for i in range(len(sim_work_queues)):
        print ("sim_work_queues size: ", sim_work_queues[i].qsize())
        while (not sim_work_queues[i].empty()): ### Empty the queue
            ## Don't block
            try:
                data_ = sim_work_queues[i].get(False)
            except Exception as inst:
                # print ("SimWorker model parameter message queue empty.")
                pass
        # sim_work_queues[i].cancel_join_thread()
        print ("sim_work_queues size: ", sim_work_queues[i].qsize())
        
        
    for i in range(len(eval_sim_work_queues)):
        print ("eval_sim_work_queues size: ", eval_sim_work_queues[i].qsize())
        while (not eval_sim_work_queues[i].empty()): ### Empty the queue
            ## Don't block
            try:
                data_ = eval_sim_work_queues[i].get(False)
            except Exception as inst:
                # print ("SimWorker model parameter message queue empty.")
                pass
        print ("eval_sim_work_queues size: ", eval_sim_work_queues[i].qsize())
    
    
    print ("Finish sim")
    if (int(settings["num_available_threads"]) == -1): # This is okay if there is one thread only...
        exp_val.finish()
    
    print ("Save last versions of files.")
    masterAgent.saveTo(directory)
    masterAgent.finish()
    
    f = open(directory+"trainingData_" + str(settings['agent_name']) + ".json", "w")
    from util.utils import NumpyEncoder 
    json.dump(trainData, f, sort_keys=True, indent=4, cls=NumpyEncoder)
    f.close()
    
    """except: # catch *all* exceptions
    e = sys.exc_info()[0]
    print ("Error: " + str(e))
    print ("State " + str(state_) + " action " + str(pa) + " newState " + str(resultState) + " Reward: " + str(reward))
    
    """ 
    
    print("Delete any plots being used")
    
    if settings['visualize_learning']:    
        rlv.finish()
    if (settings['train_forward_dynamics']):
        if settings['visualize_learning']:
            nlv.finish()
        if (settings['train_reward_predictor']):
            if settings['visualize_learning']:
                rewardlv.finish()
             
    if (settings['debug_critic']):
        if (settings['visualize_learning']):
            critic_loss_viz.finish()
            critic_regularization_viz.finish()
    if (settings['debug_actor']):
        if (settings['visualize_learning']):
            actor_loss_viz.finish()
            actor_regularization_viz.finish()
    
    if ("learning_backend" in settings and
        (settings["learning_backend"] == "tensorflow")):
        import keras        
        sess = keras.backend.get_session()
        keras.backend.clear_session()
        sess.close()
        del sess
    
    if ((("email_log_data_periodically" in settings)
            and (settings["email_log_data_periodically"] == True))
        or 
         ("save_video_to_file" in settings)):
        loggingWorkerQueue.put("perform_logging")
        loggingWorkerQueue.put(False)
        loggingWorker.join()
    # print ("sys.modules: ", json.dumps(str(sys.modules), indent=2))
    ### This will find ALL your memory deallocation issues in C++...
    ### And errors in terinating processes properly...
    gc.collect()
    
    """
    if ("save_video_to_file" in settings):
        from ModelEvaluation import modelEvaluation
        ### Render a video of the policies current performance
        modelEvaluation("", settings)
    """
    ### Return the collected training data
    if ("return_model" in settings 
        and (settings['return_model'] == True)):
        trainData['masterAgent'] = masterAgent
    print ("Done sim")
    return trainData
        
import inspect
def print_full_stack(tb=None):
    """
    Only good way to print stack trace yourself.
    http://blog.dscpl.com.au/2015/03/generating-full-stack-traces-for.html
    """
    if tb is None:
        tb = sys.exc_info()[2]
    out = ""
    print ('Traceback (most recent call last):')
    if (not (tb == None)):
        for item in reversed(inspect.getouterframes(tb.tb_frame)[1:]):
            out = out + ' File "{1}", line {2}, in {3}\n'.format(*item)
            for line in item[4]:
                out = out + ' ' + line.lstrip()
            for item in inspect.getinnerframes(tb):
                out = out + ' File "{1}", line {2}, in {3}\n'.format(*item)
            for line in item[4]:
                out = out + ' ' + line.lstrip()
    print (out)
    return out
            
import signal
import sys
def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        # global sim_processes
        # sim_processes = sim_workers
        # global learning_processes
        # learning_processes = learning_workers
        print("sim processes: ", sim_processes)
        print("learning_processes: ", learning_processes)
        
        # cancel_join_thread()
        ## cancel all the queues
        _input_anchor_queue.cancel_join_thread()
        _output_experience_queue.cancel_join_thread()
        _eval_episode_data_queue.cancel_join_thread()
        for sim_queue in _sim_work_queues:
            sim_queue.cancel_join_thread()
        
        
        for proc in sim_processes:
            if (not (proc == None)):
                print ("Killing process: ", proc)
                print ("process id: ", proc.pid())
                os.kill(proc.pid(), signal.SIGINT)
        for proc in learning_processes:
            if (not (proc == None)):
                print ("Killing process: ", proc.pid())
                os.kill(proc.pid(), signal.SIGINT)
            
        print_full_stack()
        sys.exit(0)
# signal.signal(signal.SIGINT, signal_handler)


if (__name__ == "__main__"):
    
    """
        python trainModel.py <sim_settings_file>
        Example:
        python trainModel.py settings/navGame/PPO_5D.json 
    """
    import time
    import datetime
    from util.simOptions import getOptions
    
    options = getOptions(sys.argv)
    options = vars(options)
    # print("options: ", options)
    # print("options['configFile']: ", options['configFile'])
        
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

        # print ("Settings: " + str(json.dumps(settings, indent=4)))
    metaSettings = None
    if ( 'metaConfigFile' in settings and (settings['metaConfigFile'] is not None)):
        ### Import meta settings
        file = open(settings['metaConfigFile'])
        metaSettings = json.load(file)
        file.close()
        
        
    t0 = time.time()
    simData = []
    if ( (metaSettings is None)
        or ((metaSettings is not None) and (not metaSettings['testing'])) ):
        # try:
            simData = trainModelParallel((sys.argv[1], settings))
        # except:
            ### Nothing to really do, but can still send email of progress
            # print("Printing stack trace:")
            # print_full_stack()
    t1 = time.time()
    sim_time_ = datetime.timedelta(seconds=(t1-t0))
    print ("Model training complete in " + str(sim_time_) + " seconds")
    print ("simData", simData)
    
    ### If a metaConfig is supplied email out the results
    if ( (metaSettings is not None) ):
        settings["email_log_data_periodically"] = True
        settings.pop('save_video_to_file', None)
        collectEmailData(settings, metaSettings, sim_time_, simData)

    print("All Done.")
    sys.exit(0)