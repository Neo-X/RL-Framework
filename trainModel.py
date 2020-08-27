
import copy
import cProfile, pstats, io
import datetime
import gc
import inspect
import json
import multiprocessing
import logging
import os
import pdb
import random
import signal
import string
import sys
import tarfile
import time
import traceback

from model.LearningMultiAgent import LearningMultiAgent
from simulation.LoggingWorker import LoggingWorker
from util.SimulationUtil import createActor, getAgentName, createSampler, createForwardDynamicsModel
from util.simOptions import getOptions
from util.SimulationUtil import setupEnvironmentVariable, setupLearningBackend
from util.SimulationUtil import validateSettings, getFDStateSize
from util.SimulationUtil import getDataDirectory, getAgentNameString
from util.SimulationUtil import addDataToTarBall, addPicturesToTarBall
from util.SimulationUtil import getDataDirectory, getBaseDataDirectory, getRootDataDirectory, getAgentName
from ModelEvaluation import modelEvaluation

sys.setrecursionlimit(50000)
sys.path.append("../")

# Global variables to manage multiprocessing / multithreading.
sim_processes = []
learning_processes = []
_input_anchor_queue = None
_output_experience_queue = None
_eval_episode_data_queue = None
_sim_work_queues = []

log = logging.getLogger(__file__)

def random_string(N_chars):
    randstate = random.getstate()
    random.seed(int.from_bytes(os.urandom(2), 'big'))
    rstring = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(N_chars))
    random.setstate(randstate)
    return rstring

def collectEmailData(settings, metaSettings, sim_time_=0, simData={}, exp=None):
    if (("email_log_data_periodically" in settings)
        and (settings["email_log_data_periodically"] == True)
        and (not (("experiment_logging" in settings)
            and ("use_comet" in settings["experiment_logging"])
            and (settings["experiment_logging"]["use_comet"] == True)))):
        # print ('settings["experiment_logging"]', settings["experiment_logging"])
        ### Create a tar file of all the sim data
        root_data_dir = getDataDirectory(settings)+"/"
        tarFileName = (root_data_dir + '_sim_data.tar.gz_') ## gmail doesn't like compressed files....so change the file name ending..
        dataTar = tarfile.open(tarFileName, mode='w:gz')
        addDataToTarBall(dataTar, settings)
            
        print("root_data_dir: ", root_data_dir)
        pictureFileName=None
        try:
            ## Add pictures to tar file
            _data_dir = getDataDirectory(settings)
            addPicturesToTarBall(dataTar, settings, data_folder=_data_dir)
            pictureFileName= [ root_data_dir + getAgentName() + ".png",
                              root_data_dir + "trainingGraphNN" + ".png",
                              root_data_dir + "rewardTrainingGraph" + ".png"]
        except Exception as e:
            # dataTar.close()
            print("Error plotting data there my not be a DISPLAY available.")
            print("Error: ", e)
        dataTar.close()
        
        
        ## Send an email so I know this training has completed
        contents_ = json.dumps(metaSettings, indent=4, sort_keys=True)
        sub = "Simulation complete: " + str(sim_time_)
        simData = {}
        if ('error' in simData):
            contents_ = contents_ + "\n" + simData['error']
            sub = "ERROR*****     " + "Simulation terminated: " + str(sim_time_)
#         try:
#             sendEmail(subject=sub, contents=contents_, hyperSettings=metaSettings, simSettings=settings['configFile'], dataFile=tarFileName,
#                       pictureFile=pictureFileName)
#         except Exception as e:
#             print("Error sending email this computer might not be authorized to use the email account.")
#             print("Error: ", e)
#             print (traceback.format_exc()) 
    
    if ("save_video_to_file" in settings):
        ### Render a video of the policies current performance
        print ("exp for video: ", exp)
        modelEvaluation("", settings=settings, exp=exp)
        
        ### Backup data
    if (("backup_exp_data" in settings)
        and (settings["backup_exp_data"] == True)):
        import subprocess
        try:
            print("Backing up learning data.")
            subprocess.call("./backup_data.sh", shell=True)
        except Exception as e:
            print("Error Backing up data using rsync.")
            print("Error: ", e)
            print (traceback.format_exc())

def pretrainCritic(masterAgent, states, actions, resultStates, rewards_, falls_, G_ts_, exp_actions, advantage_,
                    datas=None, sampler=None):
    settings__ = copy.deepcopy(masterAgent.getSettings())
    settings__2 = copy.deepcopy(masterAgent.getSettings())
    settings__["train_actor"] = False
    settings__["clear_exp_mem_on_poli"] = True
    ### Protects for the case when they are singular and don't want to skip training the critic and train the policy
    settings__["ppo_use_seperate_nets"] = True
    """
    ### This will not change the settings of the simWorkers that will be expecting "fast"
    if (settings__["on_policy"] == "fast"):
        settings__["on_policy"] = True
    """
    masterAgent.setSettings(settings__)
    masterAgent.getPolicy().setSettings(settings__)
    # masterAgent.getForwardDynamics().setSettings(settings)
    for i in range(int(settings__["pretrain_critic"])):
        print ("pretraining critic round: ", i)
        masterAgent.train(_states=states, _actions=actions, _rewards=rewards_, _result_states=resultStates,
                                       _falls=falls_, _advantage=advantage_, _exp_actions=exp_actions, 
                                       _G_t=G_ts_, datas=datas, p=1.0, trainInfo={"epoch": i})
        sampler.sendKeepAlive(masterAgent)
    ### back to normal settings
    masterAgent.setSettings(settings__2)
    masterAgent.getPolicy().setSettings(settings__2)
    print ("Done pretraining critic")
    
def pretrainFD(masterAgent, states, actions, resultStates, rewards_, falls_, G_ts_, exp_actions, advantage_,
                    datas=None, sampler=None):
    
    ### comet logging does not like being pickeled
    set = masterAgent.getSettings()
    if ("logger_instance" in set):
        clog = set["logger_instance"]
        set["logger_instance"] = None
        
    settings__ = copy.deepcopy(set)
    settings__2 = copy.deepcopy(set)
    settings__["train_actor"] = False
    settings__["train_critic"] = False
    settings__["refresh_rewards"] = False
    settings__["clear_exp_mem_on_poli"] = True
    ### Protects for the case when they are singular and don't want to skip training the critic and train the policy
    settings__["ppo_use_seperate_nets"] = True
    masterAgent.setSettings(settings__, forceCopy=True)
    masterAgent.getPolicy().setSettings(settings__)
    # masterAgent.getForwardDynamics().setSettings(settings)
    for i in range(int(settings__["pretrain_fd"])):
        print ("pretraining fd round: ", i)
        masterAgent.train(_states=states, _actions=actions, _rewards=rewards_, _result_states=resultStates,
                                       _falls=falls_, _advantage=advantage_, _exp_actions=exp_actions, 
                                       _G_t=G_ts_, datas=datas, p=1.0, trainInfo={"epoch": i})
        sampler.sendKeepAlive(masterAgent)

    ### back to normal settings
    if ("logger_instance" in set):
        settings__2["logger_instance"] = clog
        set["logger_instance"] = clog
    masterAgent.setSettings(settings__2,  forceCopy="all")
    # masterAgent.getPolicy().setSettings(settings__2)
    print ("Done pretraining fd")


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
def _initialize_train_data():
    trainData = {}
    trainData["round"]=0
    return trainData

def trainModelParallel_(input):
    return trainModelParallel(input[0], input[1])

def trainModelParallel(settingsFileName, settings):
    # TODO this function is way too long
    from util.SimulationUtil import getDataDirectory, getAgentNameString, getAgentName, getAgentNameString
    from datetime import datetime
#     settings = inputData[1]
    settings["round"] = 0
#     settingsFileName = inputData[0]  
    print (settingsFileName)
    print (settings)
    settings['doodad_config'] = settingsFileName
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
    settings['data_folder'] = settings['data_folder'] + '/' + timestamp
    settingsFileName=settings['settingsFileName']

    # Tag_FullObserve_SLAC_mini.json: True (not in settings)
    if ("perform_multiagent_training" not in settings):
        settings["perform_multiagent_training"] = 1
        settings["state_bounds"] = [settings["state_bounds"]]
        settings["action_bounds"] = [settings["action_bounds"]]
        settings["reward_bounds"] = [settings["reward_bounds"]]
        settings["exploration_rate"] = [settings["exploration_rate"]]
        settings["experience_length"] = [settings["experience_length"]]
        settings["critic_network_layer_sizes"] = [settings["critic_network_layer_sizes"]]
        settings["policy_network_layer_sizes"] = [settings["policy_network_layer_sizes"]]
        # Tag_FullObserve_SLAC_mini.json: True
        if (settings["train_forward_dynamics"]):
            settings["fd_network_layer_sizes"] = [settings["fd_network_layer_sizes"]]
            settings["reward_network_layer_sizes"] = [settings["reward_network_layer_sizes"]]
        
    print ("Number of agents: ", settings["perform_multiagent_training"])
    if (not validateSettings(settings)):
        return False
    
        ### Try and load previous data
    if ( ((settings["load_saved_model"] == True)
          or (settings["load_saved_model"] == 'last')) and
        (settings["save_experience_memory"] == "continual")):
        from util.SimulationUtil import getDataDirectory, getAgentNameString
        ### load training data
        directory = getDataDirectory(settings)
        file_name_data = directory+"trainingData_" + str(getAgentNameString(settings['agent_name'])) + ".json"
        file_name_settings=directory+os.path.basename(settingsFileName)
        print ("loading previous training data: ", file_name_data)
        if os.path.exists(file_name_data):
            fp = open(file_name_data, 'r')
            # print ("Train data: ", trainData)
            trainData = json.load(fp)
            fp.close()
            
            fp = open(file_name_settings, 'r')
            # print ("Train data: ", trainData)
            settings = json.load(fp)
            fp.close()
            
            
            if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
                print ("Loading training data")
                print ("Round: ", trainData["round"])
                # sys.exit()
        else:
            print(" Actually this is the first run..")
            settings["load_saved_model"] = False

    # Creates and stores the comet logger.
    exp_logger = setupEnvironmentVariable(settings)
    settings["logger_instance"] = None
    settings['sample_single_trajectories'] = True    
    try:
        trainData = _initialize_train_data()
        rounds = settings["rounds"]
        epochs = settings["epochs"]
        epsilon = settings["epsilon"]
        discount_factor=settings["discount_factor"]
        reward_bounds=settings["reward_bounds"]

        # Tag_FullObserve_SLAC_mini.json: True, 64
        if ( 'value_function_batch_size' in settings): batch_size=settings["value_function_batch_size"]
        else: batch_size=settings["batch_size"]
        train_on_validation_set=settings["train_on_validation_set"]
        
        
        from simulation.Sampler import Sampler 
        sampler = Sampler(settings, log)
        
        ### Keep forward models on the CPU
        create_videos = settings.get("email_log_data_periodically", False) or "save_video_to_file" in settings
        video_creation_period_supplied = settings.get("checkpoint_vid_rounds", None) is not None
        create_logging_worker = create_videos and video_creation_period_supplied
        
        if create_logging_worker:
            loggingWorkerQueue = multiprocessing.Queue(1)
            loggingWorker = LoggingWorker(settings, collectEmailData, loggingWorkerQueue)
            loggingWorker.start()
            if settings.get("test_movie_rendering", False):
                return
        
        
        values = []
        discounted_values = []
        bellman_error = []
        reward_over_epoc = []
        dynamicsLosses = []
        dynamicsRewardLosses = []
        
        ## Theano and numpy needs to be imported after the flags are set.
        ## TODO explain why this is true. Modules should be imported at the top of the file.
        import numpy as np
        import math
        import random
        import time
        import datetime
        np.random.seed(int(settings['random_seed']))
        setupLearningBackend(settings)

        # TODO all of these imports should happen at the beginning of the file.
        from simulation.collectExperience import collectExperience
        from model.ModelUtil import validBounds, fixBounds, anneal_value, getLearningData
        # from model.LearningMultiAgent import LearningMultiAgent, LearningWorker
        # from model.LearningAgent import LearningMultiAgent, LearningWorker
        from util.SimulationUtil import createEnvironment, logExperimentData, saveData
        from util.SimulationUtil import createRLAgent, createNewFDModel, processBounds
        from util.SimulationUtil import createActor, getAgentName, updateSettings, getAgentNameString
        from util.SimulationUtil import getDataDirectory, createForwardDynamicsModel, createSampler
        from util.ExperienceMemory import ExperienceMemory
        
        model_type= settings["model_type"]
        directory= getDataDirectory(settings)
        
        if not os.path.exists(directory):
            os.makedirs(directory)

        log_fn = "{}/trainModel_log_{}.log".format(directory, random_string(8))
        log_level = getattr(logging, settings.get("log_level", "info").upper(), logging.INFO)
        handlers = [logging.FileHandler(log_fn),
                    logging.StreamHandler()]
        _ = [__.setLevel(log_level) for __ in handlers]

        # You could change the logging level by setting the level= argument here, e.g. via the settings file
        logging.basicConfig(level=log_level,
                            format="[%(filename)s:%(lineno)s:%(thread)d:%(process)d - %(funcName)10s():%(levelname)s] %(message)s",
                            handlers=handlers)
            
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
            
        saveData(settings, settingsFileName, exp_logger)
            
        state_bounds = settings['state_bounds']
        discrete_actions = settings['discrete_actions']
        log.debug("Sim config file name: " + str(settings["sim_config_file"]))
        action_space_continuous=settings['action_space_continuous']
        
        if action_space_continuous: action_bounds = settings["action_bounds"]
        else: action_bounds = [None]
        
        ### Using a wrapper for the type of actor now
        actor = createActor(settings['environment_type'], settings, None)
        exp_val = None
        for i in range(len(state_bounds)):
            # print ("state_bounds[i]: ", state_bounds[i])
            if (action_space_continuous 
                and (action_bounds[i] != "ask_env")
                and (isinstance(action_bounds[i], list))
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
        
        exp_val = createEnvironment(settings["sim_config_file"], settings['environment_type'], settings, render=settings['shouldRender'], index=0)
        exp_val.setActor(actor)
        exp_val.getActor().init()
        exp_val.init()
        
        ### This should really be moved inside createRLAgent
        # pdb.set_trace()
        (state_bounds, action_bounds, settings) = processBounds(state_bounds, action_bounds, settings, exp_val)
        
        ### This is for a single-threaded Synchronous sim only.
        if (int(settings["num_available_threads"]) == -1): # This is okay if there is one thread only...
            sim_workers[0].setEnvironment(exp_val)
            sim_workers[0].start()
            if ( 'override_sim_env_id' in settings and (settings['override_sim_env_id'] != False)):
                eval_sim_workers[0].setEnvironment(exp_val)
                eval_sim_workers[0].start()
        
        model = createRLAgent(settings['agent_name'], state_bounds, discrete_actions, reward_bounds, settings, print_info=True)
        # sys.exit()
        forwardDynamicsModel = None
        if (settings['train_forward_dynamics']):
            forwardDynamicsModel = createNewFDModel(settings, exp_val, model)
        
        if ("train_reward_distance_metric" in settings and
            (settings['train_reward_distance_metric'] == True )):
            print ("Creating reward distance model")
            settings_ = copy.deepcopy(settings)
            settings_ = updateSettings(settings_, settings_["reward_metric_settings"])
            rewardModel = createNewFDModel(settings_, exp_val, model)
            rewardModel.setActor(actor)
            rewardModel.init(len(state_bounds[0]), len(action_bounds[0]), state_bounds, action_bounds, actor, None, settings)
        
#         exp_val.finish()
        (agent, learning_workers) = createLearningAgent(settings, None, print_info=True)
        masterAgent = agent
        
        
        masterAgent.setPolicy(model)
        if (settings['train_forward_dynamics']):
            masterAgent.setForwardDynamics(forwardDynamicsModel)
            
        if ("train_reward_distance_metric" in settings and
            (settings['train_reward_distance_metric'] == True )):
            masterAgent.setRewardModel(rewardModel)
            
        if ("policy_connections" in settings):
            for c in range(len(settings["policy_connections"])): 
                print ("Sending policy ", model[settings["policy_connections"][c][0]],
                                                " to policy ",  model[settings["policy_connections"][c][1]])
                masterAgent.getAgents()[settings["policy_connections"][c][1]].getPolicy().setFrontPolicy(
                    masterAgent.getAgents()[settings["policy_connections"][c][0]])
        state_bounds = masterAgent.getStateBounds()
        action_bounds = masterAgent.getActionBounds()
        reward_bounds = masterAgent.getRewardBounds()
        settings['state_bounds'] = masterAgent.getStateBounds()
        settings['action_bounds'] = masterAgent.getActionBounds()
        settings['reward_bounds'] = masterAgent.getRewardBounds()
        
        tmp_p=1.0
        
        sampler.updateParameters(masterAgent, p=tmp_p)
        
            # We don't have threads.
        # TODO what does this function do.
        experience, state_bounds, reward_bounds, action_bounds, \
        (states, actions, resultStates, rewards_, falls_, G_ts_, exp_actions, advantage_, datas), \
        experiencefd = collectExperience(actor,
                                        masterAgent,
                                        settings,
                                        sampler=sampler)
            
        masterAgent.setExperience(experience)
        fd_epxerience_length = settings['experience_length']
        
        if ("fd_experience_length" in settings):
            fd_epxerience_length = settings["fd_experience_length"]
        if ( settings['train_forward_dynamics'] and 
             ('keep_seperate_fd_exp_buffer' in settings and (settings['keep_seperate_fd_exp_buffer']))
             ):
            state_bounds__ = getFDStateSize(settings)
            ### Might be some memory expenditure here with a double copy
            masterAgent.setFDExperience(copy.deepcopy(experiencefd))
            # masterAgent.setFDStateBounds(copy.deepcopy(state_bounds__))
            # masterAgent.setFDActionBounds(copy.deepcopy(action_bounds))
            # masterAgent.setFDRewardBounds(copy.deepcopy(reward_bounds))
        if (settings["load_saved_model"] and
            (settings["save_experience_memory"] == "continual")):
            ### load exp mem
            if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
                print ("Loading Experience memory")
            file_name=directory+getAgentName()
            masterAgent.loadExperience(file_name)
            if (settings['train_forward_dynamics']):
                if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
                    print ("Loading Experience FD memory")
                masterAgent.loadFDExperience(file_name)
            
        if (action_space_continuous
            and not validBounds(action_bounds)):
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
        # print("Master agent state bounds: ",  repr(masterAgent.getStateBounds()))
            
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
        from util.Plotting import Plotter
        plotter = Plotter(settings)
                        
        settings["logger_instance"] = exp_logger
        settings["round"] = int(trainData["round"])
        masterAgent.setSettings(settings, forceCopy="all")

        if ("pretrain_critic" in settings and (settings["pretrain_critic"] > 0)
            and (trainData["round"] == 0)):
            # Pretrain the critic
            pretrainCritic(masterAgent, states, actions, resultStates, rewards_, 
                           falls_, G_ts_, exp_actions, advantage_, datas, sampler=sampler)
            
        if ("pretrain_fd" in settings and (settings["pretrain_fd"] > 0)
            and (trainData["round"] == 0)):
            # Pretrain forward dynamics
            pretrainFD(masterAgent=masterAgent, states=states, actions=actions, resultStates=resultStates, rewards_=rewards_, 
                           falls_=falls_, G_ts_=G_ts_, exp_actions=exp_actions, advantage_=advantage_,
                           datas=datas, sampler=sampler)
        
        log.info("Starting first round: " + str(trainData["round"]))
        if (settings['on_policy']):
            sim_epochs_ = epochs
            # epochs = 1
        # for round_ in range(0,rounds):
        while (trainData["round"] <= rounds):
            trainData["round"] = int(trainData["round"])
            settings["round"] = int(trainData["round"])
            masterAgent.setSettings(settings)
            if ( 'annealing_schedule' in settings and (settings['annealing_schedule'] != False)):
                p = anneal_value(float(trainData["round"]/rounds), settings_=settings)
            else:
                p = ((settings['initial_temperature']/math.log(trainData["round"]+2))) 
            p = max(settings['min_epsilon'], min(1.0, p))*settings['epsilon'] # Keeps it between 1.0 and 0.2
            if ( settings['load_saved_model'] == True):
                p = settings['min_epsilon']
            settings["p"] = p
                
            # pr = cProfile.Profile()
            for epoch in range(epochs):
                if (settings['on_policy']):
                    
                    # if ( settings['num_available_threads'] > 0 ):  
                    if ("skip_rollouts" in settings and 
                        (settings["skip_rollouts"] == True)):
                        out = (([],[],[],[],[],[],[],[], []), [], [], [])
                        sampler.sendKeepAlive(masterAgent)
                    else:
                        out = sampler.obtainSamples( masterAgent=masterAgent,
                                                     rollouts=settings['num_on_policy_rollouts']
                                                   ,p=p)
                    
                    (tuples, discounted_sum, q_value, evalData) = out
                    (__states, __actions, __result_states, __rewards, __falls, __G_ts, advantage__, exp_actions__, datas__) = tuples
                    if ( ('anneal_on_policy' in settings) and settings['anneal_on_policy']):  
                        p_tmp_ = p 
                    else:
                        p_tmp_ = 1.0
                    
                    for i in range(1):
                        masterAgent.train(_states=__states, _actions=__actions, _rewards=__rewards, _result_states=__result_states,
                                           _falls=__falls, _advantage=advantage__, _exp_actions=exp_actions__, _G_t=__G_ts, p=p_tmp_, 
                                           datas=datas__, trainInfo={"epoch": epoch, "round": settings["round"]})
                    masterAgent.reset()
                    
                    
                    if ("skip_rollouts" in settings and 
                        (settings["skip_rollouts"] == True)):
                        pass
                    else:
                        sampler.updateParameters(masterAgent, p=tmp_p)
                    
                else:
                    ### Old off-policy method not really supported now.
                    episodeData = {}
                    episodeData['data'] = epoch
                    episodeData['type'] = 'sim'
                    input_anchor_queue.put(episodeData, timeout=timeout_)       
                    
            if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
                if (settings['train_forward_dynamics']):
                    print ("Round: " + str(trainData["round"]) + " of ", rounds,  ", Epoch: " + str(epoch) + " p: " + str(p))
                else:
                    print ("Round: " + str(trainData["round"]) + " of ", rounds,  ", Epoch: " + str(epoch) + " p: " + str(p))
            if (trainData["round"] % settings['plotting_update_freq_num_rounds']) == 0:
                
                plotter.updatePlots(masterAgent, trainData, sampler, out, p, settings)
                
            ## This will let me know which part of learning is going slower training updates or simulation
            if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
                sampler.info()
                
            if create_logging_worker and trainData["round"] % settings["checkpoint_vid_rounds"] == 0:
                loggingWorkerQueue.put(('checkpoint_vid_rounds', trainData["round"]))

            trainData["round"] += 1
                
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
    
    # input_anchor_queue.close()            
    # input_anchor_queue_eval.close()
    
    
    print ("Save last versions of files.")
    masterAgent.saveTo(directory)
    masterAgent.finish()
    
    f = open(directory+"trainingData_" + str(getAgentNameString(settings['agent_name'])) + ".json", "w")
    from util.utils import NumpyEncoder 
    json.dump(trainData, f, sort_keys=True, indent=4, cls=NumpyEncoder)
    f.close()
    
    """except: # catch *all* exceptions
    e = sys.exc_info()[0]
    print ("Error: " + str(e))
    print ("State " + str(state_) + " action " + str(pa) + " newState " + str(resultState) + " Reward: " + str(reward))
    
    """ 
    
    if ("learning_backend" in settings and
        (settings["learning_backend"] == "tensorflow")):
        import keras        
        sess = keras.backend.get_session()
        keras.backend.clear_session()
        sess.close()
        del sess
    
#     if ((("email_log_data_periodically" in settings)
#             and (settings["email_log_data_periodically"] == True))
#         or 
#          ("save_video_to_file" in settings)):
#         loggingWorkerQueue.put("perform_logging")
#         loggingWorkerQueue.put(False)
#         loggingWorker.join()
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

def main():
    
    """
        python trainModel.py <sim_settings_file>
        Example:
        python trainModel.py settings/navGame/PPO_5D.json 
    """
    
    options = getOptions(sys.argv)
    options = vars(options)
    file = open(options['configFile'])
    settings = json.load(file)
    file.close()
    
    for option in options:
        if ( not (options[option] is None) ):
            log.info("Updating option: {}={} ".format(option, options[option]))
            settings[option] = options[option]
            try:
                settings[option] = json.loads(settings[option])
            except Exception as e:
                pass # dataTar.close()
            if ( options[option] == 'true'):
                settings[option] = True
            elif ( options[option] == 'false'):
                settings[option] = False
    metaSettings = None

    # Tag_FullObserve_SLAC_mini.json: false
    if ( 'metaConfigFile' in settings and (settings['metaConfigFile'] is not None)):
        ### Import meta settings
        file = open(settings['metaConfigFile'])
        metaSettings = json.load(file)
        file.close()

    # Tag_FullObserve_SLAC_mini.json: false
    if 'checkpoint_vid+rounds' in settings:
        # Tag_FullObserve_SLAC_mini.json: false
        if 'save_video_to_file' in settings:
            log.error('\nerror: checkpoint_vid_rounds set but save_video_to_file is unset. Exiting.')        
            sys.exit()
        # Tag_FullObserve_SLAC_mini.json: false            
        elif 'saving_update_freq_num_rounds' not in settings or settings['saving_update_freq_num_rounds'] > settings['checkpoint_vid_rounds']:
            log.warning('saving_update_freq_num_rounds > checkpoint_vid_rounds. Updating saving_update_freq_num_rounds to checkpoing_vid_rounds')
            settings['saving_update_freq_num_rounds'] = settings['checkpoint_vid_rounds']
        else:
            log.warning("Unhandled else statement!")

    t0 = time.time()
    simData = []
    if ( (metaSettings is None)
        or ((metaSettings is not None) and (not metaSettings['testing'])) ):
        settings['settingsFileName'] = sys.argv[1]
        simData = trainModelParallel(sys.argv[1], settings)
    t1 = time.time()
    sim_time_ = datetime.timedelta(seconds=(t1-t0))
    print ("Model training complete in " + str(sim_time_) + " seconds")
    print ("simData", simData)
    
    ### If a metaConfig is supplied email out the results
    if ( (metaSettings is not None) ):
        settings["email_log_data_periodically"] = True
        settings.pop('save_video_to_file', None)
        settings.pop("experiment_logging", None)
        collectEmailData(settings, metaSettings, sim_time_, simData)

    print("All Done.")
    sys.exit(0)

if (__name__ == "__main__"):
    main()
