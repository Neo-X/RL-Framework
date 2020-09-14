
import copy
import logging
import sys
from builtins import isinstance
sys.setrecursionlimit(50000)
import os
import json
from pydoc import locate
import logging

import dill
import dill as pickle
import dill as cPickle

# import cProfile, pstats, io
# import memory_profiler
# import psutil
import gc
# from guppy import hpy; h=hpy()
# from memprof import memprof

log = logging.getLogger(os.path.basename(__file__))

def updateSettings(settings1, newSettings):
    """
        Replace all of the setting sin settings1 with the settings in newSettings
    """
    for key_ in newSettings.keys():
        settings1[key_] = newSettings[key_]
        
    return settings1

def getAgentNameString(agent_name):
    out = ""
    if (type(agent_name) is list):
        for a in agent_name:
            out = out + str(agent_name)
    else:
        out = agent_name
    return out
    
def getGPUBusIndex(index=0):
    import re
    print ("args: ", sys.argv)
    raw_devices = files = [f for f in os.listdir("/dev") if re.match('nvidia[0-9]+', f)]
    print ("raw_devices: ", raw_devices)
    if (index > (len(raw_devices)-1)):
        print ("Not enough GPU devices returning default (0)")
        return "0"
    raw_devices.sort() ### in place sort
    print ("sorted devices: ", raw_devices)
    print ("selected device: ", raw_devices[index])
    ### return BUS ID
    return raw_devices[index][6:]

def saveData(settings, settingsFileName, exp_logger):
    
    directory= getDataDirectory(settings)
    
    ### Put git versions in settings file before save
    from util.utils import get_git_revision_hash, get_git_revision_short_hash
#     settings['git_revision_hash'] = get_git_revision_hash()
#     settings['git_revision_short_hash'] = get_git_revision_short_hash()     
    ### copy settings file
    out_file_name=directory+os.path.basename(settingsFileName)
    print ("Saving settings file with data: ", out_file_name)
    if ("logger_instance" in settings 
        and (settings["logger_instance"] is not None)):
#         exp_logger = settings["logger_instance"]
        settings["logger_instance_key"] = exp_logger.get_key()
        settings["logger_instance"] = None
    out_file = open(out_file_name, 'w')
    out_file.write(json.dumps(settings, indent=4))
    out_file.close()
    out_file = open(directory+"params.json", 'w')
    out_file.write(json.dumps(settings, indent=4))
    out_file.close()
#     if ("logger_instance" in settings):
#         settings["logger_instance"] = exp_logger
    
    
#     ### Try and save algorithm and model files for reference
#     if "." in settings['model_type']:
#         ### convert . to / and copy file over
#         file_name = settings['model_type']
#         k = file_name.rfind(".")
#         file_name = file_name[:k]
#         file_name_read = file_name.replace(".", "/")
#         file_name_read = file_name_read + ".py"
#         print ("model file name:", file_name)
#         print ("os.path.basename(file_name): ", os.path.basename(file_name))
#         file = open(file_name_read, 'r')
#         out_file = open(directory+file_name+".py", 'w')
#         out_file.write(file.read())
#         file.close()
#         out_file.close()
#     if "." in settings['agent_name']:
#         ### convert . to / and copy file over
#         file_name = settings['agent_name']
#         k = file_name.rfind(".")
#         file_name = file_name[:k]
#         file_name_read = file_name.replace(".", "/")
#         file_name_read = file_name_read + ".py"
#         print ("model file name:", file_name)
#         print ("os.path.basename(file_name): ", os.path.basename(file_name))
#         file = open(file_name_read, 'r')
#         out_file = open(directory+file_name+".py", 'w')
#         out_file.write(file.read())
#         file.close()
#         out_file.close()
#         
#     if (settings['train_forward_dynamics']):
#         if "." in settings['forward_dynamics_model_type']:
#             ### convert . to / and copy file over
#             file_name = settings['forward_dynamics_model_type']
#             k = file_name.rfind(".")
#             file_name = file_name[:k]
#             file_name_read = file_name.replace(".", "/")
#             file_name_read = file_name_read + ".py"
#             print ("model file name:", file_name)
#             print ("os.path.basename(file_name): ", os.path.basename(file_name))
#             file = open(file_name_read, 'r')
#             out_file = open(directory+file_name+".py", 'w')
#             out_file.write(file.read())
#             file.close()
#             out_file.close()

def logExperimentData(trainData, key, value, settings):
    """This function logs scalar metrics info, possibly to comet

    :param trainData: 
    :param key: str key to log (optional, not used if type(value) == OrderDict)
    :param value: OrderedDict or value to log
    :param settings: settings object
    :returns: None
    """
    import numpy as np
    from collections import OrderedDict
    
    if ("logger_instance" in settings
        and (settings["logger_instance"] is not None)):
        logger = settings["logger_instance"] 
        logger.set_step(step=settings["round"])

        # The log_metrics function requires a dictionary mapping strs to one of Float/Integer/Boolean/String
        if (isinstance(value, OrderedDict) or isinstance(value, dict)):
            for key in value.keys():
                logger.log_metrics({key:np.mean(value[key])}, step=settings["round"])
        else:
            logger.log_metrics({key:np.mean(value)}, step=settings["round"])
        
#     if key in trainData:
#         trainData[key].append(np.mean(value))
#     elif (isinstance(value, OrderedDict) or isinstance(value, dict)):
#         pass
#     else:
#         trainData[key] = [np.mean(value)]
    if key in trainData:
        trainData[key].append(value)
    else:
        trainData[key] = [value]
        
def logExperimentImage(path, overwrite=True, image_format="mp4", settings=None):
    """This function logs scalar metrics info, possibly to comet

    :param trainData: 
    :param key: str key to log (optional, not used if type(value) == OrderDict)
    :param value: OrderedDict or value to log
    :param settings: settings object
    :returns: None
    """
    import numpy as np
    from collections import OrderedDict
    
    if ("logger_instance" in settings
        and (settings["logger_instance"] is not None)):
        logger = settings["logger_instance"] 
        logger.set_step(step=settings["round"])
        logger.log_image(path, overwrite=overwrite, image_format=image_format)
        
def logExperimentFile(path, fileName, overwrite=True, image_format="mp4", settings=None):
    """This function logs scalar metrics info, possibly to comet

    :param trainData: 
    :param key: str key to log (optional, not used if type(value) == OrderDict)
    :param value: OrderedDict or value to log
    :param settings: settings object
    :returns: None
    """
    import numpy as np
    from collections import OrderedDict
    
    if ("logger_instance" in settings
        and (settings["logger_instance"] is not None)):
        logger = settings["logger_instance"] 
        logger.set_step(step=settings["round"])
        logger.log_asset(path, file_name=fileName, overwrite=True)

def setupEnvironmentVariable(settings, eval=False):
    import os    
    os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device="+settings['training_processor_type']+",floatX="+settings['float_type']
    if ("learning_backend" in settings):
        # KERAS_BACKEND=tensorflow
        os.environ['KERAS_BACKEND'] = settings['learning_backend']
        
    ### Setup GPU resources
    if ("GPU_BUS_Index" in settings):
        print ("GPU_BUS_index: ", settings["GPU_BUS_Index"])
        # sys.exit()
        if (("force_sim_net_to_cpu" in settings
                and (settings["force_sim_net_to_cpu"] == True))
            and
            ("simulation_model" in settings
                and (settings["simulation_model"] == True))
            ):
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = settings["GPU_BUS_Index"]
            # os.environ["CUDA_VISIBLE_DEVICES"] = getGPUBusIndex(index=int(settings["GPU_BUS_Index"]))
        
    ### Reduce the use of OpenMPI in numpy
    
    if ("simulation_model" in settings 
        and (settings["simulation_model"] == True)):
        os.environ["PRETEND_CPUS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OMP_THREAD_LIMIT"] = "1"
        os.environ["LD_PRELOAD"] = "/opt/borgy/libpretend/libpretend.so"
    else:
        ### Only do this in the main thread
        pass
        
        ### log training via commet.ml
        try:
            ### This will only start if experiment logging settings are specified and a meta log file is specified
            ### This is to avoid this logging from occuring when just debugging and coding. 
            if ("experiment_logging" in settings 
                and settings["log_comet"]
                and (not eval)):
                from comet_ml import Experiment
                exp_config = settings["experiment_logging"]
                if (isinstance(exp_config, str)):
                    print ("exp_config: ", exp_config)
                    exp_config = json.loads(exp_config)

                # Add the following code anywhere in your machine learning file
                print ("Tracking training via commet.ml")
                if ("logger_instance_key" in settings):
                    from comet_ml import ExistingExperiment
                    experiment = ExistingExperiment(api_key="v063r9jHG5GDdPFvCtsJmHYZu", previous_experiment=settings["logger_instance_key"],
                                                 project_name=exp_config["project_name"], workspace="glenb")
                    print("Continuing existing experiment: ", experiment)
                else:
                    
                    experiment = Experiment(api_key="v063r9jHG5GDdPFvCtsJmHYZu",
                                            project_name=exp_config["project_name"], workspace="glenb")
                    experiment.log_parameters(settings)
                    experiment.add_tag("comet_test")
                    experiment.set_name(settings["data_folder"])
                    # experiment.log_dependency(self, "terrainRLAdapter", version)
                    experiment.set_filename(fname="cometML_test")
                return experiment
        except Exception as inst:
            print ("Not tracking training via commet.ml")
            print ("Error: ", inst)
            # sys.exit()
        
        
def setupLearningBackend(settings):
    import keras
    # from util.MakeKerasPicklable import make_keras_picklable
    # import theano
    if ("learning_backend" in settings and
        (settings["learning_backend"] == "tensorflow")):
        from keras.backend import tensorflow_backend
        import tensorflow as tf
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        ### Limit the thread pool size for each process
        if ("simulation_model" in settings 
            and (settings["simulation_model"] == True)):
            config.intra_op_parallelism_threads = 1
            config.inter_op_parallelism_threads = 1
        # config.session_inter_op_thread_pool = 1
        session = tf.Session(config=config)
        keras.backend.set_session(session)
        
    # keras.backend.set_session(tf.Session())
    keras.backend.set_floatx(settings['float_type'])
    if ("image_data_format" in settings):
        keras.backend.set_image_data_format(settings['image_data_format'])
    print ("K.floatx()", keras.backend.floatx())
    # print ("theano.config.floatX", theano.config.floatX)
    
def loadNetwork(net_file_path):
    print("Loading model: ", net_file_path)
    f = open(net_file_path, 'rb')
    model = dill.load(f)
    f.close()
    return model

def getDataDirectory(settings):
    return getBaseDataDirectory(settings)+settings["model_type"]+"/"

def processBounds(state_bounds, action_bounds, settings, sim):
    import gym
    import numpy as np
    
    if ((action_bounds == "ask_env")
        or (action_bounds == ["ask_env"])):
        print ("Getting action bounds from environment")
        if (not isinstance(sim.getEnvironment().action_space, gym.spaces.Discrete)):
            a_min = sim.getEnvironment().action_space.low
            a_max = sim.getEnvironment().action_space.high
            print (sim.getEnvironment().action_space.low)
            settings['action_bounds'] = [a_min,a_max]
        else:
            settings['action_bounds'] = [[-1] * sim.getEnvironment().action_space.n, [1] * sim.getEnvironment().action_space.n]
        if ("perform_multiagent_training" in settings):
            settings['action_bounds'] = [settings['action_bounds']]
        action_bounds = settings['state_bounds']

    print ("Getting state bounds from environment")        
    if ("perform_multiagent_training" in settings):
        state_bounds_ = []
        for i in range (settings["perform_multiagent_training"]):
            if (state_bounds[i] == "ask_env"):
                s_min = sim.getEnvironment().observation_space.low.flatten()
                s_max = sim.getEnvironment().observation_space.high.flatten()
                if ("use_centralized_critic" in settings and 
                    (settings["use_centralized_critic"] == True)):
                    bounds = [[],[]]
                    for i in range(settings["perform_multiagent_training"]):
                        bounds[0].extend(s_min)
                        bounds[1].extend(s_max)
                    
                    other_agents_action_len = len(action_bounds[0][0])* (settings["perform_multiagent_training"]-1)
                    bounds[0].extend(-np.ones((other_agents_action_len )))
                    bounds[1].extend(np.ones((other_agents_action_len)))
                    state_bounds_.append(bounds)
                else:
                    state_bounds_.append([s_min, s_max])
            else:
                ### else pull bounds from given values.
                state_bounds_.append(state_bounds[i])
        settings['state_bounds'] = state_bounds_
        state_bounds = settings['state_bounds']
        print ("settings['state_bounds']: ", np.array(settings['state_bounds']).shape)
    elif ((state_bounds == "ask_env")
        or (state_bounds == ["ask_env"])):
        s_min = sim.getEnvironment().observation_space.low.flatten()
        s_max = sim.getEnvironment().observation_space.high.flatten()
        print (sim.getEnvironment().observation_space.low)
        if ("include_suffstate_in_state" in settings
            and (settings["include_suffstate_in_state"] == True)):
            length = (len(s_min) * 3 ) + 1
            settings['state_bounds'] = [-np.ones(length),np.ones(length)]
        else:
            settings['state_bounds'] = [s_min,s_max]
        if ("perform_multiagent_training" in settings):
            if ("use_centralized_critic" in settings and 
                (settings["use_centralized_critic"] == True)):
                bounds = [[],[]]
                for i in range(settings["perform_multiagent_training"]):
                    bounds[0].extend(s_min)
                    bounds[1].extend(s_max)
                settings['state_bounds'] = bounds
                settings['state_bounds'] = [settings['state_bounds']] * settings["perform_multiagent_training"]
            else:
                settings['state_bounds'] = [settings['state_bounds']] * settings["perform_multiagent_training"]
        state_bounds = settings['state_bounds']
        print ("settings['state_bounds']: ", np.array(settings['state_bounds']).shape)
        
    return (state_bounds, action_bounds, settings)

def getBaseDataDirectory(settings):
    """
    if ("folder_instance_name" in settings):
        print ("")
        print ("****** data folder name: ", settings["data_folder"])
        print ("")
        if (settings["data_folder"][-1] == '/'):
            
            return getRootDataDirectory(settings)+"/"+settings["data_folder"][:-1]+settings["folder_instance_name"]+"/"
        else:
            return getRootDataDirectory(settings)+"/"+settings["data_folder"]+settings["folder_instance_name"]+"/"
    else:
    """
    return getRootDataDirectory(settings)+"/"+settings["data_folder"]+"/"+str(settings["random_seed"])+"/"

def getRootDataDirectory(settings):
    from launchers.config_private_rlframe import LOCAL_LOG_DIR
    try: 
        LOCAL_LOG_DIR = settings['doodad_config'].base_log_dir
    except:
            print ("not using doodad")
            pass
    print ("LOCAL_LOG_DIR : ", LOCAL_LOG_DIR )
    if (type(settings["agent_name"]) is list):
        return LOCAL_LOG_DIR + settings["environment_type"]+"/"+settings["agent_name"][0]
    return LOCAL_LOG_DIR + settings["environment_type"]+"/"+settings["agent_name"]

def getAgentName(settings=None):
    return 'agent'

def getTaskDataDirectory(settings):
    return settings["environment_type"]+"/"+settings["agent_name"]+"/"+settings["task_data_folder"]+"/"+settings["model_type"]+"/"

def addDataToTarBall(tarfile_, settings, fileName=None):
    import os
    import tarfile
    ## add all json and py files
    if ( fileName is None ):
        dir = getDataDirectory(settings)
        for filename_tmp in os.listdir(dir):
            print("Possible include file: ", os.path.splitext(filename_tmp))
            split_ = os.path.splitext(filename_tmp)
            if (split_[1] in ['.py', '.json']):
                print("Adding file: ", filename_tmp)
                tarfile_.add(dir+filename_tmp)
    """
    fileName_ = dir+"trainingData_" + str(settings['agent_name']) + ".json"
    if os.path.exists(fileName_):
        tarfile_.add(fileName_)
    else:
        print ( "File does not exists: ", fileName_)
    """
    if ( not ( fileName is None) ):
        if os.path.exists(fileName):
            tarfile_.add(fileName)
        else:
            print ( "File does not exists: ", fileName)
    
        
    # tarfile.add('/README.md')

def addPicturesToTarBall(tarfile_, settings, fileName=None, data_folder=None):
    import os
    import tarfile
    ## add all json and py files
    if ( fileName is None ):
        if (data_folder is not None):
            dir = data_folder
        else:
            dir = getRootDataDirectory(settings)+"/" + settings['data_folder'] + "/"
        # dir = getDataDirectory(settings)
        for filename_tmp in os.listdir(dir):
            print("Possible include file: ", os.path.splitext(filename_tmp))
            split_ = os.path.splitext(filename_tmp)
            if (split_[1] in ['.png', '.svg']):
                print("Adding file: ", filename_tmp)
                tarfile_.add(dir+filename_tmp)
    
        
    # tarfile.add('/README.md')

def my_import(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def getFDStateSize(settings):
    if ("use_dual_state_representations" in settings
        and (settings["use_dual_state_representations"] == True)
        and (settings["forward_dynamics_model_type"] == "SingleNet")):
            res_state_bounds__ = [[0] * settings["dense_state_size"], 
                                 [1] * settings["dense_state_size"]]
            state_bounds__ = state_bounds
            ### Usually the state and next state are the same size, not in this case...
            # experiencefd.setStateBounds(state_bounds__)
            # experiencefd.setResultStateBounds(res_state_bounds__)
    elif ("fd_num_terrain_features" in settings):
        state_size__ = settings["fd_num_terrain_features"]
        if ("fd_num_terrain_features" in settings ):
            if ("append_camera_velocity_state" in settings 
                and (settings["append_camera_velocity_state"] == True)):
                state_size__ = state_size__ + 2
            elif ("append_camera_velocity_state" in settings 
                and (settings["append_camera_velocity_state"] == "3D")):
                state_size__ = state_size__ + 3
            print ("state_size__ for fd: ", state_size__)
            # experiencefd = ExperienceMemory(state_size__, len(action_bounds[0]), fd_epxerience_length, 
            #    
            if ("fd_use_multimodal_state" in settings
                and (settings["fd_use_multimodal_state"] == True) ):
                state_size__ = state_size__ + settings["dense_state_size"]

        state_bounds__ = [[0] * (state_size__), 
                             [1] * (state_size__)]
        # experiencefd.setStateBounds(state_bounds__)
        # experiencefd.setResultStateBounds(state_bounds__)
        # sys.exit()
    else: 
        state_bounds__ = settings["state_bounds"]
          
    if ("use_dual_dense_state_representations" in settings
        and (settings["use_dual_dense_state_representations"] == True)):
        # state_bounds__ = np.array(settings['state_bounds'])
        state_bounds__ = [[0] * settings["dense_state_size"], 
                         [1] * settings["dense_state_size"]]
        # experiencefd.setStateBounds(state_bounds__)
    return state_bounds__

def validateSettings(settings):
    """
        This method is used to check for special conditions in the settings file that 
        are known to conflict. Meaning any simulation with this combination of settings
        will only produce garbage.
    """
    """
    if ("perform_multiagent_training" in settings and
        ("on_policy" in settings and
         settings["on_policy"] == "fast")):
        print ("******")
        print ("MultiAgent training does not support fast on policy simulation yet.")
        print ("******")
        return False
    """    
    
    if ("use_fall_reward_shaping" in settings and
        (settings["use_fall_reward_shaping"] == True)
        and
        ("learned_reward_smoother" in settings 
         and
         (settings["learned_reward_smoother"]) == False)):
        ### The use of the "fall" data is overloaded and conflicts here
        print ("******")
        print ("Basic fall reward shaping does not work with a non gaussian learned_reward_smoother.")
        print ("******")
        return False
    
    if ("use_single_network" in settings 
        and (settings["use_single_network"] == True)
         and (settings["agent_name"] == "algorithm.TRPO_KERAS.TRPO_KERAS")):
        ### A single network model does not work well for TRPO
        print ("******")
        print ("A single network model does not work well for TRPO.")
        print ("******")
        return False
    """    
    if ("pretrain_fd" in settings
        and (settings['pretrain_fd'] > 0)
        and ("perform_multiagent_training" in settings)):
        print ("******")
        print ("Multi agent training does not yet support pretraining the FD models.")
        print ("******")
        return False
    """
    if ("max_ent_rl" in settings
        and settings['max_ent_rl']
        and ("use_stochastic_policy" not in settings or not settings["use_stochastic_policy"])):
        print ("******")
        print ("Max Ent RL requires a stochastic policy")
        print ("******")
        return False
    
    return True

def createNetworkModel(model_type, state_bounds, action_bounds, reward_bounds, settings, print_info=False):
    if settings['action_space_continuous']:
        n_out_ = len(action_bounds[0])
    else:
        n_out_ = settings["discrete_actions"]
    if (settings['load_saved_model'] == True):
        return None
    
    elif (model_type == "DumbModel" ):
        model = DumbModel(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    else:
        from model.ModelInterface import ModelInterface
        # modelClass = my_import(path_)
        modelClass = locate(model_type)
        print ("modelClass: ", modelClass)
        if ( issubclass(modelClass, ModelInterface)): ## Double check this load will work
            model = modelClass(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                              action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings,
                              print_info=print_info)
            print("Created model: ", model)
            return model
        else:
            
            print ("Unknown network model type: ", str(model_type), " I hope you know what you are doing....")
        # sys.exit(2)
            return
    # import lasagne
    print (" network type: ", model_type, " : ", model)
    # print ("Number of Critic network parameters", lasagne.layers.count_params(model.getCriticNetwork()))
    # print ("Number of Actor network parameters", lasagne.layers.count_params(model.getActorNetwork()))
    
    # if (settings['train_forward_dynamics'] and (settings['forward_dynamics_model_type'] == 'SingleNet')):
    #     print ("Number of Forward Dynamics network parameters", lasagne.layers.count_params(model.getForwardDynamicsNetwork()))
    #     print ("Number of Reward predictor network parameters", lasagne.layers.count_params(model.getRewardNetwork()))

    return model

def createRLAgent(algorihtm_type, state_bounds, discrete_actions, reward_bounds, settings, print_info=False):
    import math
    import numpy as np
    import random
    action_bounds = settings['action_bounds']
    if ("perform_multiagent_training" in settings):
        pass
    else:
        networkModel = createNetworkModel(settings["model_type"], state_bounds, np.array(action_bounds), reward_bounds, settings, print_info=print_info)
    
    directory= getDataDirectory(settings)
    if (settings['load_saved_model']):
        if ("learning_backend" in settings and 
            ((settings['learning_backend'] == "tensorflow")
             or (settings['learning_backend'] == "theano")
            )):
            from algorithm.AlgorithmInterface import AlgorithmInterface
            settings_ = copy.deepcopy(settings)
            ### This is faster....
            settings_['load_saved_model'] = False
            # modelClass = my_import(path_)
            if ("perform_multiagent_training" in settings):
                models = []
                assert settings["perform_multiagent_training"] == len(state_bounds)
                for m in range(settings["perform_multiagent_training"]):
                    settings__ = copy.deepcopy(settings_)
                    if (type(algorihtm_type) is list):
                        algorihtm_type_ = algorihtm_type_[m]
                    else:
                        algorihtm_type_ = algorihtm_type
                    modelAlgorithm = locate(algorihtm_type_)
                    if ( issubclass(modelAlgorithm, AlgorithmInterface)): ## Double check this load will work
                        settings__["agent_id"] = m
                        settings__["critic_network_layer_sizes"] = settings["critic_network_layer_sizes"][m]
                        settings__["policy_network_layer_sizes"] = settings["policy_network_layer_sizes"][m]
                        if (type(settings["exploration_rate"]) is list):
                            settings__["exploration_rate"] = settings["exploration_rate"][m]
                        networkModel = createNetworkModel(settings__["model_type"], np.array(state_bounds[m]), np.array(action_bounds[m]), np.array(reward_bounds[m]),
                                                           settings__, print_info=print_info)
                        print ("networkModel: ", networkModel)
                        model_ = modelAlgorithm(networkModel, n_in=len(state_bounds[m][0]), n_out=len(action_bounds[m][0]), state_bounds=np.array(state_bounds[m]), 
                                  action_bounds=np.array(action_bounds[m]), reward_bound=np.array(reward_bounds[m]), settings_=settings__, print_info=print_info)
                        model_.setSettings(settings__) ### Maybe this should be the normal settings...
                        if (settings['load_saved_model'] == 'last'):
                            model_.loadFrom(directory+getAgentName()+str(m))
                        else:
                            model_.loadFrom(directory+getAgentName()+str(m)+"_Best")
                        models.append(model_)
                    else:
                        print ("Unknown learning algorithm type: " + str(algorihtm_type))
                        raise ValueError("Unknown learning algorithm type: " + str(algorihtm_type))
                print("Loaded algorithm: ", models)
                model = models
                """
                if ("policy_connections" in settings):
                    for c in range(len(settings["policy_connections"])): 
                        print ("Sending policy ", model[settings["policy_connections"][c][0]],
                                                        " to policy ",  model[settings["policy_connections"][c][1]])
                        model[settings["policy_connections"][c][1]].setFrontPolicy(
                            model[settings["policy_connections"][c][0]])
                """        
            else:
                modelAlgorithm = locate(algorihtm_type)
                if ( issubclass(modelAlgorithm, AlgorithmInterface)): ## Double check this load will work
                    networkModel = createNetworkModel(settings["model_type"], state_bounds, action_bounds, reward_bounds, settings_)
                    print ("networkModel: ", networkModel)
                    model = modelAlgorithm(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                                  action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
                    model.setSettings(settings)
                    if (settings['load_saved_model'] == 'last'):
                        model.loadFrom(directory+getAgentName())
                    else:
                        model.loadFrom(directory+getAgentName()+"_Best")
                else:
                    print ("Unknown learning algorithm type: " + str(algorihtm_type))
                    raise ValueError("Unknown learning algorithm type: " + str(algorihtm_type))
            print("Loaded algorithm: ", model)
            # return model
            # sys.exit(2)
        else:
            print ("Loading pre compiled network")
            if (settings['load_saved_model'] == 'last'):
                file_name=directory+getAgentName()+".pkl"
            else:
                file_name=directory+getAgentName()+"_Best.pkl"
            f = open(file_name, 'rb')
            model = dill.load(f)
            f.close()
            model.setSettings(settings)
            if (settings['load_saved_model'] == 'last'):
                model.loadFrom(directory+getAgentName())
            else:
                model.loadFrom(directory+getAgentName()+"_Best")
    elif (algorihtm_type == "Distillation" ):
        from algorithm.Distillation import Distillation
        model = Distillation(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)    
    else:
        from algorithm.AlgorithmInterface import AlgorithmInterface
        # modelClass = my_import(path_)
        if ("perform_multiagent_training" in settings):
            models = []
            assert settings["perform_multiagent_training"] == len(state_bounds), "settings['perform_multiagent_training]: " + str(settings["perform_multiagent_training"]) +  " ==  len(state_bounds) " + str(len(state_bounds))
            settings_ = copy.deepcopy(settings)
            for m in range(settings["perform_multiagent_training"]):
                settings__ = copy.deepcopy(settings)
                if (type(algorihtm_type) is list):
                    algorihtm_type_ = algorihtm_type[m]
                else:
                    algorihtm_type_ = algorihtm_type 
                modelAlgorithm = locate(algorihtm_type_)
                if ( issubclass(modelAlgorithm, AlgorithmInterface)): ## Double check this load will work
                    settings__["agent_id"] = m
                    settings__["policy_network_layer_sizes"] = settings["policy_network_layer_sizes"][m]
                    """
                    if ( "use_centralized_critic" in settings
                         and (settings["use_centralized_critic"] == True)):
                        state_bounds__ = copy.deepcopy(state_bounds[m])
                        for bounds_ in [x for i,x in enumerate(state_bounds) if i!=m]:
                            state_bounds__[0].extend(bounds_[0])
                            state_bounds__[1].extend(bounds_[1])
                        ### Add action bounds for other agents
                        for bounds_ in [x for i,x in enumerate(action_bounds) if i!=m]:
                            state_bounds__[0].extend(bounds_[0])
                            state_bounds__[1].extend(bounds_[1])
                        settings__["state_bounds"] = state_bounds__
                        print ("state_bounds__ shape: ", np.array(state_bounds__).shape)
                        print ("state_bounds[m] shape: ", np.array(state_bounds[m]).shape)
                        print 
                        networkModel = createNetworkModel(settings__["model_type"], state_bounds__, action_bounds[m], reward_bounds[m], settings__, print_info=print_info)
                        models.append(modelAlgorithm(networkModel, n_in=len(state_bounds__[0]), n_out=len(action_bounds[m][0]), state_bounds=state_bounds__, 
                              action_bounds=action_bounds[m], reward_bound=reward_bounds[m], settings_=settings__, print_info=print_info))
                    else:
                    """
                    settings__["critic_network_layer_sizes"] = settings["critic_network_layer_sizes"][m]
                    if (type(settings["exploration_rate"]) is list):
                        
                        settings__["exploration_rate"] = settings["exploration_rate"][m]
                    print("Creating agent: ", m)
                    networkModel = createNetworkModel(settings__["model_type"], 
                                                      state_bounds[m], 
                                                      action_bounds[m], 
                                                      reward_bounds[m], 
                                                      settings__, print_info=print_info)
                    models.append(modelAlgorithm(networkModel, n_in=len(state_bounds[m][0]), n_out=len(action_bounds[m][0]), state_bounds=state_bounds[m], 
                          action_bounds=action_bounds[m], reward_bound=reward_bounds[m], settings_=settings__, print_info=print_info))
                    
                    print("Loaded algorithm: ", models)
                else:
                    print ("Unknown learning algorithm type: " + str(algorihtm_type))
                    raise ValueError("Unknown learning algorithm type: " + str(algorihtm_type))
            model = models
            """
            if ("policy_connections" in settings):
                for c in range(len(settings["policy_connections"])): 
                    print ("Sending policy ", model[settings["policy_connections"][c][0]],
                                                    " to policy ",  model[settings["policy_connections"][c][1]])
                    model[settings["policy_connections"][c][1]].setFrontPolicy(
                        model[settings["policy_connections"][c][0]])
            """
        else:
            modelAlgorithm = locate(algorihtm_type)
            if ( issubclass(modelAlgorithm, AlgorithmInterface)): ## Double check this load will work
                model = modelAlgorithm(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                              action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings, print_info=print_info)
                print("Loaded algorithm: ", model)
            else:
                print ("Unknown learning algorithm type: " + str(algorihtm_type))
                raise ValueError("Unknown learning algorithm type: " + str(algorihtm_type))
        # return model
        # sys.exit(2)
        
    if (settings['load_saved_model'] == "network_and_scales"):
        ### In this case we want to change algroithm but want to keep the policy network
        directory= getDataDirectory(settings)
        print ("Loading pre compiled network and scaling values, not learing algorithm.")
        file_name=directory+getAgentName()+"_Best.pkl"
        f = open(file_name, 'rb')
        model_ = dill.load(f)
        model_.setSettings(settings)
        model.setNetworkParameters(model_.getNetworkParameters())
        # model.setTargetModel(model_.getTargetModel())
        
        model.setStateBounds(model_.getStateBounds())
        model.setActionBounds(model_.getActionBounds())
        model.setRewardBounds(model_.getRewardBounds())
        f.close()
        
    print ("Using model type ", algorihtm_type , " : ", model)
    
    return model


def createEnvironment(config_file, env_type, settings, render=False, index=None):
    
    ### For multitasking, can specify a list of config files
    # if ( isinstance(config_file, list ) ):
    if type(config_file) is list:
        config_file = config_file[index]
        print ("Using config file: ", config_file)
    else:
        # print("Not a list hoser, it is a ", type(config_file), " for ", config_file)
        print (config_file[0])
    
    print("Creating sim Type: ", env_type)
    if env_type == 'ballgame_2d':
        from rlsimenv.BallGame2D import BallGame2D
        from sim.BallGame2DEnv import BallGame2DEnv
        file = open(config_file)
        conf = json.load(file)
        # print ("Settings: " + str(json.dumps(conf)))
        file.close()
        conf['render'] = render
        exp = BallGame2D(conf)
        exp = BallGame2DEnv(exp, settings)
        return exp
    
    
    elif env_type == 'open_AI_Gym':
        import gym
        from gym import wrappers
        from gym import envs
        from sim.OpenAIGymEnv import OpenAIGymEnv
        
        try:
            import roboschool
        except:
            print ("roboschool not installed")
            pass
        try:
            import gymdrl
        except:
            print ("Membrane/gymdrl not installed")
            pass
        try:
            import pybullet_envs
        except:
            print ("pybullet not installed")
            pass
        try:
            import multiworld
            multiworld.register_all_envs()
        except:
            print ("multiworld not installed")
            pass
        try:
            import metaworld.envs.mujoco
            metaworld.envs.mujoco.register_custom_envs()
        except:
            print ("metaworld not installed")
            pass
        try:
            import rlsimenv
        except:
            print ("rlsimenv not installed")
            pass
        try:
            ### This should import the terrainrlsim environments
            import simAdapter 
            import terrainRLSim
        except:
            print ("TerrainRLSim not installed")
            pass
        try:
            sys.path.append('/home/gberseth/playground/BayesianSurpriseCode/')
            import surprise
        except:
            print ("surprise not installed")
            pass
        # print(envs.registry.all())
        
        env_name = config_file
        env = gym.make(env_name)
        
        conf = copy.deepcopy(settings)
        conf['render'] = render
        exp = OpenAIGymEnv(env, conf)
        return exp
    
    elif env_type == 'UniTree':
        import gym
        from gym import wrappers
        from gym import envs
        from sim.OpenAIGymEnv import OpenAIGymEnv
        import envs.env_builder as env_builder
        motion_file="/home/gberseth/playground/motion_imitation/motion_imitation/motions/dog_pace.txt"
        env = env_builder.build_imitation_env(motion_files=[motion_file],
                                        num_parallel_envs=1,
                                        mode="train",
                                        enable_randomizer=False, ## Don't perform dynamic randomization
                                        enable_rendering=render, 
                                        dual_state = settings["use_dual_state_representations"])
        # print(envs.registry.all())
        conf = copy.deepcopy(settings)
        conf['render'] = False
        exp = OpenAIGymEnv(env, conf)
        return exp
    
    elif ((env_type == 'miniGrid')):
        ### This code could be cleaned up to work better.
        sys.path.append('/home/gberseth/playground/BayesianSurpriseCode/')
        if (settings["sim_config_file"] == "simpleRoomLatent-v0"):
            from surprise.envs.minigrid.envs.simple_room_latent import SimpleEnemyEnv
            from surprise.buffers.buffers import BernoulliBuffer
            from surprise.wrappers.base_surprise import BaseSurpriseWrapper
            from surprise.wrappers.visitation_count import VisitationCountWrapper
            from sim.OpenAIGymEnv import OpenAIGymEnv
            env_name = config_file
            def env_factory():
                #env = SimpleEnemyEnv(max_steps=500, agent_pos=(6,9))
                env = SimpleEnemyEnv(max_steps=500)
                env.see_through_walls = True
                env = BaseSurpriseWrapper(
                        env, 
                        BernoulliBuffer(132), 
                        env.max_steps
                    )
                return env
            env = env_factory()
        elif (settings["sim_config_file"] == "simpleRoomHMM-v0"):
            from surprise.envs.minigrid.envs.simple_room_hmm import SimpleEnemyEnvHMM
            from surprise.buffers.buffers import BernoulliBuffer
            from surprise.wrappers.base_surprise import BaseSurpriseWrapper
            from surprise.wrappers.visitation_count import VisitationCountWrapper
            from sim.OpenAIGymEnv import OpenAIGymEnv
            import numpy as np
            env_name = config_file
            def env_factory():
                #env = SimpleEnemyEnv(max_steps=500, agent_pos=(6,9))
                env = SimpleEnemyEnvHMM(max_steps=500)
                env.see_through_walls = True
                return env
            env = env_factory()
        elif settings["sim_config_file"] == "simpleRoom-v0":
            from surprise.envs.minigrid.envs.simple_room import SimpleEnemyEnv
            from surprise.buffers.buffers import BernoulliBuffer
            from surprise.wrappers.base_surprise import BaseSurpriseWrapper
            from surprise.wrappers.visitation_count import VisitationCountWrapper
            from sim.OpenAIGymEnv import OpenAIGymEnv
            env_name = config_file
            def env_factory():
                #env = SimpleEnemyEnv(max_steps=500, agent_pos=(6,9))
                env = SimpleEnemyEnv(max_steps=500)
                env.see_through_walls = True
                env = BaseSurpriseWrapper(
                        env, 
                        BernoulliBuffer(49), 
                        env.max_steps
                    )
                return env
            env = env_factory()
        elif settings["sim_config_file"] == "simpleRoomVisual-v0":
            from surprise.envs.minigrid.envs.simple_room_visual import SimpleRoomVisualEnv
            from surprise.buffers.buffers import BernoulliBuffer
            from surprise.wrappers.base_surprise import BaseSurpriseWrapper
            from surprise.wrappers.visitation_count import VisitationCountWrapper
            from sim.OpenAIGymEnv import OpenAIGymEnv
            env_name = config_file
            def env_factory():
                #env = SimpleEnemyEnv(max_steps=500, agent_pos=(6,9))
                env = SimpleRoomVisualEnv(max_steps=500)
                env.see_through_walls = True
                # env = BaseSurpriseWrapper(
                #         env, 
                #         BernoulliBuffer(49), 
                #         env.max_steps
                #     )
                return env
            env = env_factory()
        else:
            raise ValueError("Unknown minigrid sim config! {}".format(settings["sim_config_file"]))
        conf = copy.deepcopy(settings)
        conf['render'] = render
        exp = OpenAIGymEnv(env, conf, multiAgent=False)
        return exp
    
    
    
    
    elif ( (env_type == 'GymMultiChar') 
        or (env_type == 'terrainRLSim')
        ):
        # terrainRL_PATH = os.environ['TERRAINRL_PATH']
        # sys.path.append(terrainRL_PATH+'/lib')
        # from simAdapter import terrainRLAdapter
        # from sim.TerrainRLEnv import TerrainRLEnv
        from simAdapter import terrainRLSim
        from sim.GymMultiCharEnv import GymMultiCharEnv
        if ("GPU_BUS_Index" in settings):
            env = terrainRLSim.getEnv(env_name=config_file, render=render, GPU_device=int(settings["GPU_BUS_Index"]))
        else:
            env = terrainRLSim.getEnv(env_name=config_file, render=render)
        print ("Using Environment Type: " + str(env_type) + ", " + str(config_file))
        
        ### Check action space size
        actionSpace = env.getActionSpace()
        
        assert (len(actionSpace.high) == len(settings["action_bounds"][0]), 
                "Length of action vector is " + str (len(settings["action_bounds"][0])) + " is should be " + 
                str(len(actionSpace.high)))
        # sim.setRender(render)
        # sim.init()
        conf = copy.deepcopy(settings)
        conf['render'] = render
        exp = GymMultiCharEnv(env, conf)
        # env.getEnv().setRender(render)
        # exp = TerrainRLEnv(env.getEnv(), settings)
        return exp
    else:
        print ("Invalid environment type: " + str(env_type))
        raise ValueError("Invalid environment type: " + str(env_type))
        # sys.exit()
    
    exp._conf = c # OMFG HACK so that python does not garbage collect the configuration and F everything up!    
    return exp

def createActor(env_type, settings, experience):
    actor=None
   
    if (env_type == 'open_AI_Gym'
          or (env_type == 'RLSimulations')
          or (env_type == 'miniGrid')
          or (env_type == 'UniTree')
          ):
        from actor.OpenAIGymActor import OpenAIGymActor
        actor = OpenAIGymActor(settings, experience)
    elif (env_type == 'terrainRLSim'):
        from actor.OpenAIGymActorMARL import OpenAIGymActorMARL
        actor = OpenAIGymActorMARL(settings, experience)
    elif (env_type == 'MultiworldFixedLLC') or (env_type == 'MetaworldFixedLLC'):
        from actor.MultiworldMultiCharActor import MultiworldMultiCharActor
        actor = MultiworldMultiCharActor(settings, experience)
    elif (env_type == 'GymMultiChar'):
        from actor.GymMultiCharActor import GymMultiCharActor
        actor = GymMultiCharActor(settings, experience)
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
        from algorithm.Sampler import Sampler
        # modelClass = my_import(path_)
        modelAlgorithm = locate(settings['sampling_method'])
        if ( issubclass(modelAlgorithm, Sampler)): ## Double check this load will work
            sampler = modelAlgorithm(exp, settings['look_ahead_planning_steps'], settings)
            print("Loaded sampler: ", sampler)
            # return model
        else:
            print ("Sampler method not supported: " + str(settings['sampling_method']) )
            sys.exit()
    
    return sampler


def createNewFDModel(settings, env, model):
    print ("Creating new FD model with different session")
    
    state_bounds = getFDStateSize(settings)
    action_bounds = settings['action_bounds']
    
    forwardDynamicsModel = None
    if (settings['train_forward_dynamics']):
        if ("perform_multiagent_training" in settings):
            forwardDynamicsModel = []
            for i in range(settings["perform_multiagent_training"]):
                actor = createActor(settings['environment_type'], settings, None)
                settings__ = copy.deepcopy(settings)
                settings__['state_bounds'] = settings['state_bounds'][i]
                settings__['action_bounds'] = settings['action_bounds'][i]
                settings__['reward_bounds'] = settings['reward_bounds'][i]
                if ("fd_action_bounds" in settings):
                    settings__['action_bounds'] = settings['fd_action_bounds'][i]
                settings__['fd_network_layer_sizes'] = settings['fd_network_layer_sizes'][i]
                settings__['reward_network_layer_sizes'] = settings['reward_network_layer_sizes'][i]
                state_bounds = getFDStateSize(settings__)
                action_bounds = settings__['action_bounds']
                settings__["agent_id"] = i
                if  (type(settings['train_forward_dynamics']) is list 
                    and (settings['train_forward_dynamics'][i] == False)):
                    forwardDynamicsModel_ = None
                    forwardDynamicsModel.append(forwardDynamicsModel_)
                    continue
                elif ( settings['forward_dynamics_model_type'] == "SingleNet"
                     and (settings['use_single_network'] == True)):
                    print ("Creating forward dynamics network: Using single network model")
                    # settings__["critic_network_layer_sizes"] = settings["critic_network_layer_sizes"][m]
                    # settings__["policy_network_layer_sizes"] = settings["policy_network_layer_sizes"][m]
                    forwardDynamicsModel_ = createForwardDynamicsModel(settings__, state_bounds, action_bounds, None, None, agentModel=model)
                    # forwardDynamicsModel = model
                else:
                    print ("Creating forward dynamics network")
                    # forwardDynamicsModel = ForwardDynamicsNetwork(state_length=len(state_bounds[0]),action_length=len(action_bounds[0]), state_bounds=state_bounds, action_bounds=action_bounds, settings_=settings)
                    forwardDynamicsModel_ = createForwardDynamicsModel(settings__, state_bounds, action_bounds, None, None, agentModel=None)
                # masterAgent.setForwardDynamics(forwardDynamicsModel)
                forwardDynamicsModel_.setActor(actor)
                # forwardDynamicsModel.setEnvironment(exp)
                forwardDynamicsModel_.init(len(state_bounds[0]), len(action_bounds[0]), state_bounds, action_bounds, actor, None, settings)
                forwardDynamicsModel.append(forwardDynamicsModel_)
        else:
            actor = createActor(settings['environment_type'], settings, None)
            if ( settings['forward_dynamics_model_type'] == "SingleNet"
                 and (settings['use_single_network'] == True)):
                print ("Creating forward dynamics network: Using single network model")
                forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, None, None, agentModel=model)
                # forwardDynamicsModel = model
            else:
                print ("Creating forward dynamics network")
                forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, None, None, agentModel=None)
            forwardDynamicsModel.setActor(actor)
            forwardDynamicsModel.init(len(state_bounds[0]), len(action_bounds[0]), state_bounds, action_bounds, actor, None, settings)
            

    if ("fd_algorithm" in settings and
            settings["fd_algorithm"] == "algorithm.VAE.VAE" and
            "VAE" in settings["environment_type"]):
        env.setVAE(forwardDynamicsModel)

    return forwardDynamicsModel

def createForwardDynamicsModel(settings, state_bounds, action_bounds, actor, exp, agentModel, reward_bounds=0, print_info=True):
    import numpy as np
    directory= getDataDirectory(settings)
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
        file_name_dynamics=data_folder+"forward_dynamics_"+"_Best.pkl"
        forwardDynamicsModel = dill.load(open(file_name_dynamics))
    elif settings["forward_dynamics_predictor"] == "network":
        print ("Using forward dynamics method: " + str(settings["forward_dynamics_predictor"]))
        if (settings['load_saved_model'] 
            or ('load_saved_fd_model' in settings and 
             (settings['load_saved_fd_model'] == True))):
            print ("**** Loading pre compiled fd network")
            if ("learning_backend" in settings and 
                ((settings['learning_backend'] == "tensorflow")
                 or (settings['learning_backend'] == "theano")
                )):
                from algorithm.AlgorithmInterface import AlgorithmInterface
                state_bounds__ = getFDStateSize(settings)
                settings_ = copy.deepcopy(settings)
                ### This is faster....
                settings_['load_saved_model'] = False
                networkModel = createForwardDynamicsNetwork(state_bounds__, action_bounds, settings)
                # networkModel = createNetworkModel(settings["model_type"], state_bounds, action_bounds, reward_bounds, settings_)
                # modelClass = my_import(path_)
                algorihtm_type = settings['fd_algorithm']
                modelAlgorithm = locate(algorihtm_type)
                if ( issubclass(modelAlgorithm, AlgorithmInterface)): ## Double check this load will work
                    print ("networkModel: ", networkModel)
                    forwardDynamicsModel = modelAlgorithm(networkModel, state_length=len(state_bounds__[0]), 
                                           action_length=len(action_bounds[0]), state_bounds=state_bounds, 
                                  action_bounds=action_bounds, reward_bounds=reward_bounds, settings_=settings)
                    forwardDynamicsModel.setSettings(settings)
                    print ("Loading pre trained FD model:")
                    if (settings['load_saved_model'] == 'last'):
                        try: 
                            forwardDynamicsModel.loadFrom(directory+"forward_dynamics"+str(settings["agent_id"]))
                        except Exception as e:
                            forwardDynamicsModel.loadFrom(directory+"forward_dynamics")
                    else:
                        forwardDynamicsModel.loadFrom(directory+"forward_dynamics"+"_Best")
                    print("Loaded algorithm: ", forwardDynamicsModel)
                    # return model
                else:
                    print ("Unknown learning algorithm type: " + str(algorihtm_type))
                    raise ValueError("Unknown learning algorithm type: " + str(algorihtm_type))
                # sys.exit(2)
            else:
                print ("Loading pre compiled network")
                if (settings['load_saved_model'] == 'last'):
                    file_name=directory+getAgentName()+".pkl"
                else:
                    file_name=directory+getAgentName()+"_Best.pkl"
                f = open(file_name, 'rb')
                forwardDynamicsModel = dill.load(f)
                f.close()
                forwardDynamicsModel.setSettings(settings)
                if (settings['load_saved_model'] == 'last'):
                    forwardDynamicsModel.loadFrom(directory+"forward_dynamics")
                else:
                    forwardDynamicsModel.loadFrom(directory+"forward_dynamics"+"_Best")
            print ("FD state bounds: ", forwardDynamicsModel.getStateBounds())
        else:
            if ( settings['forward_dynamics_model_type'] == "SingleNet"
                 and (settings['use_single_network'] == True)):
                ## Hopefully this will allow for parameter sharing across both models...
                fd_net = agentModel.getModel()
            else:
                fd_net = createForwardDynamicsNetwork(state_bounds, action_bounds, settings)
            
            if ('fd_algorithm' in settings ):
                from algorithm.AlgorithmInterface import AlgorithmInterface
                algorihtm_type = settings['fd_algorithm']
                # modelClass = my_import(path_)
                modelAlgorithm = locate(algorihtm_type)
                if ( issubclass(modelAlgorithm, AlgorithmInterface)): ## Double check this load will work
                    forwardDynamicsModel = modelAlgorithm(fd_net, state_length=len(state_bounds[0]), action_length=len(action_bounds[0]),
                                            state_bounds=state_bounds, 
                                  action_bounds=action_bounds, settings_=settings,
                                  reward_bounds=reward_bounds, 
                                  print_info=print_info)
                    log.info("Loaded FD algorithm: {}".format(forwardDynamicsModel))
                    # return model
                else:
                    print ("Unknown learning algorithm type: " + str(algorihtm_type))
                    raise ValueError("Unknown learning algorithm type: " + str(algorihtm_type))
                # sys.exit(2)
            elif ('train_state_encoding' in settings and (settings['train_state_encoding'])):
                from algorithm.EncodingModel import EncodingModel
                forwardDynamicsModel = EncodingModel(fd_net, state_length=len(state_bounds[0]), action_length=len(action_bounds[0]), 
                                                       state_bounds=state_bounds, action_bounds=action_bounds, settings_=settings)
            elif ('train_gan' in settings and (settings['train_gan'])):
                from algorithm.GAN import GAN
                forwardDynamicsModel = GAN(fd_net, state_length=len(state_bounds[0]), action_length=len(action_bounds[0]), 
                                                       state_bounds=state_bounds, action_bounds=action_bounds, settings_=settings,
                                                       reward_bounds=reward_bounds)
            else:
                from algorithm.ForwardDynamics import ForwardDynamics
                forwardDynamicsModel = ForwardDynamics(fd_net, state_length=len(state_bounds[0]), action_length=len(action_bounds[0]), 
                                                       state_bounds=state_bounds, action_bounds=action_bounds, settings_=settings,
                                                       reward_bounds=reward_bounds)
    else:
        print ("Unrecognized forward dynamics method: " + str(settings["forward_dynamics_predictor"]))
        raise ValueError("Unrecognized forward dynamics method: " + str(settings["forward_dynamics_predictor"]))
        # sys.exit()
        
    return forwardDynamicsModel

def createForwardDynamicsNetwork(state_bounds, action_bounds, settings, 
                                 stateName="State", resultStateName="ResultState",**kwargs):
    
    from model.ModelInterface import ModelInterface
    # modelClass = my_import(path_)
    fd_net_type = settings["forward_dynamics_model_type"]
    if (fd_net_type == "SingleNet"):
        ### Use adaptive instead
        fd_net_type = "model.FDNNKerasAdaptive.FDNNKerasAdaptive"
    print("Loading FD model type:", fd_net_type)
    modelClass = locate(fd_net_type)
    if ( issubclass(modelClass, ModelInterface)): ## Double check this load will work
        model = modelClass(len(state_bounds[0]), len(action_bounds[0]), 
                            state_bounds, action_bounds, settings_=settings, reward_bound=settings["reward_bounds"],
                            stateName=stateName, resultStateName=stateName, **kwargs)
        print("Created model: ", model)
        return model
    else:
        print ("Unrecognized forward dynamics network type: " + str(settings["forward_dynamics_model_type"]))
        raise ValueError("Unrecognized forward dynamics network type: " + str(settings["forward_dynamics_model_type"]))
    # sys.exit()

