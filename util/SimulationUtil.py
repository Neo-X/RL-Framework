import copy
import sys
sys.setrecursionlimit(50000)
import os
import json
sys.path.append("../")
sys.path.append("../env")
sys.path.append("../characterSimAdapter/")
sys.path.append("../simbiconAdapter/")
sys.path.append("../simAdapter/")

from pydoc import locate

import dill
import dill as pickle
import dill as cPickle

# import cProfile, pstats, io
# import memory_profiler
# import psutil
import gc
# from guppy import hpy; h=hpy()
# from memprof import memprof

def updateSettings(settings1, newSettings):
    """
        Replace all of the setting sin settings1 with the settings in newSettings
    """
    for key_ in newSettings.keys():
        settings1[key_] = newSettings[key_]
        
    return settings1
    
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

def setupEnvironmentVariable(settings):
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
    return getRootDataDirectory(settings)+"/"+settings["data_folder"]+"/"

def getRootDataDirectory(settings):
    return settings["environment_type"]+"/"+settings["agent_name"]

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
        state_bounds__ = np.array([[0] * settings["dense_state_size"], 
                         [1] * settings["dense_state_size"]])
        # experiencefd.setStateBounds(state_bounds__)
    return state_bounds__

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

def createNetworkModel(model_type, state_bounds, action_bounds, reward_bounds, settings, print_info=False):
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
    elif (model_type == "Deep_CNN_2D" ):
        from model.DeepCNN2D import DeepCNN2D
        model = DeepCNN2D(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
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
    elif (model_type == "Deep_CNN_SingleNet_Big" ):
        from model.DeepCNNSingleNetBig import DeepCNNSingleNetBig
        print("Creating network model: ", model_type)
        model = DeepCNNSingleNetBig(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
        return model 
    
    elif (model_type == "DumbModel" ):
        model = DumbModel(n_in=len(state_bounds[0]), n_out=n_out_, state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    else:
        from model.ModelInterface import ModelInterface
        # modelClass = my_import(path_)
        modelClass = locate(model_type)
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
    action_bounds = np.array(settings['action_bounds'])
    if ("perform_multiagent_training" in settings):
        pass
    else:
        networkModel = createNetworkModel(settings["model_type"], state_bounds, action_bounds, reward_bounds, settings, print_info=print_info)
    num_actions= len(discrete_actions) # number of rows
    if settings['action_space_continuous']:
            action_bounds = np.array(settings["action_bounds"], dtype=float)
            num_actions = len(action_bounds[0])
    
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
            modelAlgorithm = locate(algorihtm_type)
            if ( issubclass(modelAlgorithm, AlgorithmInterface)): ## Double check this load will work
                if ("perform_multiagent_training" in settings):
                    models = []
                    assert settings["perform_multiagent_training"] == len(state_bounds)
                    for m in range(settings["perform_multiagent_training"]):
                        settings__ = copy.deepcopy(settings_)
                        settings__["critic_network_layer_sizes"] = settings["critic_network_layer_sizes"][m]
                        settings__["policy_network_layer_sizes"] = settings["policy_network_layer_sizes"][m]
                        networkModel = createNetworkModel(settings__["model_type"], state_bounds[m], action_bounds[m], reward_bounds[m], settings__, print_info=print_info)
                        print ("networkModel: ", networkModel)
                        model_ = modelAlgorithm(networkModel, n_in=len(state_bounds[m][0]), n_out=len(action_bounds[m][0]), state_bounds=state_bounds[m], 
                                  action_bounds=action_bounds[m], reward_bound=reward_bounds[m], settings_=settings__, print_info=print_info)
                        model_.setSettings(settings__) ### Maybe this should be the normal settings...
                        if (settings['load_saved_model'] == 'last'):
                            model_.loadFrom(directory+getAgentName()+str(m))
                        else:
                            model_.loadFrom(directory+getAgentName()+str(m)+"_Best")
                        models.append(model_)
                    print("Loaded algorithm: ", models)
                    model = models
                else:
                    networkModel = createNetworkModel(settings["model_type"], state_bounds, action_bounds, reward_bounds, settings_)
                    print ("networkModel: ", networkModel)
                    model = modelAlgorithm(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                                  action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
                    model.setSettings(settings)
                    if (settings['load_saved_model'] == 'last'):
                        model.loadFrom(directory+getAgentName())
                    else:
                        model.loadFrom(directory+getAgentName()+"_Best")
                print("Loaded algorithm: ", model)
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
            model = dill.load(f)
            f.close()
            model.setSettings(settings)
            if (settings['load_saved_model'] == 'last'):
                model.loadFrom(directory+getAgentName())
            else:
                model.loadFrom(directory+getAgentName()+"_Best")
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
    elif (algorihtm_type == "Distillation" ):
        from algorithm.Distillation import Distillation
        model = Distillation(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)    
    else:
        from algorithm.AlgorithmInterface import AlgorithmInterface
        # modelClass = my_import(path_)
        modelAlgorithm = locate(algorihtm_type)
        if ( issubclass(modelAlgorithm, AlgorithmInterface)): ## Double check this load will work
            if ("perform_multiagent_training" in settings):
                models = []
                assert settings["perform_multiagent_training"] == len(state_bounds)
                settings_ = copy.deepcopy(settings)
                for m in range(settings["perform_multiagent_training"]):
                    settings__ = copy.deepcopy(settings)
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
                    networkModel = createNetworkModel(settings__["model_type"], state_bounds[m], action_bounds[m], reward_bounds[m], settings__, print_info=print_info)
                    models.append(modelAlgorithm(networkModel, n_in=len(state_bounds[m][0]), n_out=len(action_bounds[m][0]), state_bounds=state_bounds[m], 
                          action_bounds=action_bounds[m], reward_bound=reward_bounds[m], settings_=settings__, print_info=print_info))
                    
                    print("Loaded algorithm: ", models)
                model = models
            else:
                model = modelAlgorithm(networkModel, n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                              action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings, print_info=print_info)
                print("Loaded algorithm: ", model)
            # return model
        else:
            print ("Unknown learning algorithm type: " + str(algorihtm_type))
            raise ValueError("Unknown learning algorithm type: " + str(algorihtm_type))
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
    elif env_type == 'ballgame_1d':
        from rlsimenv.BallGame1D import BallGame1D
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
        from rlsimenv.GapGame1D import GapGame1D
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
        from rlsimenv.GapGame2D import GapGame2D
        from sim.GapGame2DEnv import GapGame2DEnv
        file = open(config_file)
        conf = json.load(file)
        # print ("Settings: " + str(json.dumps(conf)))
        file.close()
        conf.update( settings.items() )
        conf['render'] = render
        exp = GapGame2D(conf)
        exp = GapGame2DEnv(exp, settings)
        return exp
    elif env_type == 'nav_Game':
        from rlsimenv.NavGame import NavGame
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
    elif env_type == 'nav_Game_MultiAgent':
        from rlsimenv.NavGameMultiAgent import NavGameMultiAgent
        from sim.NavGameMultiAgentEnv import NavGameMultiAgentEnv
        # file = open(config_file)
        # conf = json.load(file)
        conf = copy.deepcopy(settings)
        # print ("Settings: " + str(json.dumps(conf)))
        # file.close()
        conf['render'] = render
        exp = NavGameMultiAgent(conf)
        exp = NavGameMultiAgentEnv(exp, settings)
        return exp
    elif env_type == 'Particle_Sim':
        from rlsimenv.ParticleGame import ParticleGame
        from sim.ParticleSimEnv import ParticleSimEnv
        # file = open(config_file)
        # conf = json.load(file)
        conf = copy.deepcopy(settings)
        # print ("Settings: " + str(json.dumps(conf)))
        # file.close()
        conf['render'] = render
        exp = ParticleGame(conf)
        exp = ParticleSimEnv(exp, settings)
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
        # from OpenGL import GL
        # load_roboschool
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
    
    elif ((env_type == 'RLSimulations')):
        from rlsimenv.EnvWrapper import getEnv
        from sim.OpenAIGymEnv import OpenAIGymEnv
        env_name = config_file
        env = getEnv(env_name, render=render)
        
        conf = copy.deepcopy(settings)
        conf['render'] = render
        if (env.getNumberofAgents() > 1):
            exp = OpenAIGymEnv(env, conf, multiAgent=True)
        else:
            exp = OpenAIGymEnv(env, conf, multiAgent=False)
        return exp
    
    elif ((env_type == 'HRLSimulations')):
        from rlsimenv.EnvWrapper import getEnv
        from sim.OpenAIGymHRLEnv import OpenAIGymHRLEnv
        env_name = config_file
        env = getEnv(env_name, render=render)
        
        conf = copy.deepcopy(settings)
        conf['render'] = render
        if (env.getNumberofAgents() > 1):
            exp = OpenAIGymHRLEnv(env, conf, multiAgent=True)
        else:
            exp = OpenAIGymHRLEnv(env, conf, multiAgent=False)
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
    elif env_type == 'terrainRLSimOld':
        # terrainRL_PATH = os.environ['TERRAINRL_PATH']
        # sys.path.append(terrainRL_PATH+'/lib')
        # from simAdapter import terrainRLAdapter
        # from sim.TerrainRLEnv import TerrainRLEnv
        from simAdapter import terrainRLSim
        from sim.OpenAIGymEnv import OpenAIGymEnv
        
        env = terrainRLSim.getEnv(env_name=config_file, render=render)
        print ("Using Environment Type: " + str(env_type) + ", " + str(config_file))
        # sim.setRender(render)
        # sim.init()
        conf = copy.deepcopy(settings)
        conf['render'] = render
        exp = OpenAIGymEnv(env, conf)
        # env.getEnv().setRender(render)
        # exp = TerrainRLEnv(env.getEnv(), settings)
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
        
        assert (len(actionSpace.getMaximum() == len(settings["action_bounds"][0])), 
                "Length of action vector is " + str (len(settings["action_bounds"][0])) + " is should be " + 
                str(len(actionSpace.getMaximum())))
        # sim.setRender(render)
        # sim.init()
        conf = copy.deepcopy(settings)
        conf['render'] = render
        exp = GymMultiCharEnv(env, conf)
        # env.getEnv().setRender(render)
        # exp = TerrainRLEnv(env.getEnv(), settings)
        return exp
        
    elif env_type == 'terrainRLBiped2D':
        terrainRL_PATH = os.environ['TERRAINRL_PATH']
        sys.path.append(terrainRL_PATH+'/lib')
        from simAdapter import terrainRLAdapter
        from sim.TerrainRLEnv import TerrainRLEnv
        sim = terrainRLAdapter.cSimAdapter(['train', '-arg_file=', terrainRL_PATH+'/'+config_file, '-relative_file_path=', terrainRL_PATH+'/'])
        sim.setRender(render)
        # sim.init(['train', '-arg_file=', config_file])
        # print ("Num state: ", c._NUMBER_OF_STATES)
        # sim = simbiconAdapter.SimbiconWrapper(c)
        print ("Using Environment Type: " + str(env_type))
        exp = TerrainRLEnv(sim, settings)
        # exp._conf = c # OMFG HACK so that python does not garbage collect the configuration and F everything up!
        return exp
    elif env_type == 'terrainRLFlatBiped2D':
        terrainRL_PATH = os.environ['TERRAINRL_PATH']
        sys.path.append(terrainRL_PATH+'/lib')
        from simAdapter import terrainRLAdapter
        from sim.TerrainRLFlatEnv import TerrainRLFlatEnv
        sim = terrainRLAdapter.cSimAdapter(['train', '-arg_file=', terrainRL_PATH+'/'+config_file, '-relative_file_path=', terrainRL_PATH+'/'])
        sim.setRender(render)
        # sim.init(['train', '-arg_file=', config_file])
        # print ("Num state: ", c._NUMBER_OF_STATES)
        # sim = simbiconAdapter.SimbiconWrapper(c)
        print ("Using Environment Type: " + str(env_type))
        exp = TerrainRLFlatEnv(sim, settings)
        # exp._conf = c # OMFG HACK so that python does not garbage collect the configuration and F everything up!
        return exp
    elif (env_type == 'terrainRLImitateBiped2D' or (env_type == 'terrainRLImitateBiped3D')):
        terrainRL_PATH = os.environ['TERRAINRL_PATH']
        sys.path.append(terrainRL_PATH+'/lib')
        from simAdapter import terrainRLAdapter
        from sim.TerrainRLImitateEnv import TerrainRLImitateEnv
        sim = terrainRLAdapter.cSimAdapter(['train', '-arg_file=', terrainRL_PATH+'/'+config_file, '-relative_file_path=', terrainRL_PATH+'/'])
        sim.setRender(render)
        # sim.init(['train', '-arg_file=', config_file])
        # print ("Num state: ", c._NUMBER_OF_STATES)
        # sim = simbiconAdapter.SimbiconWrapper(c)
        print ("Using Environment Type: " + str(env_type))
        exp = TerrainRLImitateEnv(sim, settings)
        # exp._conf = c # OMFG HACK so that python does not garbage collect the configuration and F everything up!
        return exp
    elif ((env_type == 'terrainRLHLCBiped3D')):
        terrainRL_PATH = os.environ['TERRAINRL_PATH']
        sys.path.append(terrainRL_PATH+'/lib')
        from simAdapter import terrainRLAdapter
        from sim.TerrainRLHLCEnv import TerrainRLHLCEnv
        sim = terrainRLAdapter.cSimAdapter(['train', '-arg_file=', terrainRL_PATH+'/'+config_file, '-relative_file_path=', terrainRL_PATH+'/'])
        sim.setRender(render)
        # sim.init(['train', '-arg_file=', config_file])
        # print ("Num state: ", c._NUMBER_OF_STATES)
        # sim = simbiconAdapter.SimbiconWrapper(c)
        print ("Using Environment Type: " + str(env_type))
        exp = TerrainRLHLCEnv(sim, settings)
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
    elif (env_type == 'nav_Game_MultiAgent'):
        from actor.NavGameMultiAgentActor import NavGameMultiAgentActor
        actor = NavGameMultiAgentActor(settings, experience)    
    elif (env_type == 'Particle_Sim'):
        from actor.ParticleSimActor import ParticleSimActor
        actor = ParticleSimActor(settings, experience)
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
    elif ( env_type == 'terrainRLImitateBiped2D' or (env_type == 'terrainRLImitateBiped3D')
          # or (env_type == 'terrainRLSim') 
          ):
        from actor.TerrainRLImitationActor import TerrainRLImitationActor
        actor = TerrainRLImitationActor(settings, experience)
    elif (env_type == 'terrainRLHLCBiped3D'):
        from actor.TerrainRLHLCActor import TerrainRLHLCActor
        actor = TerrainRLHLCActor(settings, experience)
    elif (env_type == 'paperGibbon_env'):
        from actor.PaperGibbonAgent import PaperGibbonAgent
        actor = PaperGibbonAgent(settings, experience)
    elif (env_type == 'pendulum'
          or (env_type == 'pendulum_env')
          ):
        from actor.ActorInterface import ActorInterface
        actor = ActorInterface(settings, experience)
    elif (env_type == 'open_AI_Gym'
          or (env_type == 'RLSimulations')
          ):
        from actor.OpenAIGymActor import OpenAIGymActor
        actor = OpenAIGymActor(settings, experience)
    elif ( (env_type == 'HRLSimulations')
          ):
        from actor.OpenAIGymHRLActor import OpenAIGymHRLActor
        actor = OpenAIGymHRLActor(settings, experience)
    elif (
          (env_type == 'terrainRLSim')
          ):
        from actor.OpenAIGymActor2 import OpenAIGymActor2
        actor = OpenAIGymActor2(settings, experience)
    elif env_type == 'GymMultiChar':
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
    state_bounds = settings['state_bounds']
    if ("use_dual_dense_state_representations" in settings
        and (settings["use_dual_dense_state_representations"] == True)):
        state_bounds = settings['state_bounds']
    elif (("use_dual_state_representations" in settings
          and (settings["use_dual_state_representations"] == True))
        and (not (settings["forward_dynamics_model_type"] == "SingleNet"))
        and ("fd_use_multimodal_state" in settings
             and (settings["fd_use_multimodal_state"] == True))
        ):
        state_size__ = settings["fd_num_terrain_features"] + settings["dense_state_size"]
        if ("append_camera_velocity_state" in settings 
            and (settings["append_camera_velocity_state"] == True)):
            state_size__ = state_size__ + 2
        elif ("append_camera_velocity_state" in settings 
            and (settings["append_camera_velocity_state"] == "3D")):
            state_size__ = state_size__ + 3
            # print ("***** Adding ", settings["append_camera_velocity_state"], " camera")
            # sys.exit()
        print ("Creating multi modal state size****")
        state_bounds = [[0] * (state_size__), 
                                 [1] * (state_size__)]
    elif ("use_dual_state_representations" in settings
        and (settings["use_dual_state_representations"] == True)
        and (not (settings["forward_dynamics_model_type"] == "SingleNet"))):
        state____ = env.getState()
        print("state____ shape: ", state____)
        
        state_bounds = [[0] * state____[0][1].size, 
                                 [1] * state____[0][1].size]
    # if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
    #     print("fd state bounds:", state_bounds)
    action_bounds = settings['action_bounds']
    
    forwardDynamicsModel = None
    if (settings['train_forward_dynamics']):
        actor = createActor(settings['environment_type'], settings, None)
        if ( settings['forward_dynamics_model_type'] == "SingleNet"
             and (settings['use_single_network'] == True)):
            print ("Creating forward dynamics network: Using single network model")
            forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, None, None, agentModel=model)
            # forwardDynamicsModel = model
        else:
            print ("Creating forward dynamics network")
            # forwardDynamicsModel = ForwardDynamicsNetwork(state_length=len(state_bounds[0]),action_length=len(action_bounds[0]), state_bounds=state_bounds, action_bounds=action_bounds, settings_=settings)
            forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, None, None, agentModel=None)
        # masterAgent.setForwardDynamics(forwardDynamicsModel)
        forwardDynamicsModel.setActor(actor)
        # forwardDynamicsModel.setEnvironment(exp)
        forwardDynamicsModel.init(len(state_bounds[0]), len(action_bounds[0]), state_bounds, action_bounds, actor, None, settings)
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
                state_bounds__ = state_bounds
                if ("use_dual_state_representations" in settings
                    and "fd_num_terrain_features" in settings
                        and (settings["use_dual_state_representations"] == True)
                    and ("fd_use_multimodal_state" in settings
                     and (settings["fd_use_multimodal_state"] == True))):
                        state_size__ = settings["fd_num_terrain_features"] + settings["dense_state_size"]
                        if ("append_camera_velocity_state" in settings 
                            and (settings["append_camera_velocity_state"] == True)):
                            state_size__ = state_size__ + 2
                        elif ("append_camera_velocity_state" in settings 
                            and (settings["append_camera_velocity_state"] == "3D")):
                            state_size__ = state_size__ + 3
                        print ("Creating multi modal state size****")
                        state_bounds__ = [[0] * (state_size__), 
                                                 [1] * (state_size__)]
                elif ("use_dual_state_representations" in settings
                    and "fd_num_terrain_features" in settings
                        and (settings["use_dual_state_representations"] == True)):
                        state_bounds__ = np.array([[0] * settings["fd_num_terrain_features"], 
                                         [1] * settings["fd_num_terrain_features"]])
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
                    print("Loaded FD algorithm: ", forwardDynamicsModel)
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

def createForwardDynamicsNetwork(state_bounds, action_bounds, settings, stateName="State", resultStateName="ResultState"):
    
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
                                                         stateName=stateName, resultStateName=stateName)
            print("Created model: ", model)
            return model
        else:
            print ("Unrecognized forward dynamics network type: " + str(settings["forward_dynamics_model_type"]))
            raise ValueError("Unrecognized forward dynamics network type: " + str(settings["forward_dynamics_model_type"]))
        # sys.exit()
    import lasagne
    print ("Number of Forward Dynamics network parameters", lasagne.layers.count_params(forwardDynamicsNetwork.getForwardDynamicsNetwork()))
    print ("Number of Reward predictor network parameters", lasagne.layers.count_params(forwardDynamicsNetwork.getRewardNetwork()))
    return forwardDynamicsNetwork

