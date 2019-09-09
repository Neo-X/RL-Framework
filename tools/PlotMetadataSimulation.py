# import matplotlib
# matplotlib.use('Agg')

# import matplotlib.pyplot as plt
# from matplotlib import mpl
# import matplotlib.animation as animation
import sys
import json
sys.path.append("../")
try:
    from PolicyTrainVisualize import PolicyTrainVisualize
except Exception as e:
    from tools.PolicyTrainVisualize import PolicyTrainVisualize
import os
import re

"""
    plot_meta_simulation.py <settings_file_name> <path_to_data>
    Example:
    python3 tools/plot_meta_simulation.py ../../../backup/Particle_Sim/algorithm.PPO.PPO/ ../../../backup/Particle_Sim/algorithm.PPO.PPO/Nav_Sphere_SMBAE_10D_Hyper_action_learning_rate_0.025/PPO_SMBAE_10D_Hyper.json
"""

def getDataFolderNames(prefixPath, folderPrefix, settings):
    path = prefixPath
    folder_ = folderPrefix
    name_suffix = "/"+settings["model_type"]+"/"+"trainingData_" + str(settings['agent_name']) + ".json"
    folderNames = []
    print ("path: ", path)
    print ("folder_: ", folder_)
    print ("name_suffix: ", name_suffix)
    if os.path.exists(path+folder_):
        for filename in os.listdir(path+folder_):
            print ("checking path: ", filename)
            if re.match("_\d+", filename):
                folderNames.append(path+folder_ + filename + name_suffix)
                print ("***Found Folder Name: ", folderNames[-1])
    ## For the case where I put a slash at the end of the path name...
    path = path + folder_[:-1]
    print ("New path: ", path)
    if os.path.exists(path+folder_):
        for filename in os.listdir(path+folder_):
            print ("Testing path: ", filename)
            print ("Testing patern: ", folder_[-1:] + "\d+")
            if re.match("_\d+", filename):
                folderNames.append(path+folder_ + filename + name_suffix)
                print ("***Found Folder Name: ", folderNames[-1])
    return folderNames
        
def plotMetaDataSimulation(data_path, settings, settingsFiles, folder=''):   
    
    """
    
    """
    
    from util.SimulationUtil import getDataDirectory, getBaseDataDirectory, getRootDataDirectory
    
    rlv = PolicyTrainVisualize("Policy Training Curves: " + str(settings['agent_name']), settings=settings)
    otherDatas = []
    # settingsFiles = sys.argv[2:]
    print("settingsFiles: ", settingsFiles)
    for settingsFile_ in settingsFiles:
        print ("Loading settings file: ", settingsFile_)
        settingsFile_ = open(settingsFile_, 'r')
        settings = json.load(settingsFile_)
        settingsFile_.close()
    
        folder_ = settings['data_folder']
        data_path = getRootDataDirectory(settings)+"/"
        folderNames_ = getDataFolderNames(data_path, folder_, settings)
        trainingDatas = []
        # Need to train a better Baseline
        for folderName in folderNames_:
            trainData={}
            trainData['fileName']=folderName
            trainData['name']= settings['data_folder']
            # trainData['colour'] = (1.0, 0.0, 0.0, 1.0)
            trainingDatas.append(trainData)
    
        otherDatas.append(trainingDatas)
    
    
    min_length = 1000000000
    for j in range(len(otherDatas)):
        tmp_min_length = 1000000000
        for i in range(len(otherDatas[j])):
            datafile = otherDatas[j][i]['fileName']
            file = open(datafile)
            otherDatas[j][i]['data'] = json.load(file)
            print ("Data file: ", file)
            print("Length of data: ", len(otherDatas[j][i]['data']["mean_eval"]))
            if min_length > (len(otherDatas[j][i]['data']["mean_eval"])):
                min_length = len(otherDatas[j][i]['data']["mean_eval"])
            if tmp_min_length > (len(otherDatas[j][i]['data']["mean_eval"])):
                tmp_min_length = len(otherDatas[j][i]['data']["mean_eval"])
            # print ("otherDatas[j][i]['data']: ", otherDatas[j][i]['data'])
            # print "Training data: " + str(trainingData)
            file.close()
        print ("Min length for ", otherDatas[j][0]['fileName'], " is ", tmp_min_length)
        
    # tmp_min_length = tmp_min_length + 1
    """
    trainData["mean_reward"]=[]
    trainData["std_reward"]=[]
    trainData["mean_bellman_error"]=[]
    trainData["std_bellman_error"]=[]
    trainData["mean_discount_error"]=[]
    trainData["std_discount_error"]=[]
    
    """
    subsample = 1
    rlv.setLength(min_length)
    # rlv.setLength(64)
    rlv.setBinSize(subsample)
    rlv.updateRewards(trainingDatas, otherDatas, mean_key="mean_reward", std_key="std_reward")
    rlv.init()
    rlv.saveVisual(folder+"Training_curves")
    # rlv.show()
    
    rlv_value_function = PolicyTrainVisualize("Value Function Training Curves: " + str(settings['agent_name']), 
                                              settings=settings, y_lable="Bellman Error")
    rlv_value_function.setLength(min_length)
    # rlv.setLength(64)
    rlv_value_function.setBinSize(subsample)
    rlv_value_function.updateRewards(trainingDatas, otherDatas, 
                                     mean_key="mean_discount_error", std_key="std_discount_error")
    rlv_value_function.init()
    rlv_value_function.saveVisual(folder+"Training_curves_discounted_error")
    # rlv.show()
    
    if (settings["train_forward_dynamics"] == True and False):
        rlv_fd = PolicyTrainVisualize("FD Training Curves: " + str(settings['agent_name']), 
                                      settings=settings, y_lable="Loss")
        rlv_fd.setLength(min_length)
        # rlv.setLength(64)
        rlv_fd.setBinSize(subsample)
        rlv_fd.updateRewards(trainingDatas, otherDatas, 
                                         mean_key="mean_forward_dynamics_loss", std_key="std_forward_dynamics_loss")
        rlv_fd.init()
        rlv_fd.saveVisual(folder+"Training_curves_fd")
        
        if ("train_reward_predictor" in settings and
            (settings["train_reward_predictor"] == True)):
            rlv_reward = PolicyTrainVisualize("Reward Training Curves: " + str(settings['agent_name']), 
                                              settings=settings, y_lable="Loss")
            rlv_reward.setLength(min_length)
            # rlv.setLength(64)
            rlv_reward.setBinSize(subsample)
            rlv_reward.updateRewards(trainingDatas, otherDatas, 
                                             mean_key="mean_forward_dynamics_reward_loss", std_key="std_forward_dynamics_reward_loss")
            rlv_reward.init()
            rlv_reward.saveVisual(folder+"Training_curves_reward")
        
    
    
    return (rlv)
