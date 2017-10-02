import matplotlib.pyplot as plt
# from matplotlib import mpl
import numpy as np
# import matplotlib.animation as animation
import random
import sys
import json
from PolicyTrainVisualize import PolicyTrainVisualize
import os
import re

"""
    plot_meta_simulation.py <settings_file_name> <path_to_data>
    Example:
"""

def getDataFolderNames(prefixPath, folderPrefix, settings):
    path = prefixPath
    folder_ = folderPrefix
    name_suffix = "/"+settings["model_type"]+"/"+"trainingData_" + str(settings['agent_name']) + ".json"
    folderNames = []
    print ("path: ", path)
    print ("folder_: ", folder_)
    print ("name_suffix: ", name_suffix)
    for filename in os.listdir(path):
        if re.match(folder_ + "\d+", filename):
            folderNames.append(prefixPath + filename + name_suffix)
            print ("Folder Name: ", prefixPath + filename + name_suffix)
    ## For the case where I put a slash at the end of the path name...
    for filename in os.listdir(path + folder_[:-1]):
        if re.match(folder_ + "\d+", filename):
            folderNames.append(prefixPath + filename + name_suffix)
            print ("Folder Name: ", prefixPath + filename + name_suffix)
    return folderNames
        
if __name__ == "__main__":
    
    settings = None
    if (len(sys.argv) > 2 ):
        settingsFileName = sys.argv[2]
        settingsFile = open(settingsFileName, 'r')
        settings = json.load(settingsFile)
        settingsFile.close()
        rlv = PolicyTrainVisualize("Training Curves", settings=settings)
        path = sys.argv[1]
        # length = int(sys.argv[2])
        # rlv.setLength(length)
    else:
        print ("Please specify arguments properly")
        print ("python plot_meta_simulation.py <path_to_data> <settings_file_name> <settings_file_name> ...")
        sys.exit()
    
    
    otherDatas = []
    
    settingsFiles = ['settings/particleSim/A_CACLA_10D.json', 
                     'settings/particleSim/A_CACLA_MBAE_10D.json', 
                     'settings/particleSim/A_CACLA_SMBAE_10D.json',
                     'settings/particleSim/A_CACLA_MBAE_10D.json']
    """
    settingsFiles = ['settings/particleSim/PPO_10D.json', 
                     'settings/particleSim/PPO_MBAE_10D.json', 
                     'settings/particleSim/PPO_SMBAE_10D.json',
		     'settings/particleSim/PPO_Dropout_SMBAE_10D.json']

    """
    settingsFiles = sys.argv[2:]
    for settingsFile_ in settingsFiles:
        print ("Loading settings file: ", settingsFile_)
        settingsFile_ = open(settingsFile_, 'r')
        settings = json.load(settingsFile_)
        settingsFile_.close()
    
        folder_ = settings['data_folder'] + "_"
        folderNames_ = getDataFolderNames(path, folder_, settings)
        trainingDatas = []
        # Need to train a better Baseline
        for folderName in folderNames_:
            trainData={}
            trainData['fileName']=folderName
            trainData['name']= settings['data_folder']
            # trainData['colour'] = (1.0, 0.0, 0.0, 1.0)
            trainingDatas.append(trainData)
    
        otherDatas.append(trainingDatas)
    
    
    
    for j in range(len(otherDatas)):
        for i in range(len(trainingDatas)):
            datafile = otherDatas[j][i]['fileName']
            file = open(datafile)
            otherDatas[j][i]['data'] = json.load(file)
            # print "Training data: " + str(trainingData)
            file.close()
        
    
    """
    trainData["mean_reward"]=[]
    trainData["std_reward"]=[]
    trainData["mean_bellman_error"]=[]
    trainData["std_bellman_error"]=[]
    trainData["mean_discount_error"]=[]
    trainData["std_discount_error"]=[]
    
    """
    rlv.updateRewards(trainingDatas, otherDatas)
    rlv.init()
    rlv.saveVisual("MBAE_Training_curves")
    rlv.show()
