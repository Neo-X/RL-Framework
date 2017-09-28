import matplotlib.pyplot as plt
# from matplotlib import mpl
import numpy as np
# import matplotlib.animation as animation
import random
import sys
import json
from PolicyTrainVisualize import PolicyTrainVisualize

        
if __name__ == "__main__":
    
    trainingDatas = []
    otherDatas = []
    
    # Need to train a better Baseline
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/nav_Game/algorithm.PPO.PPO/Nav_Sphere_10D/Deep_NN_TanH/trainingData_algorithm.PPO.PPO.json'
    trainData['name']='Baseline'
    trainData['colour'] = (1.0, 0.0, 0.0, 1.0)
    trainingDatas.append(trainData)
    
    # Need to train a better Baseline
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/nav_Game/algorithm.PPO.PPO/Nav_Sphere_10D/model.DeepNNTanH.DeepNNTanH/trainingData_algorithm.PPO.PPO.json'
    trainData['name']='Baseline'
    trainData['colour'] = (1.0, 0.0, 0.0, 1.0)
    trainingDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/nav_Game/algorithm.PPO.PPO/Nav_Sphere_10D/Deep_NN_TanH_2/trainingData_algorithm.PPO.PPO.json'
    trainData['name']='Baseline'
    trainData['colour'] = (1.0, 0.0, 0.0, 1.0)
    trainingDatas.append(trainData)
  
    """ 
    # Final method
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/nav_Game/algorithm.PPO.PPO/Nav_Sphere_MBAE_10D/Deep_NN_TanH/trainingData_algorithm.PPO.PPO.json'
    trainData['name']='PPO + MBAE'
    # trainData['colour'] = (0.0, 0.0, 1.0, 1.0)
    trainingDatas.append(trainData)
    
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/nav_Game/PPO/Nav_Sphere_MBAE_FULL_10D/Deep_NN_TanH/trainingData_PPO.json'
    trainData['name']='PPO + MBAE + pre training'
    # trainData['colour'] = (0.0, 1.0, 0.0, 1.0)
    trainingDatas.append(trainData)
    """
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/nav_Game/PPO/Nav_Sphere_SMBAE_FULL_10D/Deep_NN_TanH/trainingData_PPO.json'
    trainData['name']='PPO + MBAE'
    trainData['colour'] = (0.0, 0.0, 1.0, 1.0)
    otherDatas.append(trainData)
    
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/nav_Game/PPO/Nav_Sphere_SMBAE_FULL_10D_pretained/Deep_NN_TanH/trainingData_PPO.json'
    trainData['name']='PPO + MBAE + pre-training'
    trainData['colour'] = (0.0, 0.0, 1.0, 1.0)
    otherDatas.append(trainData)
    
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/nav_Game/PPO/Nav_Sphere_SMBAE_FULL_10D_MORE_SMBAE/Deep_NN_TanH/trainingData_PPO.json'
    trainData['name']='PPO + MBAE More + pre-training'
    trainData['colour'] = (0.0, 0.0, 1.0, 1.0)
    otherDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/nav_Game/PPO/Nav_Sphere_SMBAE_10D/Deep_NN_TanH/trainingData_PPO.json'
    trainData['name']='PPO + MBAE'
    # trainData['colour'] = (1.0, 1.0, 0.0, 1.0)
    otherDatas.append(trainData)
    
    for i in range(len(trainingDatas)):
        datafile = trainingDatas[i]['fileName']
        file = open(datafile)
        trainingDatas[i]['data'] = json.load(file)
        # print "Training data: " + str(trainingData)
        file.close()
        
    for i in range(len(otherDatas)):
        datafile = otherDatas[i]['fileName']
        file = open(datafile)
        otherDatas[i]['data'] = json.load(file)
        # print "Training data: " + str(trainingData)
        file.close()
    
    if (len(sys.argv) == 3):
        length = int(sys.argv[2])
    
    """
    trainData["mean_reward"]=[]
    trainData["std_reward"]=[]
    trainData["mean_bellman_error"]=[]
    trainData["std_bellman_error"]=[]
    trainData["mean_discount_error"]=[]
    trainData["std_discount_error"]=[]
    
    """
    settings = None
    if (len(sys.argv) >= 2):
        settingsFileName = sys.argv[1]
        settingsFile = open(settingsFileName, 'r')
        settings = json.load(settingsFile)
        settingsFile.close()
    rlv = PolicyTrainVisualize("Training Curves", settings=settings)
    if (len(sys.argv) == 3):
        length = int(sys.argv[2])
        rlv.setLength(length)
    rlv.updateRewards(trainingDatas, otherDatas)
    rlv.init()
    rlv.saveVisual("MBAE_Training_curves")
    rlv.show()