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
    
    # Need to train a better Baseline
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/gapgame_2d/A_CACLA/Gaps_Sphere_perfect/Deep_CNN/trainingData_A_CACLA.json'
    trainData['name']='Baseline'
    trainingDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/gapgame_2d/A_CACLA/Gaps_Sphere/Deep_CNN_Dropout/trainingData_A_CACLA.json'
    trainData['name']='Baseline + Dropout'
    trainingDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/gapgame_2d/A_CACLA/Gaps_Sphere_MBAE/Deep_CNN/trainingData_A_CACLA.json'
    trainData['name']='Baseline + MBAE'
    trainingDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/gapgame_2d/A_CACLA/Gaps_Sphere_OnPolicy_2/Deep_CNN/trainingData_A_CACLA.json'
    trainData['name']='Baseline + OnPolicy'
    trainingDatas.append(trainData)
    
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/gapgame_2d/A_CACLA/Gaps_Sphere_ProximalREg/Deep_CNN/trainingData_A_CACLA.json'
    trainData['name']='Baseline + ProximalRegularization'
    trainingDatas.append(trainData)
    
    
    for i in range(len(trainingDatas)):
        datafile = trainingDatas[i]['fileName']
        file = open(datafile)
        trainingDatas[i]['data'] = json.load(file)
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
    
    rlv = PolicyTrainVisualize("Training Curves")
    rlv.updateRewards(trainingDatas)
    rlv.init()
    rlv.saveVisual("GapGame2D_Training_curves")
    rlv.show()