import matplotlib.pyplot as plt
# from matplotlib import mpl
import numpy as np
# import matplotlib.animation as animation
import random
import sys
import json

        
if __name__ == "__main__":
    
    trainingDatas = []
    
    # Need to train a better Baseline
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/simbiconBiped2D/A3C/Simple_Walk/Deep_NN/trainingData_A3C.json'
    trainData['name']='Baseline'
    trainingDatas.append(trainData)
    """
    # Temporary baseline
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/simbiconBiped2D/A3C/Simple_Walk_Parameterized/Deep_NN_SingleNet_Dropout/trainingData_A3C.json'
    trainData['name']='Baseline'
    trainingDatas.append(trainData)
    
    # Baseline + Model-based Action Exploration
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/simbiconBiped2D/A3C/Simple_Walk_FD/Deep_NN/trainingData_A3C.json'
    trainData['name']='Baseline + MBAE'
    trainingDatas.append(trainData)
    
    # Baseline + MDAE and regularization
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/simbiconBiped2D/A3C/Simple_Walk_FD2_No_critic_train_on_fd/Deep_NN/trainingData_A3C.json'
    trainData['name']='Baseline + MBAE + regularization'
    trainingDatas.append(trainData)
    
    
    # Need to train just Dyna
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/simbiconBiped2D/A3C/Simple_Walk_FD/Deep_NN/trainingData_A3C.json'
    trainData['name']='+ proixmal point regularization'
    trainingDatas.append(trainData)
    """
    
    # Final method
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/simbiconBiped2D/A3C/Simple_Walk_FD_Dyna_Reg/Deep_NN/trainingData_A3C.json'
    trainData['name']='Baseline + MBAE + regularization + Dyna'
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
    rlv.saveVisual("agent")
    rlv.show()