import matplotlib.pyplot as plt
# from matplotlib import mpl
import numpy as np
# import matplotlib.animation as animation
import random
import sys
import json

class RLVisualize(object):
    
    def __init__(self, title, settings=None):
        """
            Three plots
            bellman error
            average reward
            discounted reward error
        """
        if (settings != None):
            self._iteration_scale = ((settings['plotting_update_freq_num_rounds']*settings['max_epoch_length']*settings['epochs'] * 
                                      settings['training_updates_per_sim_action']) / 
                                     settings['sim_action_per_training_update'])
        else:
            self._iteration_scale = 1
        self._title=title
        
        
    def init(self):
        """
            Three plots
            bellman error
            average reward
            discounted reward error
        """
        self._fig, (self._reward_ax) = plt.subplots(1, 1, sharey=False, sharex=True)
        for i in range(len(self._trainingDatas)):
            self._reward, = self._reward_ax.plot(range(len(self._trainingDatas[i]['data']["mean_eval"])), self._trainingDatas[i]['data']["mean_eval"], 
                                                 linewidth=3.0, label=self._trainingDatas[i]['name'])
        # self._reward_std = self._reward_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
        self._reward_ax.legend(loc="lower right",
                     ncol=1, shadow=True, fancybox=True)
        # self._reward_ax.set_title('Mean Reward')
        self._reward_ax.set_ylabel("Mean Reward")
        self._reward_ax.grid(b=True, which='major', color='black', linestyle='--')
        plt.xlabel("Iteration x" + str(self._iteration_scale))
        self._fig.suptitle(self._title, fontsize=18)
        
        # plt.grid(b=True, which='major', color='black', linestyle='--')
        # plt.grid(b=True, which='minor', color='g', linestyle='--'
        
        self._fig.set_size_inches(8.0, 4.0, forward=True)
        plt.show()
        
    def updateRewards(self, trainingDatas):
        self._trainingDatas = trainingDatas
       
        
    def show(self):
        plt.show()
        
    def redraw(self):
        self._fig.canvas.draw()
        
    def setInteractive(self):
        plt.ion()
        
    def setInteractiveOff(self):
        plt.ioff()
        
    def saveVisual(self, fileName):
        self._fig.savefig(fileName+".svg")
        self._fig.savefig(fileName+".png")
        
if __name__ == "__main__":
    
    trainingDatas = []
    
    """
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
    
    """
    # Need to train just Dyna
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/simbiconBiped2D/A3C/Simple_Walk_FD/Deep_NN/trainingData_A3C.json'
    trainData['name']='+ proixmal point regularization'
    trainingDatas.append(trainData)
    """
    
    # Final method
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/simbiconBiped2D/A3C/Simple_Walk_FD2/Deep_NN/trainingData_A3C.json'
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
    
    rlv = RLVisualize("Training Curves Different Combinations of Methods")
    rlv.updateRewards(trainingDatas)
    rlv.init()
    rlv.saveVisual("agent")
    rlv.show()