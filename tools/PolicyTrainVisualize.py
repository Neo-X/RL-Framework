import matplotlib.pyplot as plt
# from matplotlib import mpl
import numpy as np
# import matplotlib.animation as animation
import random
import sys
import json

class PolicyTrainVisualize(object):
    
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
        