import matplotlib.pyplot as plt
# from matplotlib import mpl
import numpy as np
# import matplotlib.animation as animation
import random
import sys
import json

def get_cmap(n, name='nipy_spectral'):
    
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.
    tab20, hsv and nipy_spectral are a good colour map as well'''
    return plt.cm.get_cmap(name, n)

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
        self._length = 0
        
    def setLength(self, length):
        self._length = length
        
        
    def init(self):
        """
            Three plots
            bellman error
            average reward
            discounted reward error
        """
        
        cmap = get_cmap(len(self._trainingDatas)+1)
        bin_size=5
        self._fig, (self._reward_ax) = plt.subplots(1, 1, sharey=False, sharex=True)
        self._fig_value, (self._value_ax) = plt.subplots(1, 1, sharey=False, sharex=True)
        for i in range(0, len(self._trainingDatas), 1):
            if ( (self._length) > 0 ):
                if ( (self._length) < (len(self._trainingDatas[i]['data']["mean_eval"]) ) ):
                    x_range = range(0, self._length, 1)
                else:
                    x_range = range(len(self._trainingDatas[i]['data']["mean_eval"]))
            else:
                x_range = range(len(self._trainingDatas[i]['data']["mean_eval"]))
            new_shape = (len(x_range)/bin_size, bin_size)
            new_length = new_shape[0]*new_shape[1]
            x_range_ = range(new_shape[0])
            # self._length = self._length/bin_size
            mean = np.mean(np.reshape(self._trainingDatas[i]['data']["mean_eval"][:new_length], new_shape), axis=1)
            std = np.mean(np.reshape(self._trainingDatas[i]['data']["std_eval"][:new_length], new_shape), axis=1)
            
            self._reward, = self._reward_ax.plot(x_range_, mean, 
                                                 linewidth=3.0, 
                                                 c=cmap(i),
                                                 label=self._trainingDatas[i]['name'])
            print("Line colour: ", self._reward.get_color())
            self._bellman_error_std = self._reward_ax.fill_between(x_range_, 
                                                                          np.array(mean) - std, 
                                                                          np.array(mean) + std,
                                                                          facecolor=self._reward.get_color(),
                                                                          alpha=0.25)
            
            mean_value = np.mean(np.reshape(self._trainingDatas[i]['data']["mean_discount_error"][:new_length], new_shape), axis=1)
            std_value = np.mean(np.reshape(self._trainingDatas[i]['data']["std_discount_error"][:new_length], new_shape), axis=1)
            self._value, = self._value_ax.plot(x_range_, mean_value, 
                                                 linewidth=3.0, 
                                                 c=cmap(i),
                                                 label=self._trainingDatas[i]['name'])
            print("Line colour: ", self._reward.get_color())
            self._discounted_error_std = self._value_ax.fill_between(x_range_, 
                                                                          np.array(mean_value) - std_value, 
                                                                          np.array(mean_value) + std_value,
                                                                          facecolor=self._reward.get_color(),
                                                                          alpha=0.25)
        # self._reward_std = self._reward_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
        leng = self._reward_ax.legend(loc="lower right",
                     ncol=1, shadow=True, fancybox=True)
        leng.get_frame().set_alpha(0.5)
        leng = self._value_ax.legend(loc="lower right",
                     ncol=1, shadow=True, fancybox=True)
        leng.get_frame().set_alpha(0.5)
        # self._reward_ax.set_title('Mean Reward')
        self._reward_ax.set_ylabel("Mean Reward")
        self._reward_ax.grid(b=True, which='major', color='black', linestyle='--')
        plt.xlabel("Iteration x" + str(self._iteration_scale))
        self._fig.suptitle(self._title, fontsize=18)
        
        self._value_ax.set_ylabel("Mean Reward")
        self._value_ax.grid(b=True, which='major', color='black', linestyle='--')
        plt.xlabel("Iteration x" + str(self._iteration_scale))
        self._fig_value.suptitle(self._title, fontsize=18)
        
        # plt.grid(b=True, which='major', color='black', linestyle='--')
        # plt.grid(b=True, which='minor', color='g', linestyle='--'
        
        self._fig.set_size_inches(8.0, 4.0, forward=True)
        self._fig_value.set_size_inches(8.0, 4.0, forward=True)
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
        self._fig_value.savefig(fileName+"_discounted_error.svg")
        self._fig_value.savefig(fileName+"_discounted_error.png")
        