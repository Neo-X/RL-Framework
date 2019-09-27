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
        self._settings = settings
        self._title = title
        self._agents = 1
        if ("perform_multiagent_training" in self._settings):
            ### The + 0.1 is an ugly hack to get single agent working
            self._agents = self._settings["perform_multiagent_training"] + 0.1
        
        
    def init(self):
        if (self._settings != None):
            self._sim_iteration_scale = (self._settings['plotting_update_freq_num_rounds']*self._settings['max_epoch_length']*self._settings['epochs'])
            self._iteration_scale = ((self._sim_iteration_scale * self._settings['training_updates_per_sim_action']) / 
                                     self._settings['sim_action_per_training_update'])
            if ('on_policy' in self._settings and (self._settings['on_policy'])):
                self._sim_iteration_scale = self._sim_iteration_scale * self._settings['num_on_policy_rollouts']
                self._iteration_scale = ((self._sim_iteration_scale / (self._settings['max_epoch_length'] )) *
                                     self._settings['critic_updates_per_actor_update'])
        else:
            self._iteration_scale = 1
            self._sim_iteration_scale = 1
        
        self._fig, (self._bellman_error_ax, self._reward_ax, self._discount_error_ax) = plt.subplots(3, 1, sharey=False, sharex=True)
        self._bellman_errors = []
        self._bellman_error_stds = []
        self._rewards = []
        self._reward_stds = []
        self._discount_errors = []
        self._discount_error_stds = []
        for j in range(int(self._agents)):
            if ("agent_names" in self._settings):
                self._bellman_error, = self._bellman_error_ax.plot([], [], linewidth=2.0, label=self._settings["agent_names"][j])
            else:
                self._bellman_error, = self._bellman_error_ax.plot([], [], linewidth=2.0, label="agent_"+str(j))
            self._bellman_error_std = self._bellman_error_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
            self._bellman_error_ax.set_title('Bellman Error', fontsize=16)
            self._bellman_error_ax.set_ylabel("Absolute Error", fontsize=16)
            self._bellman_error_ax.grid(b=True, which='major', color='black', linestyle='--')
            leng = self._bellman_error_ax.legend(loc="upper right",
                         ncol=1, shadow=True, fancybox=True)
            leng.get_frame().set_alpha(0.3)
            self._reward, = self._reward_ax.plot([], [], linewidth=2.0)
            self._reward_std = self._reward_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
            self._reward_ax.set_title('Mean Reward', fontsize=16)
            self._reward_ax.set_ylabel("Reward", fontsize=16)
            self._reward_ax.grid(b=True, which='major', color='black', linestyle='--')
            self._discount_error, = self._discount_error_ax.plot([], [], linewidth=2.0)
            self._discount_error_std = self._discount_error_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
            self._discount_error_ax.set_title('Discount Error', fontsize=16)
            self._discount_error_ax.set_ylabel("Absolute Error", fontsize=16)
            self._discount_error_ax.grid(b=True, which='major', color='black', linestyle='--')
            self._bellman_errors.append(self._bellman_error)
            self._bellman_error_stds.append(self._bellman_error_std)
            self._rewards.append(self._reward)
            self._reward_stds.append(self._reward_std)
            self._discount_errors.append(self._discount_error)
            self._discount_error_stds.append(self._discount_error_std)
            
        plt.xlabel("Simulated Actions x" + str(self._sim_iteration_scale) + ", Training Updates x" + str(self._iteration_scale), fontsize=16)
        self._fig.set_size_inches(8.0, 12.5, forward=True)
        
    def updateBellmanError(self, error, std):
        
        # self._bellman_error.set_data(error)
        if ( self._agents > 1):
            
            for j in range(int(self._agents)):
                self._bellman_errors[j].set_data(np.arange(len(error[:,j])), error[:,j])
                self._bellman_error_ax.collections.remove(self._bellman_error_stds[j])
                # print ("self._bellman_errors[j]: ", self._bellman_errors[j])
                self._bellman_error_stds[j] = self._bellman_error_ax.fill_between(np.arange(len(error[:,j])), error[:,j] - std[:,j], error[:,j] + std[:,j], facecolor=self._bellman_errors[j].get_color(), alpha=0.4)
        else:
            self._bellman_error.set_data(np.arange(len(error)), error)
            # self._bellman_error.set_data(error)
            self._bellman_error_ax.collections.remove(self._bellman_error_std)
            self._bellman_error_std = self._bellman_error_ax.fill_between(np.arange(len(error)), error - std, error + std, facecolor=self._bellman_error.get_color(), alpha=0.4)
        
        self._bellman_error_ax.relim()      # make sure all the data fits
        self._bellman_error_ax.autoscale()
        
    def updateReward(self, reward, std):
        
        if ( self._agents > 1):
            for j in range(int(self._agents)):
                self._rewards[j].set_data(np.arange(len(reward[:,j])), reward[:,j])
                # self._rewards[j].set_ydata(reward[:,j])
                self._reward_ax.collections.remove(self._reward_stds[j])
                self._reward_stds[j] = self._reward_ax.fill_between(np.arange(len(reward[:,j])), reward[:,j] - std[:,j], reward[:,j] + std[:,j], facecolor=self._rewards[j].get_color(), alpha=0.4)
        else:
            for j in range(int(self._agents)):
                self._rewards[j].set_data(np.arange(len(reward)), reward)
                # self._rewards[j].set_ydata(reward)
                self._reward_ax.collections.remove(self._reward_stds[j])
                self._reward_stds[j] = self._reward_ax.fill_between(np.arange(len(reward)), reward - std, reward + std, facecolor=self._rewards[j].get_color(), alpha=0.4)
            
        self._reward_ax.relim()      # make sure all the data fits
        self._reward_ax.autoscale()  # auto-scale
        
    def updateDiscountError(self, error, std):
       
        if ( self._agents > 1):
            for j in range(int(self._agents)):
                self._discount_errors[j].set_data(np.arange(len(error[:,j])), error[:,j] )
                # self._discount_errors[j].set_ydata(error[:,j])
                self._discount_error_ax.collections.remove(self._discount_error_stds[j])
                self._discount_error_stds[j] = self._discount_error_ax.fill_between(np.arange(len(error[:,j])), error[:,j] - std[:,j], error[:,j] + std[:,j], facecolor=self._discount_errors[j].get_color(), alpha=0.4)
        else:
            for j in range(int(self._agents)):
                self._discount_errors[j].set_data(np.arange(len(error)), error )
                # self._discount_errors[j].set_ydata(error)
                self._discount_error_ax.collections.remove(self._discount_error_stds[j])
                self._discount_error_stds[j] = self._discount_error_ax.fill_between(np.arange(len(error)), error - std, error + std, facecolor=self._discount_errors[j].get_color(), alpha=0.4)
        
        
        self._discount_error_ax.relim()      # make sure all the data fits
        self._discount_error_ax.autoscale()
        
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
        
    def finish(self):
        """
            Closes the figure window
        """
        plt.close(self._fig)
        plt.close()
        
        
if __name__ == "__main__":
    
    datafile = sys.argv[1]
    file = open(datafile)
    trainData = json.load(file)
    # print "Training data: " + str(trainingData)
    file.close()
    settings = None
    length = len(trainData["mean_bellman_error"])
    if (len(sys.argv) == 3):
        datafile = sys.argv[2]
        file = open(datafile)   
        settings = json.load(file)
        file.close()
        
    
    """
    trainData["mean_reward"]=[]
    trainData["std_reward"]=[]
    trainData["mean_bellman_error"]=[]
    trainData["std_bellman_error"]=[]
    trainData["mean_discount_error"]=[]
    trainData["std_discount_error"]=[]
    
    """
    
    rlv = RLVisualize(datafile, settings)
    rlv.updateBellmanError(np.array(trainData["mean_bellman_error"][:length]), np.array(trainData["std_bellman_error"][:length]))
    rlv.updateReward(np.array(trainData["mean_eval"][:length]), np.array(trainData["std_eval"][:length]))
    rlv.updateDiscountError(np.fabs(trainData["mean_discount_error"][:length]), np.array(trainData["std_discount_error"][:length]))
    rlv.saveVisual("pendulum_agent")
    rlv.show()