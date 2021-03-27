## Don't use Xwindows backend for this
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from matplotlib import mpl
import sys
import json

def get_cmap(n, name='nipy_spectral'):
    
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.
    tab20, hsv and nipy_spectral are a good colour map as well'''
    return plt.cm.get_cmap(name, n)

class PolicyTrainVisualize(object):
    
    def __init__(self, title, settings=None, y_lable="Mean Reward"):
        """
            Three plots
            bellman error
            average reward
            discounted reward error
        """
        self._settings = settings
        self._title=title
        self._length = 0
        self._bin_size = 1
        self._key_ = "mean_reward"
        self._key_std_ = "std_eval"
        self._y_label = y_lable
        
        
    def setLength(self, length):
        self._length = length
    
    def setBinSize(self, bin_size_):
        self._bin_size = bin_size_
        
        
    def init(self):
        """
            Three plots
            bellman error
            average reward
            discounted reward error
        """
        import numpy as np
        
        if (self._settings != None):
            self._sim_iteration_scale = (self._settings['plotting_update_freq_num_rounds']*
                                         self._settings['max_epoch_length']*
                                         self._settings['epochs']) * self._bin_size
            self._iteration_scale = ((self._sim_iteration_scale * 
                                      self._settings['training_updates_per_sim_action']) / 
                                     self._settings['sim_action_per_training_update']) * self._bin_size
            
            if ('on_policy' in self._settings and (self._settings['on_policy'])):
                self._sim_iteration_scale = self._sim_iteration_scale * self._settings['num_on_policy_rollouts']
                self._iteration_scale = ((self._sim_iteration_scale / self._settings['max_epoch_length']) *
                                     self._settings['critic_updates_per_actor_update'])
            
        else:
            self._iteration_scale = 1 * self._bin_size
            self._sim_iteration_scale = 1 * self._bin_size
        
        if ( self._otherDatas == None): 
            cmap = get_cmap(len(self._trainingDatas)+1)
            self._fig, (self._reward_ax) = plt.subplots(1, 1, sharey=False, sharex=True)
            self._fig_value, (self._value_ax) = plt.subplots(1, 1, sharey=False, sharex=True)
            for i in range(0, len(self._trainingDatas), 1):
                if ( (self._length) > 0 and (j > 0) ):
                    if ( (self._length) < (len(self._trainingDatas[i]['data'][self._key_]) ) ):
                        x_range = range(0, self._length, 1)
                        print ("x_range: ", x_range)
                    else:
                        x_range = range(len(self._trainingDatas[i]['data'][self._key_]))
                else:
                    x_range = range(len(self._trainingDatas[i]['data'][self._key_]))
                new_shape = (int(len(x_range)/self._bin_size), int(self._bin_size))
                new_length = new_shape[0]*new_shape[1]
                x_range_ = range(int(new_shape[0]))
                # self._length = self._length/self._bin_size
                mean = np.reshape(self._trainingDatas[i]['data'][self._key_][:new_length], new_shape)
                # print ("mean: ", mean)
                mean = np.mean(np.reshape(self._trainingDatas[i]['data'][self._key_][:new_length], new_shape), axis=1)
                std = np.mean(np.reshape(self._trainingDatas[i]['data'][self._key_std_][:new_length], new_shape), axis=1)
                
                colour_ = cmap(i)
                if ('colour' in self._trainingDatas[i]):
                    colour_ = self._trainingDatas[i]['colour']
                self._reward, = self._reward_ax.plot(x_range_, mean, 
                                                     linewidth=3.0, 
                                                     c=colour_,
                                                     label=(self._trainingDatas[i]['name'])[64:])
                # print("Line colour: ", self._reward.get_color())
                self._bellman_error_std = self._reward_ax.fill_between(x_range_, 
                                                                              np.array(mean) - std, 
                                                                              np.array(mean) + std,
                                                                              facecolor=self._reward.get_color(),
                                                                              alpha=0.25)
                """
                mean_value = np.mean(np.reshape(self._trainingDatas[i]['data']["mean_discount_error"][:new_length], new_shape), axis=1)
                std_value = np.mean(np.reshape(self._trainingDatas[i]['data']["std_discount_error"][:new_length], new_shape), axis=1)
                self._value, = self._value_ax.plot(x_range_, mean_value, 
                                                     linewidth=3.0, 
                                                     c=colour_,
                                                     alpha=0.75,
                                                     label=(self._trainingDatas[i]['name'])[64:])
                print("Line colour: ", self._reward.get_color())
                self._discounted_error_std = self._value_ax.fill_between(x_range_, 
                                                                              np.array(mean_value) - std_value, 
                                                                              np.array(mean_value) + std_value,
                                                                              facecolor=self._reward.get_color(),
                                                                              alpha=0.25)
                """
                
            # self._reward_std = self._reward_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
            leng = self._reward_ax.legend(loc="lower right",
                         ncol=1, shadow=True, fancybox=True)
            leng.get_frame().set_alpha(0.3)
            """
            leng = self._value_ax.legend(loc="lower right",
                         ncol=1, shadow=True, fancybox=True)
            leng.get_frame().set_alpha(0.3)
            """
            # self._reward_ax.set_title('Mean Reward')
            self._reward_ax.set_ylabel(self._y_label, fontsize=26)
            self._reward_ax.grid(b=True, which='major', color='black', linestyle='--')
            plt.xlabel("Iteration x" + str(self._iteration_scale), fontsize=24)
            self._reward_ax.tick_params(axis='x', labelsize=22)
            self._reward_ax.tick_params(axis='y', labelsize=22)
            self._fig.suptitle(self._title, fontsize=22)
            self._reward_ax.set_xlabel("Simulated Actions x" + str(self._sim_iteration_scale), fontsize=26)
            # plt.grid(b=True, which='major', color='black', linestyle='--')
            # plt.grid(b=True, which='minor', color='g', linestyle='--'
            
            self._fig.set_size_inches(11.0, 6.0, forward=True)
            # self._fig_value.set_size_inches(11.0, 6.0, forward=True)
            # plt.show()
        else:
            means_ = []
            # mean_values_ = []
            cmap = get_cmap(len(self._otherDatas)+1)
            self._fig, (self._reward_ax) = plt.subplots(1, 1, sharey=False, sharex=True)
            self._fig_value, (self._value_ax) = plt.subplots(1, 1, sharey=False, sharex=True)
            for j in range(len(self._otherDatas)):
                for i in range(0, len(self._otherDatas[j]), 1):
                    """
                    if (j > 0):
                        self._length = 40
                    """
                    if ( (self._length) > 0 ): ## potentially reduce the range of data plotted
                        if ( (self._length) < (len(self._otherDatas[j][i]['data'][self._key_]) ) ):
                            x_range = range(0, self._length, 1)
                        else:
                            x_range = range(len(self._otherDatas[j][i]['data'][self._key_]))
                    else:
                        x_range = range(len(self._otherDatas[j][i]['data'][self._key_]))
                    new_shape = (int(len(x_range)/self._bin_size), int(self._bin_size))
                    new_length = new_shape[0]*new_shape[1]
                    x_range_ = list(range(int(new_shape[0])))
                    # self._length = self._length/self._bin_size
                    # print (self._otherDatas[j][i]['data'][self._key_][:new_length])
                    # print (np.mean(self._otherDatas[j][i]['data'][self._key_][:new_length], axis=-1))
                    """
                    if (len(np.array(self._otherDatas[j][i]['data'][self._key_][:new_length]).shape) > 1 ):
                        ### the case for multi-agent learning
                        mean = np.mean(self._otherDatas[j][i]['data'][self._key_][:new_length], axis=-1)
                        std = np.mean(self._otherDatas[j][i]['data'][self._key_std_][:new_length], axis=-1)
                    else:
                """
                    mean = self._otherDatas[j][i]['data'][self._key_][:new_length]
                    std = self._otherDatas[j][i]['data'][self._key_std_][:new_length]
                    if (np.all(np.isfinite(mean))):
                        means_.append(mean)
                        # mean_values_.append(mean_value)
                
                # print("means_: ", means_)
                """          
                if (j == 3):
                    for i in range(len(means_)):
                        ### Add the last item of the PLAiD sim to the first of the Distillation data.
                        if ( (self._length) > 0 ): ## potentially reduce the range of data plotted
                            if ( (self._length) < (len(self._otherDatas[2][i]['data'][self._key_]) ) ):
                                x_range = range(0, self._length, 1)
                            else:
                                x_range = range(len(self._otherDatas[2][i]['data'][self._key_]))
                        else:
                            x_range = range(len(self._otherDatas[2][i]['data'][self._key_]))
                        new_shape = (int(len(x_range)/self._bin_size), int(self._bin_size))
                        new_length = new_shape[0]*new_shape[1]
                        print ("Shape of data: ", len(self._otherDatas[2][i]['data'][self._key_]))
                        mean = np.mean(np.reshape(self._otherDatas[2][i]['data'][self._key_][:new_length], new_shape), axis=1)
                        mean_value = np.mean(np.reshape(self._otherDatas[2][i]['data']["mean_discount_error"][:new_length], new_shape), axis=1)
                        print ("mean: ", mean)
                        print ("last mean: ", mean[-1])
                        means_[i] = np.insert(means_[i], 0,mean[39])
                        mean_values_[i] = np.insert(mean_values_[i], 0,mean_value[39])
                    x_range_ = list(range(39,39+int(means_[i].shape[0])))
                """
                # print ("means_: ", means_)
                # print ("x_range_:", x_range_)
                mean = np.mean(means_, axis=0)
                std = np.std(means_, axis=0)
                # print ("mean: ", type(mean[0]), mean)
                # mean_value = np.mean(mean_values_, axis=0)
                # std_value = np.std(mean_values_, axis=0)
                colour_ = cmap(j)
                
                if ('colour' in self._otherDatas[j][i]):
                    colour_ = self._otherDatas[j][i]['colour']
                    
                """
                ### Plot individual runs
                for m in means_:
                    print ("m: ", m)
                    self._reward, = self._reward_ax.plot(range(len(m)), m, 
                                                     linewidth=2.0, 
                                                     alpha=0.5,
                                                     c=colour_,
                                                     linestyle='--')
                """
                markers=[ 'o' , '^' , 's' , '+', ',', '.', '1' , '2' , '3' , '4' ]
                if (type(mean[0]) in (list, np.ndarray)):
                    print ("Multi-agent learning, plotting agents individually.")
                    for k in range(len(mean[0])):
                        # print ("agent mean: ", np.transpose(mean)[k].shape)
                        ### average wrt bin size
                        # np.transpose(mean)[k]
                        split_indices = [i for i  in range(self._bin_size, np.transpose(mean)[k].shape[0], self._bin_size) ]# math.floor(a.shape[axis] / chunk_shape[axis]))]
                        first_split = np.array_split(np.transpose(mean)[k], split_indices, axis=0)
                        vals__mean =  np.array([[np.mean(rs)] for rs in first_split]).flatten()
                        first_split = np.array_split(np.transpose(std)[k], split_indices, axis=0)
                        vals__std =  np.array([[np.mean(rs)] for rs in first_split]).flatten()
                        self._reward, = self._reward_ax.plot(x_range_, vals__mean, 
                                                     linewidth=3.0, 
                                                     c=colour_,
                                                     label=(self._otherDatas[j][i]['name'] + " agent: " + str(k) + " samples: " + str(len(means_))),
                                                     marker=markers[k])
                        # print("Line colour: ", self._reward.get_color())
                        self._bellman_error_std = self._reward_ax.fill_between(x_range_, 
                                                                              vals__mean - vals__std, 
                                                                              vals__mean + vals__std,
                                                                              facecolor=self._reward.get_color(),
                                                                              alpha=0.25)
                else:
                    split_indices = [i for i  in range(self._bin_size, len(mean), self._bin_size) ]# math.floor(a.shape[axis] / chunk_shape[axis]))]
                    first_split = np.array_split(mean, split_indices, axis=0)
                    vals__mean =  np.array([[np.mean(rs)] for rs in first_split]).flatten()
                    first_split = np.array_split(std, split_indices, axis=0)
                    vals__std =  np.array([[np.mean(rs)] for rs in first_split]).flatten()
                    self._reward, = self._reward_ax.plot(x_range_, vals__mean, 
                                                     linewidth=3.0, 
                                                     c=colour_,
                                                     label=(self._otherDatas[j][i]['name'] + " samples: " + str(len(means_))))
                    # print("Line colour: ", self._reward.get_color())
                    self._bellman_error_std = self._reward_ax.fill_between(x_range_, 
                                                                                  vals__mean - vals__std, 
                                                                                  vals__mean + vals__std,
                                                                                  facecolor=self._reward.get_color(),
                                                                                  alpha=0.25)
                """
                for v in mean_values_:
                    self._value, = self._value_ax.plot(x_range_, v, 
                                                     linewidth=2.0, 
                                                     c=colour_,
                                                     alpha=0.5,
                                                     linestyle='--')
                self._value, = self._value_ax.plot(x_range_, mean_value, 
                                                     linewidth=3.0, 
                                                     c=colour_,
                                                     label=(self._otherDatas[j][i]['name'] + " samples: " + str(len(means_)))[64:])
                print("Line colour: ", self._reward.get_color())
                self._discounted_error_std = self._value_ax.fill_between(x_range_, 
                                                                              np.array(mean_value) - std_value, 
                                                                              np.array(mean_value) + std_value,
                                                                              facecolor=self._reward.get_color(),
                                                                              alpha=0.25)
                """
                means_ = []
                # mean_values_ = []
                
            
            # self._reward_std = self._reward_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
            leng = self._reward_ax.legend(loc="lower right",
                         ncol=1, shadow=True, fancybox=True)
            leng.get_frame().set_alpha(0.3)
            leng = self._value_ax.legend(loc="lower right",
                         ncol=1, shadow=True, fancybox=True)
            leng.get_frame().set_alpha(0.3)
            # self._reward_ax.set_title('Mean Reward')
            self._reward_ax.set_ylabel(self._y_label, fontsize=26)
            self._reward_ax.grid(b=True, which='major', color='black', linestyle='--')
            plt.xlabel("Iteration x" + str(self._iteration_scale), fontsize=24)
            self._reward_ax.tick_params(axis='x', labelsize=22)
            self._reward_ax.tick_params(axis='y', labelsize=22)
            self._fig.suptitle(self._title, fontsize=22)
            self._reward_ax.set_xlabel("Simulated Actions x" + str(self._sim_iteration_scale), fontsize=26)
            # plt.grid(b=True, which='major', color='black', linestyle='--')
            # plt.grid(b=True, which='minor', color='g', linestyle='--'
            
            self._fig.set_size_inches(11.0, 6.0, forward=True)
            # self._fig_value.set_size_inches(11.0, 6.0, forward=True)
            # plt.show()
            
    def updateRewards(self, trainingDatas, otherDatas=None, mean_key="mean_eval", std_key="std_eval"):
        self._trainingDatas = trainingDatas
        self._otherDatas = otherDatas
        self._key_ = mean_key
        self._key_std_ = std_key
        
       
        
    def show(self):
        plt.tight_layout()
        plt.show()
        
    def redraw(self):
        self._fig.canvas.draw()
        
    """
    def setInteractive(self):
        plt.ion()
        
    def setInteractiveOff(self):
        plt.ioff()
    """ 
    def saveVisual(self, fileName):
        self._fig.savefig(fileName+".svg", bbox_inches = 'tight',
    pad_inches = 0)
        self._fig.savefig(fileName+".png", bbox_inches = 'tight',
    pad_inches = 0)
        # self._fig_value.savefig(fileName+"_discounted_error.svg")
        # self._fig_value.savefig(fileName+"_discounted_error.png")
        