import matplotlib.pyplot as plt
import imageio
# from matplotlib import mpl
import numpy as np
# import matplotlib.animation as animation
import random
import sys
import json

class NNVisualize(object):
    
    def __init__(self, title, settings=None, nice=False):
        """
            Three plots
            bellman error
            average reward
            discounted reward error
        """
        self._nice=nice
        self._title=title
        if (settings != None):
            self._iteration_scale = ((settings['plotting_update_freq_num_rounds']*settings['max_epoch_length']*settings['epochs'] * 
                                      settings['training_updates_per_sim_action']) / 
                                     settings['sim_action_per_training_update'])
        else:
            self._iteration_scale = 1
        """
        self._fig, (self._bellman_error_ax, self._reward_ax, self._discount_error_ax) = plt.subplots(3, 1, sharey=False, sharex=True)
        self._bellman_error, = self._bellman_error_ax.plot([], [], linewidth=2.0)
        self._bellman_error_std = self._bellman_error_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
        self._bellman_error_ax.set_title('Bellman Error')
        self._bellman_error_ax.set_ylabel("Absolute Error")
        self._reward, = self._reward_ax.plot([], [], linewidth=2.0)
        self._reward_std = self._reward_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
        self._reward_ax.set_title('Mean Reward')
        self._reward_ax.set_ylabel("Reward")
        self._discount_error, = self._discount_error_ax.plot([], [], linewidth=2.0)
        self._discount_error_std = self._discount_error_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
        self._discount_error_ax.set_title('Discount Error')
        self._discount_error_ax.set_ylabel("Absolute Error")
        plt.xlabel("Iteration")
        
        self._fig.set_size_inches(8.0, 12.5, forward=True)
        """
        self._settings = settings
        self._movie = None
        
    def init(self):
        """
            Three plots
            bellman error
            average reward
            discounted reward error
        """
        # self._fig, (self._bellman_error_ax, self._reward_ax, self._discount_error_ax) = plt.subplots(1, 1, sharey=False, sharex=True)
        self._fig, (self._bellman_error_ax) = plt.subplots(1, 1, sharey=False, sharex=True)
        self._bellman_error, = self._bellman_error_ax.plot([], [], linewidth=2.0)
        self._bellman_error_std = self._bellman_error_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
        # self._bellman_error_ax.set_title('Error', fontsize=18)
        if ( not self._nice ):
            self._bellman_error_ax.set_ylabel("Value", fontsize=16)
        else:
            self._bellman_error_ax.set_ylabel("Reward", fontsize=16)
        """
        self._reward, = self._reward_ax.plot([], [], linewidth=2.0)
        self._reward_std = self._reward_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
        self._reward_ax.set_title('Mean Reward')
        self._reward_ax.set_ylabel("Reward")
        self._discount_error, = self._discount_error_ax.plot([], [], linewidth=2.0)
        self._discount_error_std = self._discount_error_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
        self._discount_error_ax.set_title('Discount Error')
        self._discount_error_ax.set_ylabel("Absolute Error")
        plt.xlabel("Iteration")
        """
        if ( not self._nice ):
            self._fig.suptitle(self._title, fontsize=16)
        plt.grid(b=True, which='major', color='black', linestyle='--', alpha=0.5)
        plt.grid(b=True, which='minor', color='g', linestyle='--', alpha=0.5)
        if ( not self._nice ):
            plt.xlabel("Iteration x" + str(self._iteration_scale), fontsize=18)
        
        self._fig.set_size_inches(8.0, 4.5, forward=True)
        
        if ("save_video_to_file" in self._settings):
            from util.SimulationUtil import getDataDirectory
            directory = getDataDirectory(self._settings)
            self._movie = imageio.get_writer(directory + "reward_video.mp4", mode='I',  fps=30)
        
        
    def updateLoss(self, error, std):
        self._bellman_error.set_xdata(np.arange(len(error)))
        self._bellman_error.set_ydata(error)
        self._bellman_error_ax.collections.remove(self._bellman_error_std)
        self._bellman_error_std = self._bellman_error_ax.fill_between(np.arange(len(error)), error - std, error + std, facecolor='blue', alpha=0.5)
        
        
        self._bellman_error_ax.relim()      # make sure all the data fits
        self._bellman_error_ax.autoscale()
        
        
    def show(self):
        plt.show()
        
    def redraw(self):
        self._fig.canvas.draw()
        if ("save_video_to_file" in self._settings):
            import io
            buf = io.BytesIO()
            self._fig.savefig(buf, format = 'png')
            buf.seek(0)
            data = buf.getbuffer()
            # print ("fig data: ", data)
            # print ("self._fig.canvas.print_to_buffer(): ", self._fig.canvas.print_to_buffer())
            image_, (width, height) = self._fig.canvas.print_to_buffer()
            image_ = np.frombuffer(image_, np.uint8).reshape((height, width, 4))
            # image_ = np.frombuffer(data)
            # image_ = np.array(image_, dtype="uint8")
            # movieWriter.append_data(image_)
            self._movie.append_data(image_)
        
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
        if ("save_video_to_file" in self._settings):
            self._movie.finish()
        plt.close(self._fig)
        plt.close()
        
        
if __name__ == "__main__":
    
    datafile = sys.argv[1]
    file = open(datafile)
    trainData = json.load(file)
    # print "Training data: " + str(trainingData)
    file.close()
    
    length = len(trainData["mean_bellman_error"])
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
    
    nlv = NNVisualize(datafile)
    nlv.init()
    nlv.updateLoss(np.array(trainData["mean_forward_dynamics_loss"][:length]), np.array(trainData["std_forward_dynamics_loss"][:length]))
    nlv.saveVisual("pendulum_agent_FD")
    nlv.show()
