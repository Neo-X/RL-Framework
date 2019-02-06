"""
"""
import numpy as np
import math
from sim.OpenAIGymEnv import OpenAIGymEnv 
import copy 
# import scipy.integrate as integrate
# import matplotlib.animation as animation
try:
    import gym
    from gym import wrappers
    from gym import envs
except:
    print("openAI gym is not installed")
    pass
# import roboschool
### This help openAIGym but breaks my headless rendering
# from OpenGL import GL
# print(envs.registry.all())
from model.ModelUtil import getOptimalAction, getMBAEAction

class OpenAIGymHRLEnv(OpenAIGymEnv):

    def __init__(self, exp, settings, multiAgent=False):
        #------------------------------------------------------------
        # set up initial state
        super(OpenAIGymHRLEnv,self).__init__(exp, settings)
        
    def init(self):
        # self.getEnvironment().init()
        self._previous_observation = self.getEnvironment().reset()
        self._end_of_episode = False
            
        
    def step(self, action):
        action_ = np.array(action)
        if (self.getSettings()['render']):
            self.getEnvironment().render()
        if (self._multiAgent):
            observation, reward, done, info = self.getEnvironment().step(action_)
        else:
            observation, reward, done, info = self.getEnvironment().step(action_[0])
        # print ("observation: ", observation)
        self._end_of_episode = done
        self._previous_observation = observation
        return reward
    
    def needUpdatedAction(self):
        return True
    
    def getSubPolicyState(self):
        state_ = self.getState()
        # print ("state_ shape: ", np.array(state_).shape)
        llc_state = state_[:,-6:]
        # print ("llc_state: ", llc_state)
        return llc_state 
