"""
"""
import numpy as np
import math
from sim.SimInterface import SimInterface 
import copy 
# import scipy.integrate as integrate
# import matplotlib.animation as animation
import gym
from gym import wrappers
from gym import envs
print(envs.registry.all())

from model.ModelUtil import getOptimalAction, getMBAEAction


class OpenAIGymEnv(SimInterface):

    def __init__(self, exp, settings):
        #------------------------------------------------------------
        # set up initial state
        super(OpenAIGymEnv,self).__init__(exp, settings)
        self._previous_observation=None
        self._end_of_episode=False
        
        ## Should print the type of actions space, continuous/discrete, how many parameters
        print(self.getEnvironment().action_space)
        ## Should print the type of state space, continuous/discrete, how many parameters
        print(self.getEnvironment().observation_space)

    def init(self):
        self._previous_observation = self.getEnvironment().reset()
            
    def initEpoch(self):
        self._previous_observation = self.getEnvironment().reset()
        
    def finish(self):   
        self._exp.finish()
    
    def generateValidation(self, data, epoch):
        self.getEnvironment().generateValidationEnvironmentSample(epoch)
    
    def generateEnvironmentSample(self):
        self.getEnvironment().generateEnvironmentSample()
        
    def getEvaluationData(self):
        return self.getEnvironment().getEvaluationData()
    
    def getActor(self):
        return self._exp
    
    def finish(self):
        self._exp.finish()
    
    def getState(self):
        # state = np.array(self._exp.getState())
        # observation, reward, done, info = env.step(action)
        # self._previous_observation = observation
        
        state_ = np.array(observation)
        state = np.array(state_)
        state = np.reshape(state, (-1, len(state_)))
        
        return state
    
    def setState(self, st):
        self._agent = st
        self._box.state[0,0] = st[0]
        self._box.state[0,1] = st[1]
        
    def visualizeNextState(self, next_state_, action):
        _t_length = self.getEnvironment()._game_settings['num_terrain_samples']
        terrain = next_state_[:_t_length]
        terrain_dx = next_state_[_t_length]
        terrain_dy = next_state_[_t_length+1]
        character_features = next_state_[_t_length+2:]
        self.getEnvironment().visualizeNextState(terrain, action, terrain_dx)  
    
    def updateViz(self, actor, agent, directory):
        pass
