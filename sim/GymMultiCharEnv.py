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
# import roboschool
from OpenGL import GL
from model.ModelUtil import checkDataIsValid

from model.ModelUtil import getOptimalAction, getMBAEAction


class GymMultiCharEnv(SimInterface):

    def __init__(self, exp, settings):
        #------------------------------------------------------------
        # set up initial state
        super(GymMultiCharEnv,self).__init__(exp, settings)
        self._previous_observation=None
        self._end_of_episode=False
        self._num_updates_since_last_action=10000
        
        ## Should print the type of actions space, continuous/discrete, how many parameters
        print(self.getEnvironment().action_space)
        ## Should print the type of state space, continuous/discrete, how many parameters
        print(self.getEnvironment().observation_space)
        """
        if ( settings['sim_config_file'] == 'RoboschoolHopper-v1'):
            self._state_param_mask = [  True ,  False       ,  False      ,  True ,  False        ,
                                        True,  False        ,  True,  True,  True,
                                        True,  True,  True ,  True,  True]
        else:
            self._state_param_mask = [   True] * len(settings['state_bounds'][0])
        """
    def init(self):
        # self.getEnvironment().init()
        self._previous_observation = self.getEnvironment().reset()
        self._end_of_episode = False
        self._num_updates_since_last_action=10000
            
    def initEpoch(self):
        self._previous_observation = self.getEnvironment().reset()
        while not checkDataIsValid(self._previous_observation[0]):
            self._previous_observation = self.getEnvironment().reset()
        
        self._end_of_episode = False
        self._num_updates_since_last_action=10000
        
    def endOfEpoch(self):
        eoe = self._exp.endOfEpoch()
        return self._exp.endOfEpoch()
        # return self._end_of_episode

    def finish(self):   
        self._exp.finish()
    
    def generateValidation(self, data, epoch):
        self.initEpoch()
    
    def generateEnvironmentSample(self):
        self.initEpoch()
        
    def getEvaluationData(self):
        return self.getEnvironment().getEvaluationData()
    
    def finish(self):
        # self._exp.finish()
        pass
        
    def step(self, action):
        action_ = np.array(action)
        if (self.getSettings()['render']):
            self.getEnvironment().render()
        observation, reward, done, info = self.getEnvironment().step(action_)
        # print ("observation: ", observation)
        self._end_of_episode = done
        self._previous_observation = observation
        return reward
    
    def getState(self):
        # state = np.array(self._exp.getState())
        # observation, reward, done, info = env.step(action)
        # self._previous_observation = observation
        
        # state_ = np.array(self._previous_observation)
        state_ = np.array(self._previous_observation)
        # state = np.reshape(state_, (len(state_), -1))
        
        return state_
    
    def getLLCState(self):
        """
            Want just the character state at the end.
        """
        state_ = self.getEnvironment().getLLCState()
        # print ("state_: ", state_)
        # state = np.array(state_)[200:]
        # state = np.reshape(state, (-1, len(state_)-200))
        state = np.array(state_)
        state = np.reshape(state, (len(state_), -1))
        return state
    
    def update(self):
        for i in range(1):
            self.getEnvironment().update()
            self._num_updates_since_last_action+=1
        # self.getEnvironment().display()
        self._previous_observation = self.getEnvironment().getObservation()
                
    def updateAction(self, action_):
        
        self.getActor().updateAction(self, action_)
        self._num_updates_since_last_action = 0
        # print("update action: self._num_updates_since_last_action: ", self._num_updates_since_last_action)

    def updateLLCAction(self, action_ ):
        self.getActor().updateLLCAction(self, action_)
        
    def needUpdatedAction(self):
        timestep = self.getSettings()['hlc_timestep']
        # print ("needUpdateAction: self._num_updates_since_last_action: ", self._num_updates_since_last_action )
        if ( self._num_updates_since_last_action >= timestep):
            return True
        else:
            return False
        return
    
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
    
    def updateViz(self, actor, agent, directory, p=1.0):
        pass

    def generateValidationEnvironmentSample(self, numb):
        pass
    """
    def needUpdatedAction(self):
        return True
    """
    """
    def updateAction(self, action):
        self.step(action)
        self._num_updates_since_last_action = 0
        print("update action: self._num_updates_since_last_action: ", self._num_updates_since_last_action)
    """
    
    def display(self):
        if (self.getSettings()['render']):
            self.getEnvironment().render()
    
    def setRandomSeed(self, seed):
        """
            Set the random seed for the simulator
            This is helpful if you are running many simulations in parallel you don't
            want them to be producing the same results if they all init their random number 
            generator the same.
        """
        print ( "Setting random seed: ", seed )
        self.getEnvironment().setRandomSeed(seed)