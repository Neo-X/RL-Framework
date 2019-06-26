import numpy as np
from sim.SimInterface import SimInterface
from sim.OpenAIGymEnv import OpenAIGymEnv

class MultiworldEnv(OpenAIGymEnv):

    def __init__(self, exp, settings, multiAgent=False, observation_key="observation"):
        #------------------------------------------------------------
        # set up initial state
        OpenAIGymEnv.__init__(self, exp, settings, multiAgent=multiAgent)
        self._observation_key = observation_key
        assert self._observation_key in self.getEnvironment().observation_space.spaces
        self.observation_space = self.getEnvironment().observation_space.spaces[observation_key]

    def reset(self):
        # self.getEnvironment().init()
        self._previous_observation = self.getEnvironment().reset()[self._observation_key]
        self._end_of_episode = False
        self._fallen=[False]
        return self._previous_observation

    def init(self):
        # self.getEnvironment().init()
        self._previous_observation = self.getEnvironment().reset()[self._observation_key]
        self._end_of_episode = False
        self._fallen=[False]
            
    def initEpoch(self):
        self._previous_observation = self.getEnvironment().reset()[self._observation_key]
        self._end_of_episode = False
        self._fallen=[False]
        
    def step(self, action):
        action_ = np.array(action)
        if (self.getSettings()['render']):
            self.getEnvironment().render()
        if (self._multiAgent):
            observation, reward, done, info = self.getEnvironment().step(action_)
        else:
            observation, reward, done, info = self.getEnvironment().step(action_[0])
        self._end_of_episode = done
        # self._fallen = done
        self._previous_observation = observation[self._observation_key]
        return reward

    def actContinuous(self, action, bootstrapping):
        reward = self.step(action)
        self.__reward = reward
        return reward