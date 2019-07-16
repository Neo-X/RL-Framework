import numpy as np
from sim.SimInterface import SimInterface
from sim.MultiworldEnv import MultiworldEnv

class MultiworldGoalEnv(MultiworldEnv):

    def __init__(self, exp, settings, multiAgent=False,
                 observation_key="observation", goal_key="desired_goal", timeskip=20):
        #------------------------------------------------------------
        # set up initial state
        MultiworldEnv.__init__(self, exp, settings, multiAgent=multiAgent, observation_key=observation_key)
        self.goal_key = goal_key
        self._timestep = 0
        self._skip = timeskip

    def reset(self):
        # self.getEnvironment().init()
        self._goal = self.getEnvironment().sample_goals(1)[self.goal_key][0, :]
        observation = self.getEnvironment().reset()[self._observation_key]
        self._previous_observation = np.concatenate([observation, self._goal], -1)
        self._end_of_episode = False
        self._fallen=[False]
        self._timestep = 0
        return self._previous_observation

    def init(self):
        # self.getEnvironment().init()
        self._goal = self.getEnvironment().sample_goals(1)[self.goal_key][0, :]
        observation = self.getEnvironment().reset()[self._observation_key]
        self._previous_observation = np.concatenate([observation, self._goal], -1)
        self._end_of_episode = False
        self._fallen=[False]
        self._timestep = 0
            
    def initEpoch(self):
        self._goal = self.getEnvironment().sample_goals(1)[self.goal_key][0, :]
        observation = self.getEnvironment().reset()[self._observation_key]
        self._previous_observation = np.concatenate([observation, self._goal], -1)
        self._end_of_episode = False
        self._fallen=[False]
        self._timestep = 0
        
    def step(self, action):
        self._timestep = self._timestep + 1
        action_ = np.array(action)
        if (self.getSettings()['render']):
            self.getEnvironment().render(mode='cv2')
        if (self._multiAgent):
            observation, reward, done, info = self.getEnvironment().step(action_)
        else:
            observation, reward, done, info = self.getEnvironment().step(action_[0])
        self._end_of_episode = done
        # self._fallen = done
        observation = observation[self._observation_key]
        self._previous_observation = np.concatenate([observation, self._goal], -1)
        distance = observation - self._goal
        reward = -np.sqrt((distance * distance).sum())
        self.__reward = np.array([[reward]])
        if self._timestep > self._skip:
            self._timestep = 0
            self._goal = self.getEnvironment().sample_goals(1)[self.goal_key][0, :]
        return self.__reward

    def actContinuous(self, action, bootstrapping):
        return self.step(action)