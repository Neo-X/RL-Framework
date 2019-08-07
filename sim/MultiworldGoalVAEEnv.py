import numpy as np
from sim.SimInterface import SimInterface
from sim.MultiworldVAEEnv import MultiworldVAEEnv
import gym


class MultiworldGoalVAEEnv(MultiworldVAEEnv):

    def __init__(self, exp, settings, multiAgent=False,
                 image_key="image_observation",
                 timeskip=20):
        #------------------------------------------------------------
        # set up initial state
        MultiworldVAEEnv.__init__(self, exp, settings, multiAgent=multiAgent,
                               image_key=image_key)
        self._timestep = 0
        self._skip = timeskip
        self.observation_space = gym.spaces.Box(
            -1.0 * np.ones([2 * settings["encoding_vector_size"]]),
            1.0 * np.ones([2 * settings["encoding_vector_size"]]))

    def reset(self):
        super(MultiworldGoalVAEEnv, self).reset()
        goal_dict = self.getEnvironment().sample_goals(1)
        self.getEnvironment().set_to_goal(goal_dict)
        self._goal = self.encode(self.getEnvironment()._get_obs()[self._state_key])
        self.getEnvironment().set_to_goal(self._previous_dict)
        self._previous_observation = np.concatenate([self._previous_observation, self._goal], -1)
        self._timestep = 0
        return self._previous_observation

    def init(self):
        super(MultiworldGoalVAEEnv, self).init()
        goal_dict = self.getEnvironment().sample_goals(1)
        self.getEnvironment().set_to_goal(goal_dict)
        self._goal = self.encode(self.getEnvironment()._get_obs()[self._state_key])
        self.getEnvironment().set_to_goal(self._previous_dict)
        self._previous_observation = np.concatenate([self._previous_observation, self._goal], -1)
        self._timestep = 0
            
    def initEpoch(self):
        super(MultiworldGoalVAEEnv, self).initEpoch()
        goal_dict = self.getEnvironment().sample_goals(1)
        self.getEnvironment().set_to_goal(goal_dict)
        self._goal = self.encode(self.getEnvironment()._get_obs()[self._state_key])
        self.getEnvironment().set_to_goal(self._previous_dict)
        self._previous_observation = np.concatenate([self._previous_observation, self._goal], -1)
        self._timestep = 0
        
    def step(self, action):
        self._timestep = self._timestep + 1
        super(MultiworldGoalVAEEnv, self).step(action)
        reward = -np.sqrt(np.square(self._previous_observation - self._goal).sum())
        self.__reward = np.array([[reward]])
        if self._timestep >= self._skip:
            self._timestep = 0
            goal_dict = self.getEnvironment().sample_goals(1)
            self.getEnvironment().set_to_goal(goal_dict)
            self._goal = self.encode(self.getEnvironment()._get_obs()[self._state_key])
            self.getEnvironment().set_to_goal(self._previous_dict)
        self._previous_observation = np.concatenate([self._previous_observation, self._goal], -1)
        return self.__reward
