import numpy as np
from sim.SimInterface import SimInterface
from sim.OpenAIGymEnv import OpenAIGymEnv


class MultiworldEnv(OpenAIGymEnv):

    def __init__(self, exp, settings, multiAgent=False,
                 image_key="image_observation", state_key="state_observation"):
        # ------------------------------------------------------------
        # set up initial state
        self._image_key = image_key
        self._state_key = state_key
        self._previous_image = None
        assert self._state_key in exp.observation_space.spaces
        self.observation_space = exp.observation_space.spaces[self._state_key]
        self.action_space = exp.action_space
        OpenAIGymEnv.__init__(self, exp, settings, multiAgent=multiAgent)

    def reset(self):
        # self._exp.init()
        state_dict = self._exp.reset()
        self._previous_dict = state_dict
        self._previous_observation = state_dict[self._state_key]
        if self._image_key in state_dict:
            self._previous_image = state_dict[self._image_key]
        self._end_of_episode = False
        self._fallen = [False]
        return self._previous_observation

    def init(self):
        # self._exp.init()
        state_dict = self._exp.reset()
        self._previous_dict = state_dict
        self._previous_observation = state_dict[self._state_key]
        if self._image_key in state_dict:
            self._previous_image = state_dict[self._image_key]
        self._end_of_episode = False
        self._fallen = [False]

    def initEpoch(self):
        state_dict = self._exp.reset()
        self._previous_dict = state_dict
        self._previous_observation = state_dict[self._state_key]
        if self._image_key in state_dict:
            self._previous_image = state_dict[self._image_key]
        self._end_of_episode = False
        self._fallen = [False]

    def step(self, action):
        action_ = np.array(action)
        if (self.getSettings()['render']):
            self._exp.render()
        if (self._multiAgent):
            observation, reward, done, info = self._exp.step(action_)
        else:
            observation, reward, done, info = self._exp.step(action_[0])
        self._end_of_episode = done
        # self._fallen = done
        self._previous_dict = observation
        self._previous_observation = observation[self._state_key]
        if self._image_key in observation:
            self._previous_image = observation[self._image_key]
        return reward

    def actContinuous(self, action, bootstrapping):
        reward = self.step(action)
        self.__reward = reward
        return reward

    def getObservation(self):
        if (self._previous_image is not None and
                "use_dual_state_representations" in self.getSettings() and
                self.getSettings()['use_dual_state_representations']):
            return [[
                self._previous_observation,
                self._previous_image
            ]]
        else:
            return [self._previous_observation]

    def getState(self):
        if (self._previous_image is not None and
                "use_dual_state_representations" in self.getSettings() and
                self.getSettings()['use_dual_state_representations']):
            return [[
                self._previous_observation,
                self._previous_image
            ]]
        else:
            return [self._previous_observation]