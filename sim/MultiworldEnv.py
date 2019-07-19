import numpy as np
from sim.SimInterface import SimInterface
from sim.OpenAIGymEnv import OpenAIGymEnv


class MultiworldEnv(OpenAIGymEnv):

    def __init__(self, exp, settings, multiAgent=False,
                 image_key="image_observation", state_key="state_observation"):
        # ------------------------------------------------------------
        # set up initial state
        OpenAIGymEnv.__init__(self, exp, settings, multiAgent=multiAgent)
        self._image_key = image_key
        self._state_key = state_key
        assert self._image_key in self.getEnvironment().observation_space.spaces
        assert self._state_key in self.getEnvironment().observation_space.spaces
        self.observation_space = self.getEnvironment().observation_space.spaces[self._state_key]

    def reset(self):
        # self.getEnvironment().init()
        state_dict = self.getEnvironment().reset()
        self._previous_dict = state_dict
        self._previous_observation = state_dict[self._state_key]
        self._previous_image = state_dict[self._image_key]
        self._end_of_episode = False
        self._fallen = [False]
        return self._previous_observation

    def init(self):
        # self.getEnvironment().init()
        state_dict = self.getEnvironment().reset()
        self._previous_dict = state_dict
        self._previous_observation = state_dict[self._state_key]
        self._previous_image = state_dict[self._image_key]
        self._end_of_episode = False
        self._fallen = [False]

    def initEpoch(self):
        state_dict = self.getEnvironment().reset()
        self._previous_dict = state_dict
        self._previous_observation = state_dict[self._state_key]
        self._previous_image = state_dict[self._image_key]
        self._end_of_episode = False
        self._fallen = [False]

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
        self._previous_dict = observation
        self._previous_observation = observation[self._state_key]
        self._previous_image = observation[self._image_key]
        return reward

    def actContinuous(self, action, bootstrapping):
        reward = self.step(action)
        self.__reward = reward
        return reward

    def getFullViewData(self):
        return self._previous_image

    def getViewData(self):
        return self._previous_image

    def _getVisualState(self):
        return self._previous_image

    def getVisualState(self):
        return self._previous_image

    def _getImitationVisualState(self):
        return self._previous_image

    def getImitationVisualState(self):
        return self._previous_image

    def getObservation(self):
        if ("use_dual_state_representations" in self.getSettings() and
                self.getSettings()['use_dual_state_representations']):
            return [[
                self._previous_observation,
                self._previous_image
            ]]
        else:
            return self._previous_observation

    def getState(self):
        if ("use_dual_state_representations" in self.getSettings() and
                self.getSettings()['use_dual_state_representations']):
            return [[
                self._previous_observation,
                self._previous_image
            ]]
        else:
            return self._previous_observation