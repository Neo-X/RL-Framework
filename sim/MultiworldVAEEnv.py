import numpy as np
from sim.MultiworldEnv import MultiworldEnv
import gym


class MultiworldVAEEnv(MultiworldEnv):

    def __init__(self, exp, settings, multiAgent=False,
                 image_key="image_observation",
                 timeskip=20):
        # ------------------------------------------------------------
        # set up initial state
        MultiworldEnv.__init__(self, exp, settings, multiAgent=multiAgent,
                               image_key=image_key, state_key=image_key)
        self._timestep = 0
        self._skip = timeskip
        self.model = None
        self.observation_space = gym.spaces.Box(
            -1.0 * np.ones([settings["encoding_vector_size"]]),
            1.0 * np.ones([settings["encoding_vector_size"]]))

    def setVAE(self, model):
        self.model = model

    def encode(self, x):
        if self.model is None:
            return np.zeros([self.getSettings()["encoding_vector_size"]])
        return self.model._get_latent_variable([[x]])[0][0]

    def decode(self, z):
        if self.model is None:
            return np.zeros([self.getSettings()["fd_num_terrain_features"]])
        return self.model._get_reconstructed_image_from_latent_variable([[z]])[0][0]

    def reset(self):
        # self.getEnvironment().init()
        state_dict = self.getEnvironment().reset()
        self._previous_dict = state_dict
        self._previous_observation = self.encode(state_dict[self._state_key])
        if self._image_key in state_dict:
            self._previous_image = state_dict[self._image_key]
        self._end_of_episode = False
        self._fallen = [False]
        return self._previous_observation

    def init(self):
        # self.getEnvironment().init()
        state_dict = self.getEnvironment().reset()
        self._previous_dict = state_dict
        self._previous_observation = self.encode(state_dict[self._state_key])
        if self._image_key in state_dict:
            self._previous_image = state_dict[self._image_key]
        self._end_of_episode = False
        self._fallen = [False]

    def initEpoch(self):
        state_dict = self.getEnvironment().reset()
        self._previous_dict = state_dict
        self._previous_observation = self.encode(state_dict[self._state_key])
        if self._image_key in state_dict:
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
        self._previous_observation = self.encode(observation[self._state_key])
        if self._image_key in observation:
            self._previous_image = observation[self._image_key]
        return reward
