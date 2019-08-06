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

    def getObservation(self):
        encoded_obs = self.model._get_latent_variable([[self._previous_observation]])[0]
        if (self._previous_image is not None and
                "use_dual_state_representations" in self.getSettings() and
                self.getSettings()['use_dual_state_representations']):
            return [[
                encoded_obs,
                self._previous_image
            ]]
        else:
            return [encoded_obs]

    def getState(self):
        encoded_obs = self.model._get_latent_variable([[self._previous_observation]])[0]
        if (self._previous_image is not None and
                "use_dual_state_representations" in self.getSettings() and
                self.getSettings()['use_dual_state_representations']):
            return [[
                encoded_obs,
                self._previous_image
            ]]
        else:
            return [encoded_obs]
