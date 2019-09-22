import numpy as np
from sim.SimInterface import SimInterface
from sim.OpenAIGymEnv import OpenAIGymEnv
from sim.MultiworldEnv import MultiworldEnv

class MultiworldHRLEnv(MultiworldEnv):

    def __init__(self, exp, settings, multiAgent=False,
                 image_key="image_observation", state_key="state_observation"):
        #------------------------------------------------------------
        # set up initial state
        MultiworldEnv.__init__(self, exp, settings, multiAgent=multiAgent,
                               image_key=image_key, state_key=state_key)

    def reset(self):
        super(MultiworldHRLEnv, self).reset()
        self._hlc_timestep = 1000000
        self._hlc_skip = 10
        self._goal = np.zeros_like(self._previous_observation)
        if ("hlc_timestep" in self.getSettings()):
            self._hlc_skip = self.getSettings()["hlc_timestep"]
        return self.getState()

    def init(self):
        self.reset()
            
    def initEpoch(self):
        self.reset()

    def setLLC(self, llc):
        self._llc = llc

    def getEnvironment(self):
        return self

    def getEnv(self):
        return self

    def getNumAgents(self):
        return 2
        
    def step(self, action):
        """
            action[0] == hlc action
            action[1] == llc action
        """
        self._hlc_timestep = self._hlc_timestep + 1
        if self._hlc_timestep >= self._hlc_skip:
            action_ = np.array(action[0])
            self._goal = np.array(action_)
            self._hlc_timestep = 0
            llc_obs = np.concatenate([self._previous_observation, self._goal], -1)
            action[1] = self._llc.predict([llc_obs])[0, :]
        action_ = np.array(action[1])
        observation, reward, done, info = self.getEnvironment().step(action_)
        if (self.getSettings()['render']):
            self.getEnvironment().render()
        self._end_of_episode = done
        # self._fallen = done
        self._previous_dict = observation
        self._previous_observation = observation[self._state_key]
        if self._image_key in observation:
            self._previous_image = observation[self._image_key]
        distance = self._previous_observation - self._goal
        llc_reward = -np.sqrt((distance * distance).sum())
        self.__reward = np.array([[reward], [llc_reward]])
        return self.__reward

    def actContinuous(self, action, bootstrapping):
        return self.step(action)

    def calcReward(self):
        return self.__reward

    def getObservation(self):
        llc_obs = np.concatenate([self._previous_observation, self._goal], 0)
        hlc_obs = np.concatenate([self._previous_observation, np.zeros([self._goal.size])], 0)
        state_ = np.stack([hlc_obs, llc_obs])
        if (self._previous_image is not None and
                "use_dual_state_representations" in self.getSettings() and
                self.getSettings()['use_dual_state_representations']):
            return [[
                state_,
                self._previous_image
            ]]
        else:
            return [self._previous_observation]

    def getState(self):
        llc_obs = np.concatenate([self._previous_observation, self._goal], 0)
        hlc_obs = np.concatenate([self._previous_observation, np.zeros([self._goal.size])], 0)
        state_ = np.stack([hlc_obs, llc_obs])
        if (self._previous_image is not None and
                "use_dual_state_representations" in self.getSettings() and
                self.getSettings()['use_dual_state_representations']):
            return [[
                state_,
                self._previous_image
            ]]
        else:
            return [self._previous_observation]