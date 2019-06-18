import numpy as np
from sim.SimInterface import SimInterface
from sim.OpenAIGymEnv import OpenAIGymEnv
from sim.MultiworldEnv import MultiworldEnv

class MultiworldHRLEnv(MultiworldEnv):

    def __init__(self, exp, settings, multiAgent=False, observation_key="observation"):
        #------------------------------------------------------------
        # set up initial state
        OpenAIGymEnv.__init__(self, exp, settings, multiAgent=multiAgent)
        self._observation_key = observation_key
        assert self._observation_key in self.getEnvironment().observation_space.spaces
        if ("ignore_hlc_actions" in self.getSettings()
                and (self.getSettings()["ignore_hlc_actions"] == True)):
            self._ran = 0.6  ## Ignore HLC action and have env generate them if > 0.5.
        else:
            self._ran = 0.4  ## Ignore HLC action and have env generate them if > 0.5.

    def reset(self):
        # self.getEnvironment().init()
        self._previous_observation = self.getEnvironment().reset()[self._observation_key]
        self._llc_target = np.zeros(self.getEnvironment().observation_space[self._observation_key].shape)
        self._end_of_episode = False
        self._fallen=[False]
        self._hlc_timestep = 1000000
        self._hlc_skip = 10
        if ("hlc_timestep" in self.getSettings()):
            self._hlc_skip = self.getSettings()["hlc_timestep"]
        return self.getState()

    def init(self):
        self.reset()
            
    def initEpoch(self):
        self.reset()

    def setLLC(self, llc):
        self._llc = llc

    def getNumAgents(self):
        return 2
        
    def step(self, action):
        """
            action[0] == hlc action
            action[1] == llc action
        """
        self._hlc_timestep = self._hlc_timestep + 1
        if (self._hlc_timestep >= self._hlc_skip
                and (self._ran < 0.5)):
            action_ = np.array(action[0])
            self._llc_target = np.array(action_)
            self._hlc_timestep = 0
            llc_obs = np.concatenate([self._previous_observation, self._llc_target], 0)
            action[1] = self._llc.predict([llc_obs])[0, :]
        action_ = np.array(action[1])
        if ("use_hlc_action_directly" in self.getSettings()
                and (self.getSettings()["use_hlc_action_directly"] == True)):
            action_ = self._llc_target
        observation, reward, done, info = self.getEnvironment().step(action_)
        if (self.getSettings()['render']):
            self.getEnvironment().render()
        self._end_of_episode = done
        # self._fallen = done
        self._previous_observation = observation[self._observation_key]
        distance = self._previous_observation - self._llc_target
        llc_reward = -(distance*distance).sum()
        self.__reward = np.array([[reward], [llc_reward]])
        return self.__reward

    def getState(self):
        # state = np.array(self._exp.getState())
        # observation, reward, done, info = env.step(action)
        # self._previous_observation = observation

        llc_obs = np.concatenate([self._previous_observation, self._llc_target], 0)
        hlc_obs = np.concatenate([self._previous_observation, np.zeros([self._llc_target.size])], 0)
        state_ = np.stack([hlc_obs, llc_obs])
        return state_

    def getObservation(self):
        return self.getState()

    def actContinuous(self, action, bootstrapping):
        return self.step(action)

    def calcReward(self):
        return self.__reward