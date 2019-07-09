import sys
import math
from actor.ActorInterface import ActorInterface
import numpy as np
from model.ModelUtil import reward_smoother
import dill, copy
from algorithm.KERASAlgorithm import KERASAlgorithm
from util.SimulationUtil import createNetworkModel, createRLAgent

class MultiworldMultiCharActor(ActorInterface):
    
    def __init__(self, settings, experience, timeskip=20):
        super(MultiworldMultiCharActor,self).__init__(settings, experience)
        self._llc_policy = None
        self.timeskip = timeskip
        if ("hlc_timestep" in self._settings):
            self.timeskip = self._settings["hlc_timestep"]
        
    def init(self):

        print ("Loading pre compiled network")
        file_name=self._settings['llc_policy_model_path']

        import json
        file = open(file_name)
        settings = json.load(file)
        file.close()

        settings["load_saved_model"] = True
        model = createRLAgent(settings['agent_name'], state_bounds=settings["state_bounds"],
                               discrete_actions=np.array([[0]]),
                               reward_bounds=settings["reward_bounds"],
                               settings=settings)

        self._llc_policy = model
    
    # @profile(precision=5)
    def actContinuous(self, sim, action_, bootstrapping=False):

        all_rewards = 0
        for i in range(self.timeskip):
            state = sim.getState()
            goal = action_
            llc_obs = np.concatenate([state, goal], -1)
            llc_action = self._llc_policy.predict(llc_obs)
            reward = sim.step(llc_action)
            all_rewards += reward
        self._reward_sum = self._reward_sum + all_rewards
        return all_rewards
        
    def updateActor(self, sim, action_):

        return self.actContinuous(sim, action_)
        