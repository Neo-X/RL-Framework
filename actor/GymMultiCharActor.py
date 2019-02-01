import sys
import math
from actor.ActorInterface import ActorInterface
import numpy as np
from model.ModelUtil import reward_smoother
import dill, copy
from algorithm.KERASAlgorithm import KERASAlgorithm
from util.SimulationUtil import createNetworkModel, createRLAgent

class GymMultiCharActor(ActorInterface):
    
    def __init__(self, discrete_actions, experience):
        super(GymMultiCharActor,self).__init__(discrete_actions, experience)
        self._target_vel_weight=self._settings["target_velocity_decay"]
        self._target_vel = self._settings["target_velocity"]
        # self._target_vel = self._settings["target_velocity"]
        self._end_of_episode=False
        self._param_mask = [    False,        True,        True,        False,        False,    
        True,        True,        True,        True,        True,        True,        True,    
        True,        True,        True,        True,        True,        True,        True,    
        False,        True,        True,        True,        True,        True,        True,    
        False,        True,        True,        True,        True,        True,        True]
        
        self._llc_policy = None
        model = None
        if ('llc_policy_model_path' in self._settings):
            print ("Loading pre compiled network")
            file_name=self._settings['llc_policy_model_path']
            """
            if (file_name[-5:] == '.json'): ### Keras model
                import json
                file = open(file_name)
                settings = json.load(file)
                file.close()
                
                settings["load_saved_model"] = True
                # settings["load_saved_model"] = "network_and_scales"
                model = createRLAgent(settings['agent_name'], state_bounds=settings["state_bounds"],
                                       discrete_actions=np.array([[0]]), 
                                       reward_bounds=settings["reward_bounds"], 
                                       settings=settings)
    
            else: ### Lasagne model
            """
            f = open(file_name, 'rb')
            model = dill.load(f)
            # model.setSettings(settings_)
            f.close()
                
            self._llc_policy = model
        
    def updateAction(self, sim, action_):
        action_ = np.array(action_, dtype='float64')
        sim.getEnvironment().updateAction(action_)
        
    def updateLLCAction(self, sim, action_):
        """
            This can consists of a vector of actions for each LLC
        """
        action_ = np.array(action_, dtype='float64')
        sim.getEnvironment().updateLLCAction(action_)
    
    # @profile(precision=5)
    def act(self, exp, action_, bootstrapping=False):
        samp = self.getActionParams(action_)
        
        reward = self.actContinuous(exp, samp, bootstrapping=bootstrapping)
        
        return reward
    
    # @profile(precision=5)
    def actContinuous(self, sim, action_, bootstrapping=False):
        """
            sim.needUpdatedAction() is be false
        
        """
        sim.updateAction(action_)
        ## This should make sim.needUpdatedAction() == false
        # reward = sim.step(action_)
        updates_=0
        stumble_count=0
        torque_sum=0
        tmp_reward_sum=0
        # print ("sim: ", sim, " sim.needUpdatedAction(): ", sim.needUpdatedAction())
        # print ("sim.agentHasFallen(): ", sim.endOfEpoch())
        reward_ = np.array(sim.getEnvironment().calcRewards())
        while (not sim.needUpdatedAction() and (updates_ < 100)
               # and (not sim.endOfEpoch())
               ):
            # sim.updateAction(action_)
            self.updateActor(sim, action_)
            updates_+=1
            reward_ = reward_ + np.array(sim.getEnvironment().calcRewards())
            # print("Update #: ", updates_)
        if (updates_ == 0): #Something went wrong...
            print("There were no updates... This is bad")
            return np.array(sim.getEnvironment().calcRewards()) * 0.0
        # reward_ = np.reshape(sim.getEnvironment().calcRewards(), (len(action_),1))
        # reward_ = sim.getEnvironment().calcRewards()
        reward_ = reward_/updates_
        # print ("reward_: ", repr(reward_))
        self._reward_sum = self._reward_sum + np.mean(reward_)
        # return reward_[0][0]
        return reward_
        
    
    def getEvaluationData(self):
        return self._reward_sum
    
    def hasNotFallen(self, exp):
        """
            Returns True when the agent is still going (not end of episode)
            return false when the agent has fallen (end of episode)
        """
        if ( exp.endOfEpoch() ):
            return 0
        else:
            return 1
        
    def updateActor(self, sim, action_):
        # llc_state = sim.getState()[:,self._settings['num_terrain_features']:]
        llc_state = sim.getLLCState()
        # llc_state = sim.getState()
        # print("LLC state: ", llc_state.shape)
        # print("LLC state: ", llc_state)
        # llc_state = llc_state[:,self._settings["num_terrain_features"]:]
        # print("LLC state: ", llc_state[0].shape,  " ", llc_state[0])
        llc_state = np.array(llc_state)
        # action__ = np.array([[action_[0], action_[1], 0.0, action_[2], action_[3], 0.0, action_[4]]])
        # print ("llc pose state: ", llc_state.shape, repr(llc_state))
        # print ("hlc action: ", action__.shape, repr(action__))
        # llc_state = np.concatenate((llc_state, action__), axis=1)
        
        for i in range(len(action_)):
            # action__ = np.array([[action_[i][4], action_[i][0], 0.0, action_[i][1], action_[i][2], 0.0, action_[i][3]]])
            action__ = np.array([action_[i][4], action_[i][0], 0.0, action_[i][1], action_[i][2], 0.0, action_[i][3]])
            # print ("LLC goal: ", action__)
            # print ("LLC Current state: ", llc_state[i])
            # print ("LLC Current goal: ", llc_state[i][-7:])
            llc_state[i][-7:] = action__
        # print ("llc_state: ", llc_state.shape, llc_state)
        
        # llc_state = np.reshape(llc_state, (len(action_), len(llc_state)))
        # print ("llc_state shape: ", llc_state.shape)
        llc_action = self._llc_policy.predict(llc_state)
        # print("llc_action: ", llc_action.shape, llc_action)
        sim.updateLLCAction(llc_action)
        sim.update()
        if (self._settings["shouldRender"]):
            sim.display()
        # rw_ = sim.getEnvironment().calcReward()
        # tmp_reward_sum=tmp_reward_sum + rw_
        # print("reward: ", rw_, " reward_sum:, ", tmp_reward_sum)
        