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
        # self.init()
        
    def init(self):
        
        if ('llc_policy_model_path' in self._settings):
            print ("Loading pre compiled network")
            file_name=self._settings['llc_policy_model_path']
            
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
            
                f = open(file_name, 'rb')
                model = dill.load(f)
                # model.setSettings(settings_)
                f.close()
                
            self._llc_policy = model
        
        
    def updateAction(self, sim, action_):
        if (self._llc_policy is not None):
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
    
        
    def step(self, sim, action_):
        reward = self.actContinuous(sim, action_, bootstrapping=False)
        ob = sim.getState()
        done = sim.endOfEpoch()
        # falls = sim.getEnvironment().agentHasFallenMultiAgent()
        falls = [sim.getEnvironment().endOfEpochForAgent(i) for i in range(sim.getEnvironment().getNumAgents())]
        info = {"count": [[self._count]] * self.getNumAgents(),
                "falls_sim": falls}
        # print ("info: ", info)
        return ob, reward, done, info
    
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
        reward_ = np.array(sim.getEnvironment().calcRewards()) * 0.0
        while (not sim.needUpdatedAction() and (updates_ < 100)
               # and (not sim.endOfEpoch())
               ):
            # sim.updateAction(action_)
            self.updateActor(sim, action_)
            updates_+=1
            reward_ = reward_ + np.array(sim.getEnvironment().calcRewards())
            if (sim.getMovieWriter() is not None
                and (sim.movieWriterSupport())):
                ### If the sim does not have it's own writing support
                vizData = sim.getEnvironment().getFullViewData()
                # movie_writer.append_data(np.transpose(vizData))
                # print ("sim image mean: ", np.mean(vizData), " std: ", np.std(vizData))
                image_ = np.zeros((vizData.shape))
                for row in range(len(vizData)):
                    image_[row] = vizData[len(vizData)-row - 1]
                # print ("Writing image to video") 
                sim.getMovieWriter().append_data(image_)
            
            # print("Update #: ", updates_)
        if (updates_ == 0): #Something went wrong...
            print("There were no updates... This is bad")
            # return np.array(sim.getEnvironment().calcRewards()) * 0.0
        # else:
        reward_ = reward_/updates_
        # reward_ = [[sim.getEnvironment().calcRewardForAgent(a)] for a in range(sim.getEnvironment().getNumAgents())]
        # print ("sim reward_: ", reward_)
        self._reward_sum = self._reward_sum + np.mean(reward_)
        return reward_
        
    
    def getEvaluationData(self):
        return self._reward_sum
    
    def hasNotFallen(self, exp):
        """
            Returns True when the agent is still going (not end of episode)
            return false when the agent has fallen (end of episode)
        """
        falls = exp.getEnvironment().agentHasFallenMultiAgent()
        # print ("falls: ", falls)
        falls_ = [[not fall] for fall in falls]
        # print ("Not falls: ", falls_)
        return falls_
        
    def updateActor(self, sim, action_):
        
        if (self._llc_policy is None):
            sim.updateLLCAction(action_)
        else:
            llc_state = sim.getLLCState()
            llc_state = np.array(llc_state)
            
            for i in range(len(action_)):
                action__ = np.array([action_[i][4], action_[i][0], 0.0, action_[i][1], action_[i][2], 0.0, action_[i][3]])
                llc_state[i][-7:] = action__
            
            llc_action = self._llc_policy.predict(llc_state)
            sim.updateLLCAction(llc_action)
        sim.update()
        if (self._settings["shouldRender"]):
            sim.display()
        