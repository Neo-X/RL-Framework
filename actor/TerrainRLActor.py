import sys
import math
from actor.ActorInterface import ActorInterface

class TerrainRLActor(ActorInterface):
    
    def __init__(self, settings_, experience):
        super(TerrainRLActor,self).__init__(settings_, experience)
        self._target_vel_weight=self._settings["target_velocity_decay"]
        self._target_vel = self._settings["target_velocity"]
        self._default_action = [ 0.082500, 0.251474, 2.099796, 0.000000, 0.000000, -0.097252, -0.993935, 0.273527, 0.221481, 1.100288, -3.076833, 0.180141, -0.176967, 0.310372, -1.642646, -0.406771, 1.240827, -1.773369, -0.508333, -0.170533, -0.063421, -2.091676, -1.418455, -1.242994, -0.262842, 0.453321, -0.170533, -0.366870, -1.494344, 0.794701, -1.408623, 0.655703, 0.634434]
        self._param_mask = [    False,        True,        True,        False,        False,    
        True,        True,        True,        True,        True,        True,        True,    
        True,        True,        True,        True,        True,        True,        True,    
        False,        True,        True,        True,        True,        True,        True,    
        False,        True,        True,        True,        True,        True,        True]
        
    # @profile(precision=5)
    def act(self, exp, action_, bootstrapping=False):
        samp = self.getActionParams(action_)
        
        reward = self.actContinuous(exp, samp, bootstrapping)
        return reward
    
    # @profile(precision=5)
    def actContinuous(self, sim, action_, bootstrapping=False):
        # Actor should be FIRST here
        # print "Action: " + str(action_)
        # reward = exp.getEnvironment().act(action_)
        # mask some parameters
        action_idx=0
        action__=[]
        for i in range(len(self._default_action)): # because the use of parameters can be switched on and off.
            if (self._param_mask[i] == True):
                action__.append(action_[action_idx] )
                action_idx+=1
            else:
                action__.append(self._default_action[i])
        action_=action__
        sim.getEnvironment().act(action_)
        updates_=0
        while (not sim.getEnvironment().endOfAction() and (updates_ < 20)):
            sim.getEnvironment().update()
            updates_+=1
        reward_ = sim.getEnvironment().calcReward()   
        # print ("averageSpeed: ", averageSpeed)
        self._reward_sum = self._reward_sum + reward_
        print ("Reward: ", reward_)

        return reward_
    
    def getEvaluationData(self):
        return self._reward_sum
    
    def hasNotFallen(self, sim):
        if ( sim.getEnvironment().agentHasFallen() ) :
            return 0
        else:
            return 1
        # return not exp.getEnvironment().agentHasFallen()
        
        