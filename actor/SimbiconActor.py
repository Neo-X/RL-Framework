import sys
import math
from actor.ActorInterface import ActorInterface
import numpy as np

class SimbiconActor(ActorInterface):
    
    def __init__(self, settings_, experience):
        super(SimbiconActor,self).__init__(settings_, experience)
        self._target_vel_weight=self._settings["target_velocity_decay"]
        self._target_vel = self._settings["target_velocity"]
        
    # @profile(precision=5)
    def act(self, exp, action_, bootstrapping=False):
        samp = self.getActionParams(action_)
        
        reward = self.actContinuous(exp, samp, bootstrapping)
        self._reward_sum = self._reward_sum + reward
        return reward
    
    # @profile(precision=5)
    def actContinuous(self, exp, action_, bootstrapping=False):
        # Actor should be FIRST here
        # print "Action: " + str(action_)
        ## Need to make sure this is an array of doubles
        action_ = np.array(action_, dtype='float64')
        averageSpeed = exp.getEnvironment().act(action_)
        # print ("averageSpeed: ", averageSpeed)
        if (averageSpeed < 0.0):
            return 0.0
        
        vel_dif = self._target_vel - averageSpeed
        reward = math.exp((vel_dif*vel_dif)*self._target_vel_weight) # optimal is 0
        self._reward_sum = self._reward_sum + reward
        return reward
    
    def getEvaluationData(self):
        return self._reward_sum
    
    def hasNotFallen(self, exp):
        if ( exp.getEnvironment().agentHasFallen() ) :
            return 0
        else:
            return 1
        # return not exp.getEnvironment().agentHasFallen()