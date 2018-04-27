import sys
import math
from actor.ActorInterface import ActorInterface
from model.ModelUtil import randomExporation, randomUniformExporation, reward_smoother, clampAction, clampActionWarn
import numpy as np
import copy

class DoNothingActor(ActorInterface):
    
    def __init__(self, settings_, experience):
        super(DoNothingActor,self).__init__(settings_, experience)
        
    
    def updateAction(self, sim, action_):
        action_ = action_[0]
        action_ = np.array(action_, dtype='float64')
        # sim.getEnvironment().updateAction(action_)
    
    # @profile(precision=5)
    def act(self, exp, action_, bootstrapping=False):
        samp = self.getActionParams(action_)
        
        reward = self.actContinuous(exp, samp, bootstrapping=bootstrapping)
        
        return reward
    
    # @profile(precision=5)
    def actContinuous(self, exp, action_, bootstrapping=False):
        # Actor should be FIRST here
        # print ("Action: ", repr(action_))
        # action_ = copy.deepcopy(action_)
        action_ = action_[0]
        action_ = np.array(action_, dtype='float64')
        # (action_, outside_bounds) = clampActionWarn(action_, self._action_bounds)
        
        reward = 1.2

        self._reward_sum = self._reward_sum + reward
        # print("Reward Sum: ", self._reward_sum)
        return reward
    
    def changeParameters(self):
        """
            Slowly modifies the parameters during training
        """
        
            
    def getControlParameters(self):
        # return [self._target_vel, self._target_root_height, self._target_lean, self._target_hand_pos]
        return [self._target_vel]
        
    def initEpoch(self):
        super(GapGame2DActor,self).initEpoch()
        self._reward_sum = 0
    
    def getEvaluationData(self):
        return self._reward_sum
    
    def setTargetVelocity(self, exp, target_vel):
        self._target_vel = target_vel
    
    def hasNotFallen(self, exp):
        return 1