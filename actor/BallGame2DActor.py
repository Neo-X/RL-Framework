import sys
import math
from actor.ActorInterface import ActorInterface

class BallGame2DActor(ActorInterface):
    
    def __init__(self, discrete_actions):
        super(BallGame2DActor,self).__init__(discrete_actions)
        
    
    # @profile(precision=5)
    def act(self, exp, action_, bootstrapping=False):
        samp = self.getActionParams(action_)
        
        reward = exp.getEnvironment().actContinuous(samp, bootstrapping=bootstrapping)
        return reward
    
    # @profile(precision=5)
    def actContinuous(self, exp, action_, bootstrapping=False):
        # Actor should be FIRST here
        # print "Action: " + str(action_)
        reward = exp.getEnvironment().actContinuous(action_, bootstrapping=bootstrapping)
        return reward
    
