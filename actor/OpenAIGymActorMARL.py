import sys
import math
from actor.OpenAIGymActor import OpenAIGymActor
from model.ModelUtil import reward_smoother

class OpenAIGymActorMARL(OpenAIGymActor):
    
    def __init__(self, discrete_actions, experience):
        super(OpenAIGymActor,self).__init__(discrete_actions, experience)
    
    # @profile(precision=5)
    def act(self, exp, action_, bootstrapping=False):
        samp = self.getActionParams(action_)
        
        reward = self.actContinuous(exp, samp, bootstrapping=bootstrapping)
        
        return reward
    
        # @profile(precision=5)
    def actContinuous(self, sim, action_, bootstrapping=False):
        import numpy as np
        # Actor should be FIRST here
        ob, reward, done, info  = super().step(self, action_)
       
        return ob, reward, done, info
    
        
    def updateAction(self, sim, action_):
        import numpy as np
        action_ = np.array(action_, dtype='float64')
        sim.getEnvironment().updateAction(action_)
    
    
    def hasNotFallen(self, exp):
        """
            Returns True when the agent is still going (not end of episode)
            return false when the agent has fallen (end of episode)
        """
        # falls_ = [[not e] for e in exp._fallen]
        if (type(exp._end_of_episode) is list):
            falls_ = [[not e] for e in exp._end_of_episode]
        else:
            falls_ = not exp._end_of_episode
        return falls_
        """
        if ( exp._end_of_episode ):
            return 1
        else:
            return 0
        """