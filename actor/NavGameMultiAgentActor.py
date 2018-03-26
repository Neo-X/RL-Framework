import sys
import math
from actor.NavGameActor import NavGameActor
import numpy as np
from model.ModelUtil import reward_smoother, clampActionWarn

class NavGameMultiAgentActor(NavGameActor):
    
    def __init__(self, discrete_actions, experience):
        super(NavGameMultiAgentActor,self).__init__(discrete_actions, experience)
        
    
    # @profile(precision=5)
    def actContinuous(self, exp, action_, bootstrapping=False):
        # Actor should be FIRST here
        # print ("Action: " + str(action_))
        # if (settings["clamp_actions_to_stay_inside_bounds"] or (settings['penalize_actions_outside_bounds'])):
        # (action_, outside_bounds) = clampActionWarn(action_, self._action_bounds)
        #     if (settings['clamp_actions_to_stay_inside_bounds']):
        #         action_ = action__
        reward = exp.getEnvironment().actContinuous(action_, bootstrapping=bootstrapping)
        
        self._reward_sum = self._reward_sum + np.mean(reward)
        return reward
    
