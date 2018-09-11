import sys
import math
from actor.OpenAIGymActor import OpenAIGymActor
from model.ModelUtil import reward_smoother

class OpenAIGymActor2(OpenAIGymActor):
    
    def __init__(self, discrete_actions, experience):
        super(OpenAIGymActor2,self).__init__(discrete_actions, experience)
        
    
    def hasNotFallen(self, exp):
        """
            Returns True when the agent is still going (not end of episode)
            return false when the agent has fallen (end of episode)
        """
        if ( exp.getEnv().agentHasFallen() ):
            return 0
        else:
            return 1