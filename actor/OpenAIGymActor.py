import sys
import math
from actor.ActorInterface import ActorInterface
from model.ModelUtil import reward_smoother

class OpenAIGymActor(ActorInterface):
    
    def __init__(self, discrete_actions, experience):
        super(OpenAIGymActor,self).__init__(discrete_actions, experience)
        self._target_vel_weight=self._settings["target_velocity_decay"]
        self._target_vel = self._settings["target_velocity"]
        # self._target_vel = self._settings["target_velocity"]
        self._end_of_episode=False
    
    # @profile(precision=5)
    def act(self, exp, action_, bootstrapping=False):
        samp = self.getActionParams(action_)
        
        reward = self.actContinuous(exp, samp, bootstrapping=bootstrapping)
        
        return reward
    
    # @profile(precision=5)
    def actContinuous(self, exp, action_, bootstrapping=False):
        import numpy as np
        # Actor should be FIRST here
        # print ("Action: " + str(action_))
        # dist = exp.getEnvironment().step(action_, bootstrapping=bootstrapping)
        reward = exp.step(action_)
        self._reward_sum = self._reward_sum + np.mean(reward)
        return reward
        
    def updateAction(self, sim, action_):
        import numpy as np
        action_ = np.array(action_, dtype='float64')
        sim.getEnvironment().updateAction(action_)
    
    def getEvaluationData(self):
        return self._reward_sum
    
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