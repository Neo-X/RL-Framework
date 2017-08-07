import sys
import math
from actor.ActorInterface import ActorInterface
import numpy as np
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
        # Actor should be FIRST here
        # print ("Action: " + str(action_))
        # dist = exp.getEnvironment().step(action_, bootstrapping=bootstrapping)
        observation, reward, done, info = env.step(action)
        exp._end_of_episode = done
        exp._previous_observation = observation
        self._reward_sum = self._reward_sum + reward
        if (not done):
            # vel_dif = np.abs(self._target_vel - dist)
            # reward = math.exp((vel_dif*vel_dif)*self._target_vel_weight) # optimal is 0
            # reward = reward_smoother(vel_dif, self._settings, self._target_vel_weight)
            self._reward_sum = self._reward_sum + reward
            return reward
        else:
            return 0.0
        
    
    def getEvaluationData(self):
        return self._reward_sum
    
    def hasNotFallen(self, exp):
        """
            Returns True when the agent is still going (not end of episode)
            return false when the agent has fallen (end of episode)
        """
        if ( not exp._end_of_episode ):
            return 0
        else:
            return 1