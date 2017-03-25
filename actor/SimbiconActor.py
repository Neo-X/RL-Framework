import sys
import math
from actor.ActorInterface import ActorInterface
import numpy as np

class SimbiconActor(ActorInterface):
    
    def __init__(self, settings_, experience):
        super(SimbiconActor,self).__init__(settings_, experience)
        self._target_vel_weight=self._settings["target_velocity_decay"]
        self._target_vel = self._settings["target_velocity"]
        self._target_lean = 0
        self._target_torque = 0
        
    
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
        ## Need to make sure this is an vector of doubles
        action_ = np.array(action_, dtype='float64')
        
        exp.getEnvironment().updateAction(action_)
        steps_ = 0
        vel_sum= float(0)
        torque_sum= float(0)
        while (not exp.getEnvironment().needUpdatedAction() or (steps_ == 0)):
            exp.getEnvironment().update()
            simData = exp.getEnvironment().getActor().getSimData()
            # print ("avgSpeed: ", simData.avgSpeed)
            vel_sum += simData.avgSpeed
            torque_sum += simData.avgTorque
            steps_ += 1
        averageSpeed = vel_sum / steps_
        averageTorque = torque_sum / steps_
             
        # averageSpeed = exp.getEnvironment().act(action_)
        # print ("averageSpeed: ", averageSpeed)
        # if (averageSpeed < 0.0):
        #     return 0.0
        if (exp.getEnvironment().agentHasFallen()):
            return 0
        
        orientation = exp.getEnvironment().getActor().getStateEuler()[3:][:3]
        position_root = exp.getEnvironment().getActor().getStateEuler()[0:][:3]
        # print ("Pos: ", position_root)
        print ("Orientation: ", orientation)
        lean_diff = orientation[1] - self._target_lean
        vel_dif = self._target_vel - averageSpeed
        vel_reward = math.exp((vel_dif*vel_dif)*self._target_vel_weight)
        torque_diff = averageTorque - self._target_torque
        torque_reward = math.exp((torque_diff*torque_diff)*self._target_vel_weight)
        lean_reward = math.exp((lean_diff*lean_diff)*self._target_vel_weight)
        # print ("vel reward: ", vel_reward, " torque reward: ", torque_reward )
        reward = ( 
                  (vel_reward * 0.8) +
                  (torque_reward * 0.1) +
                  (lean_reward * 0.1) + 
                  ((position_root[1] - 1.0) )
                  )# optimal is 0
        
        self._reward_sum = self._reward_sum + reward
        if ( self._settings["use_parameterized_control"] ):
            self.changeParameters()
        return reward
    
    def changeParameters(self):
        """
            Slowly modifies the parameters during training
        """
        r = np.random.random(1)[0]
        ## Can change at most by +-move_scale between each action
        move_scale = 0.2 
        r = ((r - 0.5) * 2.0) * move_scale
        vel_bounds = [0.5, 1.5]
        self._target_vel += r
        if ( self._target_vel < vel_bounds[0] ):
            self._target_vel = vel_bounds[0]
        if ( self._target_vel > vel_bounds[1] ):
            self._target_vel = vel_bounds[1]
            
    def getControlParameters(self):
        return [self._target_vel]
        
    
    def getEvaluationData(self):
        return self._reward_sum
    
    def hasNotFallen(self, exp):
        if ( exp.getEnvironment().agentHasFallen() ) :
            return 0
        else:
            return 1
        # return not exp.getEnvironment().agentHasFallen()
        
