import sys
import math
from actor.ActorInterface import ActorInterface
import numpy as np
from model.ModelUtil import clampAction 
from model.ModelUtil import _scale_reward 
from model.ModelUtil import randomExporation

class SimbiconActor(ActorInterface):
    
    def __init__(self, settings_, experience):
        super(SimbiconActor,self).__init__(settings_, experience)
        self._target_vel_weight=self._settings["target_velocity_decay"]
        self._target_vel = self._settings["target_velocity"]
        self._target_lean = 0
        self._target_torque = 0
        self._target_root_height = 1.02
        
    
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
        pitch_sum = float(0)
        position_sum = float(0)
        while (not exp.getEnvironment().needUpdatedAction() or (steps_ == 0)):
            exp.getEnvironment().update()
            simData = exp.getEnvironment().getActor().getSimData()
            # print ("avgSpeed: ", simData.avgSpeed)
            vel_sum += simData.avgSpeed
            torque_sum += simData.avgTorque
            
            orientation = exp.getEnvironment().getActor().getStateEuler()[3:][:3]
            pitch_sum += orientation[0]
            
            position_root = exp.getEnvironment().getActor().getStateEuler()[0:][:3]
            position_sum += position_root[1]
            
            steps_ += 1
        averageSpeed = vel_sum / steps_
        averageTorque = torque_sum / steps_
        averagePitch = pitch_sum/ steps_
        averagePosition = position_sum / steps_
             
        # averageSpeed = exp.getEnvironment().act(action_)
        # print ("averageSpeed: ", averageSpeed)
        # if (averageSpeed < 0.0):
        #     return 0.0
        if (exp.getEnvironment().agentHasFallen()):
            return 0
        
        # orientation = exp.getEnvironment().getActor().getStateEuler()[3:][:3]
        # position_root = exp.getEnvironment().getActor().getStateEuler()[0:][:3]
        # print ("Pos: ", position_root)
        # print ("Orientation: ", orientation)
        ## Reward for going the desired velocity
        vel_diff = self._target_vel - averageSpeed
        if ( self._settings["use_parameterized_control"] ):
            vel_bounds = self._settings['controller_parameter_settings']['velocity_bounds']
            vel_diff = _scale_reward([vel_diff], vel_bounds)[0]
        vel_reward = math.exp((vel_diff*vel_diff)*self._target_vel_weight)
        ## Rewarded for using less torque
        torque_diff = averageTorque - self._target_torque
        torque_reward = math.exp((torque_diff*torque_diff)*self._target_vel_weight)
        ## Rewarded for keeping the characters torso upright
        lean_diff = averagePitch - self._target_lean
        if ( self._settings["use_parameterized_control"] ):
            root_pitch_bounds = self._settings['controller_parameter_settings']['root_pitch_bounds']
            lean_diff = _scale_reward([lean_diff], root_pitch_bounds)[0]
        lean_reward = math.exp((lean_diff*lean_diff)*self._target_vel_weight)
        ## Rewarded for keeping the y height of the root at a specific height 
        root_height_diff = (self._target_root_height - averagePosition)
        if ( self._settings["use_parameterized_control"] ):
            root_height_bounds = self._settings['controller_parameter_settings']['root_height_bounds']
            root_height_diff = _scale_reward([root_height_diff], root_height_bounds)[0]
        root_height_reward = math.exp((root_height_diff * root_height_diff) * self._target_vel_weight)
        # print ("vel reward: ", vel_reward, " torque reward: ", torque_reward )
        reward = ( 
                  (vel_reward * 0.3) +
                  (torque_reward * 0.05) +
                  (lean_reward * 0.3) + 
                  ((root_height_reward) * 0.3)
                  )# optimal is 0
        
        self._reward_sum = self._reward_sum + reward
        if ( self._settings["use_parameterized_control"] ):
            self.changeParameters()
        return reward
    
    def changeParameters(self):
        """
            Slowly modifies the parameters during training
        """
        move_scale = 0.1
        ## Can change at most by +-move_scale between each action This does not seem to work as well = 0.1
        # r = ((r - 0.5) * 2.0) * move_scale
        vel_bounds = self._settings['controller_parameter_settings']['velocity_bounds']
        self._target_vel = randomExporation(move_scale, [self._target_vel], vel_bounds)[0]
        self._target_vel = clampAction([self._target_vel], vel_bounds)[0]
        
        root_height_bounds = self._settings['controller_parameter_settings']['root_height_bounds']
        self._target_root_height = randomExporation(move_scale, [self._target_root_height], root_height_bounds)[0]
        self._target_root_height = clampAction([self._target_root_height], root_height_bounds)[0]
        
        root_pitch_bounds = self._settings['controller_parameter_settings']['root_pitch_bounds']
        self._target_lean = randomExporation(move_scale, [self._target_lean], root_pitch_bounds)[0]
        self._target_lean = clampAction([self._target_lean], root_pitch_bounds)[0]
        
        # print("New target Velocity: ", self._target_vel)
        # if ( self._settings["use_parameterized_control"] )
            
    def getControlParameters(self):
        return [self._target_vel, self._target_root_height, self._target_lean]
        
    
    def getEvaluationData(self):
        return self._reward_sum
    
    def hasNotFallen(self, exp):
        if ( exp.getEnvironment().agentHasFallen() ) :
            return 0
        else:
            return 1
        # return not exp.getEnvironment().agentHasFallen()
        
