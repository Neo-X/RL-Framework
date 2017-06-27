"""
"""
import numpy as np
import math
from sim.SimInterface import SimInterface 
import copy 
# import scipy.integrate as integrate
# import matplotlib.animation as animation

from model.ModelUtil import getOptimalAction, getMBAEAction


class NavGameEnv(SimInterface):

    def __init__(self, exp, settings):
        #------------------------------------------------------------
        # set up initial state
        super(NavGameEnv,self).__init__(exp, settings)

    def generateValidation(self, data, epoch):
        self.getEnvironment().generateValidationEnvironmentSample(epoch)
    
    def generateEnvironmentSample(self):
        self.getEnvironment().generateEnvironmentSample()
        
    def getEvaluationData(self):
        return self.getEnvironment().getEvaluationData()
    
    def getActor(self):
        return self._exp
    
    def finish(self):
        self._exp.finish()
    
    def getState(self):
        # state = np.array(self._exp.getState())
        state_ = np.array(self._exp.getState())
        state = np.array(state_)
        state = np.reshape(state, (-1, len(state_)))
        
        return state
    
    def setState(self, st):
        self._agent = st
        self._box.state[0,0] = st[0]
        self._box.state[0,1] = st[1]
        
    def visualizeNextState(self, next_state_, action):
        _t_length = self.getEnvironment()._game_settings['num_terrain_samples']
        terrain = next_state_[:_t_length]
        terrain_dx = next_state_[_t_length]
        terrain_dy = next_state_[_t_length+1]
        character_features = next_state_[_t_length+2:]
        self.getEnvironment().visualizeNextState(terrain, action, terrain_dx)  
    
    def updateViz(self, actor, agent, directory):
        U = []
        V = []
        Q = []
        U_mbae = []
        V_mbae = []
        R_mbae = []
        ## This is a sampled grid in 2D
        (X,Y) = self.getEnvironment().getStateSamples()
        for x_,y_ in zip(X,Y):
            for x,y in zip(x_,y_):
                ## Policy action
                state_ = np.array([[x,y]])
                action1 = agent.predict([[x,y]])
                action1_cp = copy.deepcopy(action1)
                # action1 = getOptimalAction(agent.getForwardDynamics(), agent.getPolicy(), state_)
                ## normalize
                action1 = action1/(np.sqrt((action1*action1).sum(axis=0)))
                U.append(action1[0])
                V.append(action1[1])
                v = agent.q_value(state_)
                Q.append(v)
                action_ = getOptimalAction(agent.getForwardDynamics(), agent.getPolicy(), state_)
                # action_ = getMBAEAction(agent.getForwardDynamics(), agent.getPolicy(), state_)
                ### How to change this action...
                action_ = action_ - action1_cp
                # next_state = agent.getForwardDynamics().predict(state_, action1)
                # print ("next_state: ", next_state)
                # action_ = next_state - state_[0]
                action_ = action_/(np.sqrt((action_*action_).sum(axis=0)))
                # action_ = action_ - action1
                U_mbae.append(action_[0])
                V_mbae.append(action_[1])
                r = agent.getForwardDynamics().predict_reward(state_, np.array(action1_cp))
                # print ("Predicted reward: ", r)
                R_mbae.append(r)
        self.getEnvironment().updatePolicy(U, V, Q)
        self.getEnvironment().updateMBAE(U_mbae, V_mbae, R_mbae)
        self.getEnvironment().saveVisual(directory+"/navAgent")
        
