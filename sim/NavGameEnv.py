"""
"""
import numpy as np
import math
from sim.SimInterface import SimInterface 

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
        ## This is a sampled grid in 2D
        (X,Y) = self.getEnvironment().getStateSamples()
        for x_,y_ in zip(X,Y):
            for x,y in zip(x_,y_):
                ## Policy action
                state_ = [[x,y]]
                action = agent.predict([[x,y]])
                U.append(action[0])
                V.append(action[1])
                v = agent.q_value([[x,y]])
                Q.append(v)
                action_ = getMBAEAction(agent.getForwardDynamics(), agent.getPolicy(), state_)
                U_mbae.append(action_[0])
                V_mbae.append(action_[1])
        self.getEnvironment().updatePolicy(U, V, Q)
        self.getEnvironment().updateMBAE(U_mbae, V_mbae, Q)
        self.getEnvironment().saveVisual(directory+"/navAgent")
        
