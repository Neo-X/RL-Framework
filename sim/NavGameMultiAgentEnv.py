"""
"""
import numpy as np
import math
from sim.NavGameEnv import NavGameEnv 
import copy 
# import scipy.integrate as integrate
# import matplotlib.animation as animation

from model.ModelUtil import getOptimalAction, getMBAEAction


class NavGameMultiAgentEnv(NavGameEnv):

    def __init__(self, exp, settings):
        #------------------------------------------------------------
        # set up initial state
        super(NavGameMultiAgentEnv,self).__init__(exp, settings)

    def getState(self):
        # state = np.array(self._exp.getState())
        state_ = np.array(self._exp.getState())
        state = np.array(state_)
        # state = np.reshape(state, (-1, state_.shape[0]))
        
        return state
    
