"""
"""
import numpy as np
import math
from sim.TerrainRLEnv import TerrainRLEnv
import sys
from actor.DoNothingActor import DoNothingActor
# sys.path.append("../simbiconAdapter/")

# import scipy.integrate as integrate
# import matplotlib.animation as animation


class TerrainRLImitateEnv(TerrainRLEnv):

    def __init__(self, exp):
        #------------------------------------------------------------
        # set up initial state
        super(TerrainRLImitateEnv,self).__init__(exp)

    
    def getState(self):
        """
            Want just the character state at the end.
        """
        state_ = self.getEnvironment().getState()
        # print ("state_: ", state_)
        # state = np.array(state_)[200:]
        # state = np.reshape(state, (-1, len(state_)-200))
        state = np.array(state_)
        state = np.reshape(state, (-1, len(state_)))
        return state
    