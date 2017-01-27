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


class TerrainRLFlatEnv(TerrainRLEnv):

    def __init__(self, exp):
        #------------------------------------------------------------
        # set up initial state
        super(TerrainRLFlatEnv,self).__init__(exp)

    
    def getState(self):
        """
            Want just the character state at the end.
        """
        state = np.array(self.getEnvironment().getState())[200:]
        return state
    