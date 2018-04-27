"""
"""
import numpy as np
import math
from sim.SimInterface import SimInterface 

# import scipy.integrate as integrate
# import matplotlib.animation as animation


class DummyEnv(SimInterface):

    def __init__(self, exp, settings):
        #------------------------------------------------------------
        # set up initial state
        self._T = 0
        self._T_max = 0
        super(DummyEnv,self).__init__(exp, settings)
        
    def setMaxT(self, t):
        self._T_max = t

    def generateValidation(self, data, epoch):
        self._T = 0
    
    def generateValidationEnvironmentSample(self, epoch):
        self._T = 0
        
    def generateEnvironmentSample(self):
        self._T = 0
        
    def getEvaluationData(self):
        return 0
    
    def init(self):
        self._T = 0
        
    def initEpoch(self):
        self._T = 0

    def endOfEpoch(self):
        # return ( self.getEnvironment().endOfEpoch() and (not checkDataIsValid(self.getState())) )
        return ( self._T >= self._T_max )
        
    def updateAction(self, action_):
        # print("Simbicon updating action:")
        pass
    
    def needUpdatedAction(self):
        return True
            
    def display(self):
        pass
        # self.getEnvironment().display(
    
    def finish(self):
        self._exp.finish()
        
    def update(self):
        # self.getEnvironment().update()
        self._T = self._T + 1
    
    def getState(self):
        # state = np.array(self._exp.getState())
        state = np.array([self._T] * 11)
        state = np.reshape(state, (-1, 11))
        return state
    
    def setState(self, st):
        pass
        
    def setTargetChoice(self, i):
        pass
        
    def visualizeNextState(self, next_state_, action):
        pass
        
    def setRandomSeed(self, seed):
        """
            Set the random seed for the simulator
            This is helpful if you are running many simulations in parallel you don't
            want them to be producing the same results if they all init their random number 
            generator the same.
        """
        print ( "Setting random seed: ", seed )
        # self.getEnvironment().setRandomSeed(seed)
        
