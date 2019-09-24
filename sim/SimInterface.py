"""
"""
import numpy as np
import math

from actor.DoNothingActor import DoNothingActor
from model.ModelUtil import checkDataIsValid
# import scipy.integrate as integrate
# import matplotlib.animation as animation


class SimInterface(object):

    def __init__(self, exp, settings_):
        #------------------------------------------------------------
        # set up initial state
        # super(BallGame1DChoiceState,self).__init__()
        self._exp = exp
        self._settings = settings_
        self._mv = None
        self._actor = DoNothingActor(settings_=settings_, experience=None)
        
    def getSettings(self):
        return self._settings
        
    def getEnvironment(self):
        return self._exp
    
    def endOfEpoch(self):
        # return ( self.getEnvironment().endOfEpoch() and (not checkDataIsValid(self.getState())) )
        return ( self.getEnvironment().endOfEpoch() )

    def init(self):
        self.getEnvironment().init()
            
    def initEpoch(self):
        self.getEnvironment().initEpoch()
        
        # while not checkDataIsValid(self.getState()):
        #     self.getEnvironment().initEpoch()
    
    def addSufficientStats(self, state):
        state = np.concatenate((state, 
                                 self.getActor()._state_mean, 
                                 self.getActor()._state_var,
                                 [[self.getActor()._count]]), axis=-1)
        # return state[:,:114]
        return state
    
    def generateValidation(self, data, epoch):
        pass
    
    def generateEnvironmentSample(self):
        pass
    
    def getEvaluationData(self):
        pass
    
    def getActor(self):
        return self._actor
    
    def setActor(self, actor):
        self._actor = actor
    
    def finish(self):
        self._exp.finish()
    
    def getState(self):
        """
            I like the state in this shape (1, state_length)
        """
        state_ = self._exp.getState()
        state = np.array(state_)
        state = np.reshape(state, (1, len(state_)))
        return state
    
    def setState(self, st):
        pass
        
    def setTargetChoice(self, i):
        # need to find which target corresponds to this bin.
        _loc = np.linspace(-self._range, self._range, self._granularity)[i]
        min_dist = 100000000.0
        _choice = -1
        for i in range(int(self._choices)):
            _target_loc = self._targets[0][i][1]
            _tmp_dist = math.fabs(_target_loc - _loc)
            if ( _tmp_dist < min_dist):
                _choice = i
                min_dist = _tmp_dist
        self._target_choice = i
        self._target = self._targets[0][i]
        
    def getStateFromSimState(self, simState):
        """
            Converts a detailed simulation state to a state better suited for learning
        """
        pass
    
    def getSimState(self):
        """
            Gets a more detailed state that can be used to re-initilize the state of the character back to this state later.
        """
        pass
    
    def setSimState(self, state_):
        """
            Sets the state of the simulation to the given state
        """
        pass
    
    def updateViz(self, actor, agent, directory, p=1.0):
        """
            Maybe the sim has some cool visualization of the policy or something.
            This will update that visualization
        """
        pass
    
    def setRandomSeed(self, seed):
        """
            Set the random seed for the simulator
            This is helpful if you are running many simulations in parallel you don't
            want them to be producing the same results if they all init their random number 
            generator the same.
        """
        pass
        
    def computeImitationReward(self, reward_func):
        """
            Uses a learned imitation based reward function to
            compute the reward in the simulation 
        """
        return self.getEnvironment().computeImitationReward(reward_func)
    
    def setMovieWriter(self, mw):
        """
            Set an object that can be used to record frames for writing out a video of the simulation
        """
        self._mv = mw
    def getMovieWriter(self):
        return self._mv
        
    def movieWriterSupport(self):
        return False
