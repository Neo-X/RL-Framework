"""
"""
import numpy as np
import math
from sim.SimInterface import SimInterface
import sys
sys.path.append("../simbiconAdapter/")
from actor.DoNothingActor import DoNothingActor
# import scipy.integrate as integrate
# import matplotlib.animation as animation


class MocapImitationEnv(SimInterface):

    def __init__(self, exp, settings):
        #------------------------------------------------------------
        # set up initial state
        super(MocapImitationEnv,self).__init__(exp, settings)
        self._action_dimension=3
        self._range = 5.0

    def initEpoch(self):
        self.getEnvironment().initEpoch()
        # self.getAgent().initEpoch()
        
    def getEnvironment(self):
        return self._exp
    
    def getEvaluationData(self):
        return self.getEnvironment().getEvaluationData()
    
    def generateValidation(self, data, epoch):
        """
            Do nothing for now
        """
        pass
        # print (("Training on validation set: ", epoch, " Data: ", data))
        # self.getEnvironment().clear()
        # print (("Done clear"))
        # for i in range(len(data)):
            # data_ = data[i]
            # print (("Adding anchor: ", data_, " index ", i))
            # self.getEnvironment().addAnchor(data_[0], data_[1], data_[2])
            
        # print (("Done adding anchors"))

    def generateValidationEnvironmentSample(self, epoch):
        pass
    def generateEnvironmentSample(self):
        pass
        # self._exp.getEnvironment().generateEnvironmentSample()
        
    def updateAction(self, action_):
        # print("Simbicon updating action:")
        self.getActor().updateAction(self, action_)
    
    def needUpdatedAction(self):
        return self.getEnvironment().needUpdatedAction()
        
    def update(self):
        self.getEnvironment().update()
            
    def display(self):
        pass
        # self.getEnvironment().display()

    def finish(self):
        self._exp.finish()
    
    def getState(self):
        """
            I like the state in this shape, a row
        """
        state_ = self.getEnvironment().getState()
        state = np.array(state_)
        state = np.reshape(state, (-1, len(state_)))
        if ( self._settings["use_parameterized_control"] ):
            state = np.append(state, [self.getActor().getControlParameters()], axis=1)
        return state
    
    def getControllerBackOnTrack(self):
        import characterSim
        """
            Push controller back into a good state space
        """
        pass
        
    def setTargetChoice(self, i):
        # need to find which target corresponds to this bin.
        pass
    
    
    def getStateFromSimState(self, simState):
        """
            Converts a detailed simulation state to a state better suited for learning
        """
        return self.getEnvironment().getStateFromSimState(simState)
    
    def getSimState(self):
        """
            Gets a more detailed state that can be used to re-initilize the state of the character back to this state later.
        """
        return self.getEnvironment().getSimState()
    
    def setSimState(self, state_):
        """
            Sets the state of the simulation to the given state
        """
        return self.getEnvironment().setSimState(state_)
        
