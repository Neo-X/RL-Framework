import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
sys.path.append("../characterSimAdapter/")
from model.ModelUtil import *

from model.AgentInterface import AgentInterface

class ForwardDynamicsSimulator(AgentInterface):
    
    def __init__(self, state_length, action_length, state_bounds, action_bounds, actor, exp, settings):
        import characterSim
        super(ForwardDynamicsSimulator,self).__init__(state_length, action_length, state_bounds, action_bounds, 0, settings)
        self._exp = exp # Only used to pull some data from
        self._reward=0
        
        c = characterSim.Configuration(str(settings['forwardDynamics_config_file']))
        # c = characterSim.Configuration("../data/epsilon0Config.ini")
        
        # this is the process that selects which game to play
        sim = characterSim.Experiment(c)
        self._actor = actor
        self._sim = sim # The real simulator that is used for predictions
        
    def setActor(self, actor):
        self._actor = actor
    def setEnvironment(self, exp):
        self._exp = exp
    """    
    def setEnvironment(self, sim):
        self._sim = sim # The real simulator that is used for predictions
       """ 
    def train(self, states, actions,  result_states):
        return 0
    
    def bellman_error(self, state, action, result_state):
        return 0

    def initEpoch(self, exp):
        self._sim.getActor().initEpoch()
        self._sim.getEnvironment().clear()
        for anchor in range(self._exp.getEnvironment().numAnchors()):
            # print (_anchor)
            anchor_ = self._exp.getEnvironment().getAnchor(anchor)
            self._sim.getEnvironment().addAnchor(anchor_.getX(), anchor_.getY(), anchor_.getZ())
        self._sim.getEnvironment().initEpoch()

    def predict(self, state, action):
        # state = norm_state(state, self._state_bounds)
        # action = norm_action(action, self._action_bounds)
        # print ("Action: " + str(action))
        # print ("State: " + str(state._id))
        # self._exp.getEnvironment().setState(state)
        # current_state = self._exp._exp.getEnvironment().getSimInterface().getController().getControllerStateVector()
        c_state = self._sim.getEnvironment().getState()
        reward = self._actor.actContinuous(self._sim,action)
        # print ("State: " + str(state.getParams()))
        state_ = self._sim.getState()
        # print ("State: " + str.(state))
        # restore previous state
        # self._exp._exp.getEnvironment().getSimInterface().getController().setControllerStateVector(current_state)
        self._sim.getEnvironment().setState(c_state)
        # print ("State: " + str(state))
        return state_
    
    def _predict(self, state__c, action):
        # state = norm_state(state, self._state_bounds)
        # action = norm_action(action, self._action_bounds)
        # print ("Action: " + str(action))
        # print ("State: " + str(state._id))
        self._sim.getEnvironment().setState(state__c)
        # current_state = self._exp._exp.getEnvironment().getSimInterface().getController().getControllerStateVector()
        # c_state = self._sim.getEnvironment().getState()
        reward = self._actor.actContinuous(self._sim,action)
        # print ("State: " + str(state.getParams()))
        state__ = self._sim.getEnvironment().getState()
        # print ("State: " + str(state))
        # restore previous state
        # self._exp._exp.getEnvironment().getSimInterface().getController().setControllerStateVector(current_state)
        self._sim.getEnvironment().setState(state__c)
        # print ("State: " + str(state))
        return state__
