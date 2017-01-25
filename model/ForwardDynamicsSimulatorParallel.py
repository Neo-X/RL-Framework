import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
sys.path.append("../characterSimAdapter/")
from model.ModelUtil import *

from model.AgentInterface import AgentInterface
from model.ForwardDynamicsSimulator import ForwardDynamicsSimulator
from multiprocessing import Queue, Process

class ForwardDynamicsSimulatorProcess(Process):

    def __init__(self, state_length, action_length, state_bounds, action_bounds, actor, exp, settings, input_state_queue,
                 outpout_state_queue):
        import characterSim
        super(ForwardDynamicsSimulatorProcess, self).__init__()
        # super(ForwardDynamicsSimulatorProcess,self).__init__(state_length, action_length, state_bounds, action_bounds, 0, settings)
        self._input_queue= input_state_queue
        self._output_state_queue = outpout_state_queue
        # self._exp = exp # Only used to pull some data from
        self._c = characterSim.Configuration(str(settings['forwardDynamics_config_file']))
        # c = characterSim.Configuration("../data/epsilon0Config.ini")
        
        # this is the process that selects which game to play
        # sim = characterSim.Experiment(self._c)
        
        self._actor = actor
        
        # self._sim = sim # The real simulator that is used for predictions
    
    def setActor(self, actor):
        self._actor = actor
    def setEnvironment(self, sim):
        self._sim = sim # The real simulator that is used for predictions
        
    def run(self):
        import characterSim
        sim = characterSim.Experiment(self._c)
        self._sim = sim # The real simulator that is used for predictions
        print ('ForwardDynamicsSimulatorProcess started')
        # do some initialization here
        step_ = 0
        while True:
            tmp = self._input_queue.get()
            if tmp == None:
                break
            elif (tmp[0] == 'init'):
                print ("Initilizing environment")
                self._sim.getActor().initEpoch()
                self._sim.getEnvironment().clear()
                for anchor_ in tmp[1]:
                    # print (_anchor)
                    # anchor_ = self._exp.getEnvironment().getAnchor(anchor)
                    self._sim.getEnvironment().addAnchor(anchor_[0], anchor_[1], anchor_[2])
                self._sim.getEnvironment().initEpoch()
                print ("Number of anchors is " + str(self._sim.getEnvironment().numAnchors()))
                
            else:
                state_c = tmp
                action = state_c[2]
                # print ("Sampling State:" + str(state_c))
                state_c = characterSim.State(state_c[0], state_c[1])
                # print ("State: " + str(state_c) + " sim " + str(self._sim.getEnvironment()))
                self._sim.getEnvironment().setState(state_c)
                # print ("State: " + str(state_c) + " Action: " + str(action))
                reward = self._actor.actContinuous(self._sim,action)
                # print ("State: " + str(state.getParams()))
                state_ = self._sim.getEnvironment().getState()
                self._sim.getEnvironment().setState(state_c)
                # characterSim.State(current_state_copy.getID(), current_state_copy.getParams())
                self._output_state_queue.put([state_.getID(), state_.getParams()])
            

class ForwardDynamicsSimulatorParallel(ForwardDynamicsSimulator):
    
    def __init__(self, state_length, action_length, state_bounds, action_bounds, actor, exp, settings):

        super(ForwardDynamicsSimulatorParallel,self).__init__(state_length, action_length, state_bounds, action_bounds, actor, exp, settings)
        self._exp = exp # Only used to pull some data from
        self._reward=0
        
        self._output_state_queue = Queue(1)
        self._input_state_queue = Queue(1)
        
    def init(self, state_length, action_length, state_bounds, action_bounds, actor, exp, settings):
        
        self._worker = ForwardDynamicsSimulatorProcess(state_length, action_length, state_bounds, action_bounds, actor, exp, settings,
                                                       self._output_state_queue, self._input_state_queue)
        
        self._worker.start()
        
   
    def initEpoch(self, exp):
        anchors=[]
        for anchor in range(self._exp.getEnvironment().numAnchors()):
            # print (_anchor)
            anchor_ = self._exp.getEnvironment().getAnchor(anchor)
            anchors.append([anchor_.getX(), anchor_.getY(), anchor_.getZ()])
        # print (anchors)
        self._output_state_queue.put(['init',anchors]) 

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
        import characterSim
        self._output_state_queue.put([state__c.getID(), state__c.getParams(), action])
        state__ = self._input_state_queue.get()
        state__ = characterSim.State(state__[0], state__[1])
        return state__
    
