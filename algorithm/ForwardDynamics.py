import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *


# For debugging
# theano.config.mode='FAST_COMPILE'
from algorithm.AlgorithmInterface import AlgorithmInterface

class ForwardDynamics(AlgorithmInterface):
    
    def __init__(self, model, state_length, action_length, state_bounds, action_bounds, settings_):

        super(ForwardDynamics,self).__init__(model, state_length, action_length, state_bounds, action_bounds, 0, settings_)
        self._model = model
        batch_size=32
        self._learning_rate = self.getSettings()["fd_learning_rate"]
        # data types for model
        # create a small convolutional neural network
        
        inputs_ = {
            self._model.getStateSymbolicVariable(): self._model.getStates(),
            self._model.getActionSymbolicVariable(): self._model.getActions(),
        }
        self._forward = lasagne.layers.get_output(self._model.getActorNetwork(), inputs_)
        
        # self._target = (Reward + self._discount_factor * self._q_valsB)
        self._diff = self._model.getResultStateSymbolicVariable() - self._forward
        self._loss = T.mean(T.pow(self._diff, 2),axis=1)
        self._loss = T.mean(self._loss)
        
        self._params = lasagne.layers.helper.get_all_params(self._model.getActorNetwork())
        self._givens_ = {
            self._model.getStateSymbolicVariable() : self._model.getStates(),
            self._model.getResultStateSymbolicVariable() : self._model.getResultStates(),
            self._model.getActionSymbolicVariable(): self._model.getActions(),
        }

        # SGD update
        self._updates_ = lasagne.updates.rmsprop(self._loss + (self._regularization_weight * lasagne.regularization.regularize_network_params(
                self._model.getActorNetwork(), lasagne.regularization.l2)), self._params, self._learning_rate, self._rho,
                                            self._rms_epsilon)
     
        self._train = theano.function([], [self._loss], updates=self._updates_, givens=self._givens_)
        self._forwardDynamics = theano.function([], self._forward,
                                       givens={self._model.getStateSymbolicVariable() : self._model.getStates(),
                                                self._model.getActionSymbolicVariable(): self._model.getActions()})
        self._bellman_error = theano.function(inputs=[], outputs=self._diff, allow_input_downcast=True, givens=self._givens_)
        # self._diffs = theano.function(input=[State])
        self._get_grad = theano.function([], outputs=lasagne.updates.get_or_compute_grads(self._loss, [lasagne.layers.get_all_layers(self._model.getActorNetwork())[0].input_var] + self._params), allow_input_downcast=True, givens=self._givens_)

    def getNetworkParameters(self):
        params = []
        params.append(lasagne.layers.helper.get_all_param_values(self._model.getActorNetwork()))
        return params
    
    def setNetworkParameters(self, params):
        lasagne.layers.helper.set_all_param_values(self._model.getActorNetwork(), params[0])
        
    def setData(self, states, actions, result_states):
        self._model.setStates(states)
        self._model.setResultStates(result_states)
        self._model.setActions(actions)
        
    def getGrads(self, states, actions, result_states):
        states = np.array(states, dtype=self.getSettings()['float_type'])
        actions = np.array(actions, dtype=self.getSettings()['float_type'])
        result_states = np.array(result_states, dtype=self.getSettings()['float_type'])
        self.setData(states, actions, result_states)
        return self._get_grad()
                
    def train(self, states, actions, result_states):
        self.setData(states, actions, result_states)
        # print ("Performing Critic trainning update")
        #if (( self._updates % self._weight_update_steps) == 0):
        #    self.updateTargetModel()
        self._updates += 1
        # all_paramsActA = lasagne.layers.helper.get_all_param_values(self._l_outActA)
        loss = self._train()
        # This undoes the Actor parameter updates as a result of the Critic update.
        # print (diff_)
        return loss
    
    def predict(self, state, action):
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        action = np.array([norm_action(action, self._action_bounds)], dtype=self.getSettings()['float_type'])
        self._model.setStates(state)
        self._model.setActions(action)
        state_ = scale_state(self._forwardDynamics()[0], self._state_bounds)
        return state_

    def bellman_error(self, states, actions, result_states):
        self.setData(states, actions, result_states)
        return self._bellman_error()
