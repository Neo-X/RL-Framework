import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *


# For debugging
# theano.config.mode='FAST_COMPILE'
from model.ModelInterface import ModelInterface

class ForwardDynamicsNetwork(ModelInterface):
    
    def __init__(self, state_length, action_length, state_bounds, action_bounds, settings_):

        super(ForwardDynamicsNetwork,self).__init__(state_length, action_length, state_bounds, action_bounds, 0, settings_)
        
        batch_size=32
        # data types for model
        self._State = T.matrix("State")
        self._State.tag.test_value = np.random.rand(batch_size,self._state_length)
        self._ResultState = T.matrix("ResultState")
        self._ResultState.tag.test_value = np.random.rand(batch_size,self._state_length)
        self._Action = T.matrix("Action")
        self._Action.tag.test_value = np.random.rand(batch_size, self._action_length)
        # create a small convolutional neural network
        new_state = theano.tensor.concatenate([self._State, self._Action], axis=1)
        network = lasagne.layers.InputLayer((None, self._state_length + self._action_length), new_state)
        
        # inputLayerState = lasagne.layers.InputLayer((None, self._state_length), self._State)
        # inputLayerAction = lasagne.layers.InputLayer((None, self._action_length), self._Action)
        # concatLayer = lasagne.layers.ConcatLayer([inputLayerState, inputLayerAction])
        l_hid2ActA = lasagne.layers.DenseLayer(
                network, num_units=256,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid3ActA = lasagne.layers.DenseLayer(
                l_hid2ActA, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid4ActA = lasagne.layers.DenseLayer(
                l_hid3ActA, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
        self._actor = lasagne.layers.DenseLayer(
                l_hid4ActA, num_units=self._state_length,
                nonlinearity=lasagne.nonlinearities.linear)
                # print ("Initial W " + str(self._w_o.get_value()) )
        
        self._crtic = None
        
        self._states_shared = theano.shared(
            np.zeros((batch_size, self._state_length),
                     dtype=theano.config.floatX))

        self._next_states_shared = theano.shared(
            np.zeros((batch_size, self._state_length),
                     dtype=theano.config.floatX))

        self._actions_shared = theano.shared(
            np.zeros((batch_size, self._action_length), dtype=theano.config.floatX),
            )
        
