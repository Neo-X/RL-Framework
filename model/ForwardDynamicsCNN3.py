import theano
from theano import tensor as T
import numpy as np
import lasagne
from lasagne import layers
from lasagne import init
from lasagne import nonlinearities
from lasagne.utils import as_tuple
import sys
sys.path.append('../')
from model.ModelUtil import *


# For debugging
# theano.config.mode='FAST_COMPILE'
from model.ModelInterface import ModelInterface
from model.ForwardDynamicsCNN import *

class ForwardDynamicsCNN3(ModelInterface):
    
    def __init__(self, state_length, action_length, state_bounds, action_bounds, settings_):
        from lasagne.layers.dnn import *
        super(ForwardDynamicsCNN3,self).__init__(state_length, action_length, state_bounds, action_bounds, 0, settings_)
        
                # data types for model
        self._State = T.matrix("State")
        self._State.tag.test_value = np.random.rand(self._batch_size, self._state_length)
        self._ResultState = T.matrix("ResultState")
        self._ResultState.tag.test_value = np.random.rand(self._batch_size, self._state_length)
        self._Action = T.matrix("Action")
        self._Action.tag.test_value = np.random.rand(self._batch_size, self._action_length)
        
        # self._b_o = init_b_weights((n_out,))
        # networkAct = lasagne.layers.InputLayer((None, 1, 1, self._state_length), self._State)
        inputLayerState = lasagne.layers.InputLayer((None, self._state_length), self._State)
        inputLayerState = lasagne.layers.ReshapeLayer(inputLayerState, (-1, 1, 1, self._state_length))
        inputLayerAction = lasagne.layers.InputLayer((None, self._action_length), self._Action)
        networkAct = lasagne.layers.dnn.Conv2DDNNLayer(
            inputLayerState, num_filters=32, filter_size=(1,8),
            stride=(1,1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        print ("Network Shape:", lasagne.layers.get_output_shape(networkAct))
        networkAct = lasagne.layers.dnn.MaxPool2DDNNLayer(networkAct, pool_size=(1,3))
        networkAct = lasagne.layers.dnn.Conv2DDNNLayer(
            networkAct, num_filters=16, filter_size=(1,4),
            stride=(1,1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        print ("Network Shape:", lasagne.layers.get_output_shape(networkAct))
        networkAct = lasagne.layers.dnn.MaxPool2DDNNLayer(networkAct, pool_size=(1,3))
        
        self._actor_task_part = networkAct
        """ 
        networkAct = lasagne.layers.Conv1DLayer(
            networkAct, num_filters=32, filter_size=4,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        
        
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        """
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=64,
                nonlinearity=lasagne.nonlinearities.rectify)
        # networkAct = lasagne.layers.ReshapeLayer(networkAct, (-1, 99))
        # networkAct = lasagne.layers.FlattenLayer(networkAct, 2)
        print ("Network Shape:", lasagne.layers.get_output_shape(networkAct))
        print ("Action Network Shape:", lasagne.layers.get_output_shape(inputLayerAction))
        networkAct = lasagne.layers.ConcatLayer([networkAct, inputLayerAction], axis=1)
        print ("Network Shape:", lasagne.layers.get_output_shape(networkAct))
        
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=64,
                nonlinearity=lasagne.nonlinearities.rectify)
                
        print ("Network Shape:", lasagne.layers.get_output_shape(networkAct))
        networkAct = lasagne.layers.ReshapeLayer(networkAct, (-1, 1, 1, 64))
        
        networkAct = lasagne.layers.dnn.Conv2DDNNLayer(
            networkAct, num_filters=16, filter_size=(1,4),
            stride=(1,1),
            nonlinearity=lasagne.nonlinearities.rectify)
        networkAct = InverseLayer(networkAct, networkAct)
        
        # networkAct = Unpool2DLayer(networkAct, ds=(1,3))
        networkAct = lasagne.layers.dnn.MaxPool2DDNNLayer(networkAct, pool_size=(1,3))
        networkAct = InverseLayer(networkAct, networkAct)
        print ("Network Shape:", lasagne.layers.get_output_shape(networkAct))
        networkAct = lasagne.layers.dnn.Conv2DDNNLayer(
            networkAct, num_filters=32, filter_size=(1,8),
            stride=(1,1),
            nonlinearity=lasagne.nonlinearities.rectify)
        networkAct = InverseLayer(networkAct, networkAct)
        # networkAct = Unpool2DLayer(networkAct, ds=(1,3))
        networkAct = lasagne.layers.dnn.MaxPool2DDNNLayer(networkAct, pool_size=(1,3))
        networkAct = InverseLayer(networkAct, networkAct)
        print ("Network Shape:", lasagne.layers.get_output_shape(networkAct))
        # networkAct = lasagne.layers.ReshapeLayer(networkAct, (-1, 1, 1, 74))
        # print ("Network Shape:", lasagne.layers.get_output_shape(networkAct))
        self._actor = lasagne.layers.DenseLayer(
                networkAct, num_units=self._state_length,
                nonlinearity=lasagne.nonlinearities.linear)
        # self._b_o = init_b_weights((n_out,))
        print ("Network Shape:", lasagne.layers.get_output_shape(self._actor))
        
        # self._actor = lasagne.layers.ReshapeLayer(self._actor, (-1, 1, 1, 208))
        
          # print ("Initial W " + str(self._w_o.get_value()) )
        
        self._states_shared = theano.shared(
            np.zeros((self._batch_size, self._state_length),
                     dtype=theano.config.floatX))

        self._next_states_shared = theano.shared(
            np.zeros((self._batch_size, self._state_length),
                     dtype=theano.config.floatX))

        self._actions_shared = theano.shared(
            np.zeros((self._batch_size, self._action_length), dtype=theano.config.floatX),
            )
        
    def setActions(self, actions):
        """
            This is reshaped to work properly with a 1D convolution that likes 
            its input as (batch_size, channel, state_dimension)
            
            Parameters
        ----------
        states : a (batch_size, state_dimension) numpy array
        """
        # states = np.array(states)
        # states = np.reshape(states, (states.shape[0], 1, 1, states.shape[1]))
        self._actions_shared.set_value(actions.astype(dtype=np.float32))
        # self._actions_shared.set_value(actions.get_value())
        
    def setStates(self, states):
        """
            This is reshaped to work properly with a 1D convolution that likes 
            its input as (batch_size, channel, state_dimension)
            
            Parameters
        ----------
        states : a (batch_size, state_dimension) numpy array
        """
        # states = np.array(states)
        # states = np.reshape(states, (states.shape[0], 1, 1, states.shape[1]))
        self._states_shared.set_value(states.astype(dtype=np.float32))
        # self._states_shared.set_value(states.get_value())
        
    def setResultStates(self, resultStates):
        """
            This is reshaped to work properly with a 1D convolution that likes 
            its input as (batch_size, channel, state_dimension)
            
            Parameters
        ----------
        resultStates : a (batch_size, state_dimension) numpy array
        """
        # resultStates = np.array(resultStates)
        # resultStates = np.reshape(resultStates, (resultStates.shape[0], 1, 1, resultStates.shape[1]))
        self._next_states_shared.set_value(resultStates.astype(dtype=np.float32))
        # self._next_states_shared.set_value(resultStates.get_value())
