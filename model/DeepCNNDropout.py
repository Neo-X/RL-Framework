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

class DeepCNNDropout(ModelInterface):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(DeepCNNDropout,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        # data types for model
        self._dropout_p=settings_['dropout_p']
        
        self._State = T.dtensor3("State")
        self._State.tag.test_value = np.random.rand(self._batch_size, 1, self._state_length)
        self._ResultState = T.dtensor3("ResultState")
        self._ResultState.tag.test_value = np.random.rand(self._batch_size, 1, self._state_length)
        self._Reward = T.col("Reward")
        self._Reward.tag.test_value = np.random.rand(self._batch_size,1)
        self._Action = T.dmatrix("Action")
        self._Action.tag.test_value = np.random.rand(self._batch_size, self._action_length)
        
        # create a small convolutional neural network
        network = lasagne.layers.InputLayer((None, 1, self._state_length), self._State)
        
        taskFeatures = lasagne.layers.SliceLayer(network, indices=slice(0, settings_['num_terrain_features']), axis=2)
        characterFeatures = lasagne.layers.SliceLayer(network, indices=slice(settings_['num_terrain_features']+1,self._state_length), axis=2)
        # network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        network = lasagne.layers.Conv1DLayer(
            taskFeatures, num_filters=32, filter_size=8,
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=lasagne.init.GlorotUniform())
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        
        # network = lasagne.layers.MaxPool1DLayer(network, pool_size=3)
        # network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        
        network = lasagne.layers.Conv1DLayer(
            network, num_filters=16, filter_size=4,
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=lasagne.init.GlorotUniform())
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        
        network = lasagne.layers.DenseLayer(
                network, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        self._critic_task_part = network
        
        # merge features together
            # Because the input is in R^3, needs to be in R^2
        network = lasagne.layers.FlattenLayer(network, outdim=2)
        network = lasagne.layers.ConcatLayer([network, characterFeatures], axis=1)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        
        
        network = lasagne.layers.DenseLayer(
                network, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        
        network = lasagne.layers.DenseLayer(
                network, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)


        self._critic = lasagne.layers.DenseLayer(
                network, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)
        # self._b_o = init_b_weights((n_out,))
        
        
        
        networkAct = lasagne.layers.InputLayer((None, 1, self._state_length), self._State)
        
        taskFeaturesAct = lasagne.layers.SliceLayer(networkAct, indices=slice(0, settings_['num_terrain_features']), axis=2)
        characterFeaturesAct = lasagne.layers.SliceLayer(networkAct, indices=slice(settings_['num_terrain_features']+1,self._state_length), axis=2)
        # networkAct = lasagne.layers.DropoutLayer(networkAct, p=self._dropout_p, rescale=True)
        networkAct = lasagne.layers.Conv1DLayer(
            taskFeaturesAct, num_filters=32, filter_size=8,
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=lasagne.init.GlorotUniform())
        networkAct = lasagne.layers.DropoutLayer(networkAct, p=self._dropout_p, rescale=True)
        # networkAct = lasagne.layers.MaxPool1DLayer(networkAct, pool_size=2)
        # networkAct = lasagne.layers.DropoutLayer(networkAct, p=self._dropout_p, rescale=True)
        
        networkAct = lasagne.layers.Conv1DLayer(
            networkAct, num_filters=16, filter_size=4,
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=lasagne.init.GlorotUniform())
        networkAct = lasagne.layers.DropoutLayer(networkAct, p=self._dropout_p, rescale=True)
        
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        self._actor_task_part = networkAct
        # networkAct = lasagne.layers.DropoutLayer(networkAct, p=self._dropout_p, rescale=True)
        
        # merge features together
            # Because the input is in R^3, needs to be in R^2
        networkAct = lasagne.layers.FlattenLayer(networkAct, outdim=2)
        networkAct = lasagne.layers.ConcatLayer([networkAct, characterFeaturesAct], axis=1)
        networkAct = lasagne.layers.DropoutLayer(networkAct, p=self._dropout_p, rescale=True)
        
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        networkAct = lasagne.layers.DropoutLayer(networkAct, p=self._dropout_p, rescale=True)
        
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=16,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        networkAct = lasagne.layers.DropoutLayer(networkAct, p=self._dropout_p, rescale=True)

        self._actor = lasagne.layers.DenseLayer(
                networkAct, num_units=self._action_length,
                nonlinearity=lasagne.nonlinearities.linear)
        # self._b_o = init_b_weights((n_out,))
        
        
          # print "Initial W " + str(self._w_o.get_value()) 
        
        self._states_shared = theano.shared(
            np.zeros((self._batch_size, 1, self._state_length),
                     dtype=theano.config.floatX))

        self._next_states_shared = theano.shared(
            np.zeros((self._batch_size, 1, self._state_length),
                     dtype=theano.config.floatX))

        self._rewards_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self._actions_shared = theano.shared(
            np.zeros((self._batch_size, self._action_length), dtype=theano.config.floatX),
            )
        
    def setStates(self, states):
        """
            This is reshaped to work properly with a 1D convolution that likes 
            its input as (batch_size, channel, state_dimension)
            
            Parameters
        ----------
        states : a (batch_size, state_dimension) numpy array
        """
        states = np.array(states)
        states = np.reshape(states, (states.shape[0], 1, states.shape[1]))
        self._states_shared.set_value(states)
    def setResultStates(self, resultStates):
        """
            This is reshaped to work properly with a 1D convolution that likes 
            its input as (batch_size, channel, state_dimension)
            
            Parameters
        ----------
        resultStates : a (batch_size, state_dimension) numpy array
        """
        resultStates = np.array(resultStates)
        resultStates = np.reshape(resultStates, (resultStates.shape[0], 1, resultStates.shape[1]))
        self._next_states_shared.set_value(resultStates)
