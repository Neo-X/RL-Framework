
import theano
from theano import tensor as T
import numpy as np
# import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.merge import Concatenate
from keras.utils.np_utils import to_categoricalnetwork = lasagne.layers.ReshapeLayer(taskFeatures, (-1, 1, self._settings['num_terrain_features']))
import keras.backend as K

# For debugging
# theano.config.mode='FAST_COMPILE'
from model.ModelInterface import ModelInterface

class DeepCNNKeras(ModelInterface):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(DeepCNNKeras,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        
        input = Input(shape=[self._state_length])
        input.trainable = True
        print ("Input ",  input)
        ## Custom slice layer, Keras does not have this layer type...
        taskFeatures = Lambda(lambda x: x[:,0:self._settings['num_terrain_features']], output_shape=(-1, self._settings['num_terrain_features']))(input)
        characterFeatures = Lambda(lambda x: x[:,self._settings['num_terrain_features']:self._state_length], output_shape=(-1, self._state_length-self._settings['num_terrain_features']))(input)
        
        network = Reshape((-1, 1, self._settings['num_terrain_features']))(taskFeatures)
        # network = lasagne.layers.ReshapeLayer(taskFeatures, (-1, 1, self._settings['num_terrain_features']))
        
        network = Dense(128, init='uniform')(input)
        print ("Network: ", network) 
        network = Activation('relu')(network)
        network = Dense(64, init='uniform')(network) 
        network = Activation('relu')(network)
        # 1 output, linear activation
        network = Dense(1, init='uniform')(network)
        network = Activation('linear')(network)
        model = Model(input=input, output=network)
        sgd = SGD(lr=0.01, momentum=0.9)
        print ("Clipping: ", sgd.decay)
        model.compile(loss='mse', optimizer=sgd)
        print ("Loss ", model.total_loss)
        weights = [input] + model.trainable_weights # weight tensors
        # weights = [weight for weight in weights if model.get_layer(weight.name[:-2]).trainable] # filter down weights tensors to only ones which are trainable
        gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors
        
        input_tensors = [model.inputs[0], # input data
                         model.sample_weights[0], # how much to weight each sample by
                         model.targets[0], # labels
                         K.learning_phase(), # train or test mode
        ]
        
        get_gradients = K.function(inputs=input_tensors, outputs=gradients)
        
        nb_sample = 32
        
        inputs = [np.random.randn(nb_sample, 1), # X
                  np.ones(nb_sample), # sample weights
                  np.random.randint(1, size=[nb_sample, 1]), # y
                  0 # learning phase in TEST mode
        ]
        
        """
            score = model.fit(_states, _actions,
              nb_epoch=1, batch_size=32,
              validation_data=(_states, _actions)
              # callbacks=[early_stopping],
              )
            
            # print (zip(weights, get_gradients(_states)))
        
            errors.extend(score.history['loss'])
        
        """
        
        # data types for model
        self._State = T.matrix("State")
        self._State.tag.test_value = np.random.rand(self._batch_size, self._state_length)
        self._ResultState = T.matrix("ResultState")
        self._ResultState.tag.test_value = np.random.rand(self._batch_size, self._state_length)
        self._Reward = T.col("Reward")
        self._Reward.tag.test_value = np.random.rand(self._batch_size,1)
        self._Target = T.col("Target")
        self._Target.tag.test_value = np.random.rand(self._batch_size,1)
        self._Action = T.matrix("Action")
        self._Action.tag.test_value = np.random.rand(self._batch_size, self._action_length)
        
        # create a small convolutional neural network
        network = lasagne.layers.InputLayer((None, self._state_length), self._State)
        
        taskFeatures = lasagne.layers.SliceLayer(network, indices=slice(0, self._settings['num_terrain_features']), axis=1)
        # characterFeatures = lasagne.layers.SliceLayer(network, indices=slice(-(self._state_length-self._settings['num_terrain_features']), None), axis=1)
        characterFeatures = lasagne.layers.SliceLayer(network, indices=slice(self._settings['num_terrain_features'], self._state_length), axis=1)
        print ("taskFeatures Shape:", lasagne.layers.get_output_shape(taskFeatures))
        print ("characterFeatures Shape:", lasagne.layers.get_output_shape(characterFeatures))
        print ("State length: ", self._state_length)
        
        network = lasagne.layers.ReshapeLayer(taskFeatures, (-1, 1, self._settings['num_terrain_features']))
        
        network = lasagne.layers.Conv1DLayer(
            network, num_filters=16, filter_size=8,
            nonlinearity=lasagne.nonlinearities.rectify)
        
        # network = lasagne.layers.MaxPool1DLayer(network, pool_size=3)
        """
        network = lasagne.layers.Conv1DLayer(
            network, num_filters=32, filter_size=4,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        """
        network = lasagne.layers.Conv1DLayer(
            network, num_filters=32, filter_size=4,
            nonlinearity=lasagne.nonlinearities.rectify)
        
        self._critic_task_part = network 
        
        """
        # network = lasagne.layers.MaxPool1DLayer(network, pool_size=3)
        
        network = lasagne.layers.Conv1DLayer(
            network, num_filters=32, filter_size=4,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        
        network = lasagne.layers.DenseLayer(
                network, num_units=128,
                nonlinearity=lasagne.nonlinearities.rectify)
        """
        network = lasagne.layers.FlattenLayer(network, outdim=2)
        network = lasagne.layers.ConcatLayer([network, characterFeatures], axis=1)
        
        network = lasagne.layers.DenseLayer(
                network, num_units=64,
                nonlinearity=lasagne.nonlinearities.rectify)
        
        network = lasagne.layers.DenseLayer(
                network, num_units=32,
                nonlinearity=lasagne.nonlinearities.rectify)
        
        network = lasagne.layers.DenseLayer(
                network, num_units=16,
                nonlinearity=lasagne.nonlinearities.rectify)
        
        self._critic = lasagne.layers.DenseLayer(
                network, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)
        # self._b_o = init_b_weights((n_out,))
        networkAct = lasagne.layers.InputLayer((None, self._state_length), self._State)
        
        taskFeaturesAct = lasagne.layers.SliceLayer(networkAct, indices=slice(0, self._settings['num_terrain_features']), axis=1)
        characterFeaturesAct = lasagne.layers.SliceLayer(networkAct, indices=slice(self._settings['num_terrain_features']+1,self._state_length), axis=1)
        
        networkAct = lasagne.layers.ReshapeLayer(taskFeaturesAct, (-1, 1, self._settings['num_terrain_features']))
        
        networkAct = lasagne.layers.Conv1DLayer(
            networkAct, num_filters=16, filter_size=8,
            nonlinearity=lasagne.nonlinearities.rectify)
        
        # network = lasagne.layers.MaxPool1DLayer(network, pool_size=3)
        """
        networkAct = lasagne.layers.Conv1DLayer(
            networkAct, num_filters=32, filter_size=4,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        """
        networkAct = lasagne.layers.Conv1DLayer(
            networkAct, num_filters=32, filter_size=4,
            nonlinearity=lasagne.nonlinearities.rectify)
        
        # network = lasagne.layers.MaxPool1DLayer(network, pool_size=3)
        
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
        networkAct = lasagne.layers.FlattenLayer(networkAct, outdim=2)
        networkAct = lasagne.layers.ConcatLayer([networkAct, characterFeaturesAct], axis=1)
        
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=64,
                nonlinearity=lasagne.nonlinearities.rectify)
        
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=32,
                nonlinearity=lasagne.nonlinearities.rectify)
        
        self._actor = lasagne.layers.DenseLayer(
                    networkAct, num_units=self._action_length,
                    nonlinearity=lasagne.nonlinearities.linear)
    
        if (self._settings['use_stocastic_policy']):
            with_std = lasagne.layers.DenseLayer(
                    networkAct, num_units=self._action_length,
                    nonlinearity=theano.tensor.nnet.softplus)
            self._actor = lasagne.layers.ConcatLayer([self._actor, with_std], axis=1)
        # self._b_o = init_b_weights((n_out,))
        
        
          # print ("Initial W " + str(self._w_o.get_value()) )
        
        self._states_shared = theano.shared(
            np.zeros((self._batch_size, self._state_length),
                     dtype=theano.config.floatX))

        self._next_states_shared = theano.shared(
            np.zeros((self._batch_size, self._state_length),
                     dtype=theano.config.floatX))

        self._rewards_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))
        
        self._target_shared = theano.shared(
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
        # states = np.array(states)
        # states = np.reshape(states, (states.shape[0], 1, states.shape[1]))
        self._states_shared.set_value(states)
    def setResultStates(self, resultStates):
        """
            This is reshaped to work properly with a 1D convolution that likes 
            its input as (batch_size, channel, state_dimension)
            
            Parameters
        ----------
        resultStates : a (batch_size, state_dimension) numpy array
        """
        # resultStates = np.array(resultStates)
        # resultStates = np.reshape(resultStates, (resultStates.shape[0], 1, resultStates.shape[1]))
        self._next_states_shared.set_value(resultStates)

