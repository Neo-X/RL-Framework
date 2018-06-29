import numpy as np
# import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten, Lambda
from keras.layers.convolutional import Conv1D
from keras.layers.merge import Concatenate
from keras.layers.advanced_activations import LeakyReLU
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K

# For debugging
# theano.config.mode='FAST_COMPILE'
from model.ModelInterface import ModelInterface

class DeepNNTanHKeras(ModelInterface):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(DeepNNTanHKeras,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        ### Apparently after the first layer the patch axis is left out for most of the Keras stuff...
        input = Input(shape=(self._state_length,))
        # input.trainable = True
        print ("Input ",  input)
        inputAction = Input(shape=(self._action_length,))
        
        # inputAct.trainable = True
        print ("Input ",  input)
        networkAct = Dense(128, init='uniform')(input)
        print ("Network: ", networkAct)         
        networkAct = Activation('tanh')(networkAct)
        
        networkAct = Dense(64, init='uniform')(networkAct) 
        networkAct = Activation('tanh')(networkAct)
        networkAct = Dense(32, init='uniform')(networkAct) 
        networkAct = Activation('tanh')(networkAct)
        # 1 output, linear activation
        networkAct = Dense(self._action_length, init='uniform')(networkAct)
        networkAct_ = Activation('linear')(networkAct)
        if (self._settings['use_stochastic_policy'] and ( not ( 'use_fixed_std' in self.getSettings() and ( self.getSettings()['use_fixed_std'])))):
            print ("Adding stochastic layer")
            with_std = Activation('softplus')(networkAct)
            # with_std = lasagne.layers.DenseLayer(
            #       networkAct, num_units=self._action_length,
            #       nonlinearity=theano.tensor.nnet.softplus)
            networkAct_ = lasagne.layers.ConcatLayer([networkAct_, with_std], axis=1)
        # self._b_o = init_b_weights((n_out,))

        self._actor = Model(input=inputAct, output=networkAct_)
        if ( settings_['agent_name'] == 'algorithm.DPG.DPG'):
            network = Dense(128, init='uniform')(input, inputAction)
        else:
            network = Dense(128, init='uniform')(input)
        print ("Network: ", network) 
        # network = LeakyReLU(alpha=np.asscalar(np.array([0.15], dtype=settings_['float_type'])))(network)
        network = Activation('relu')(network)
        # network = Dropout(0.1)(network)
        network = Dense(64, init='uniform')(network) 
        network = Activation('relu')(network)
        # network = Dropout(0.1)(network)
        network = Dense(32, init='uniform')(network) 
        network = Activation('relu')(network)
        # network = Dropout(0.1)(network)
        network = Dense(16, init='uniform')(network) 
        # network = Activation('relu')(network)
        # network = Dropout(0.1)(network)
        # 1 output, linear activation
        network = Dense(1, init='uniform')(network)
        network = Activation('linear')(network)
        self._critic = Model(input=input, output=network)
