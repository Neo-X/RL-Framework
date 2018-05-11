
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
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten, Lambda
from keras.layers.convolutional import Conv1D
from keras.layers.merge import Concatenate
from keras.layers.advanced_activations import LeakyReLU
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K
import keras

# For debugging
# theano.config.mode='FAST_COMPILE'
from model.ModelInterface import ModelInterface

class DeepNNKeras(ModelInterface):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(DeepNNKeras,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        self._networkSettings = {}
        if ("network_settings" in settings_):
            self._networkSettings = settings_["network_settings"]
        ### data types for model
        # self._State = K.variable(value=np.random.rand(self._batch_size,self._state_length) ,name="State")
        self._State = keras.layers.Input(shape=(self._state_length,), name="State")
        # self._State.tag.test_value = np.random.rand(self._batch_size,self._state_length)
        # self._ResultState = K.variable(value=np.random.rand(self._batch_size,self._state_length), name="ResultState")
        self._ResultState = keras.layers.Input(shape=(self._state_length,), name="ResultState")
        # self._ResultState.tag.test_value = np.random.rand(self._batch_size,self._state_length)
        # self._Reward = K.variable(value=np.random.rand(self._batch_size,1), name="Reward")
        self._Reward = keras.layers.Input(shape=(1,), name="Reward")
        # self._Reward.tag.test_value = np.random.rand(self._batch_size,1)
        # self._Action = K.variable(value=np.random.rand(self._batch_size, self._action_length), name="Action")
        self._Action = keras.layers.Input(shape=(self._action_length,), name="Action")
        # self._Action.tag.test_value = np.random.rand(self._batch_size, self._action_length)
        
        ### Apparently after the first layer the patch axis is left out for most of the Keras stuff...
        input = self._State
        # input.trainable = True
        print ("Input ",  input)
        
        network = Dense(128, init='uniform')(input)
        print ("Network: ", network) 
        # network = LeakyReLU(alpha=np.asscalar(np.array([0.15], dtype=settings_['float_type'])))(network)
        network = Activation('relu')(network)
        network = Dropout(0.1)(network)
        network = Dense(64, init='uniform')(network) 
        network = Activation('relu')(network)
        network = Dropout(0.1)(network)
        network = Dense(32, init='uniform')(network) 
        network = Activation('relu')(network)
        network = Dropout(0.1)(network)
        network = Dense(16, init='uniform')(network) 
        network = Activation('relu')(network)
        network = Dropout(0.1)(network)
        # 1 output, linear activation
        network = Dense(1, init='uniform')(network)
        network = Activation('linear')(network)
        self._critic = network
        
        
        inputAct = self._State
        # inputAct.trainable = True
        print ("Input ",  inputAct)
        networkAct = Dense(128, init='uniform')(inputAct)
        print ("Network: ", networkAct)         
        networkAct = Activation('tanh')(networkAct)
        
        networkAct = Dense(64, init='uniform')(networkAct) 
        networkAct = Activation('tanh')(networkAct)
        networkAct = Dense(32, init='uniform')(networkAct) 
        networkAct = Activation('tanh')(networkAct)
        # 1 output, linear activation
        networkAct = Dense(self._action_length, init='uniform')(networkAct)
        networkAct = Activation('linear')(networkAct)
        self._actor = networkAct
        

    ### Setting network input values ###    
    def setStates(self, states):
        pass
        # self._states_shared.set_value(states)
    def setActions(self, actions):
        pass
        # self._actions_shared.set_value(actions)
    def setResultStates(self, resultStates):
        # self._next_states_shared.set_value(resultStates)
        pass
    def setRewards(self, rewards):
        # self._rewards_shared.set_value(rewards)
        pass
    def setTargets(self, targets):
        pass
        # self._targets_shared.set_value(targets)