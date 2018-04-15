
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
from keras import regularizers
import keras
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K

# For debugging
# theano.config.mode='FAST_COMPILE'
from model.ModelInterface import ModelInterface


def getKerasActivation(type_name):
    """
        Compute a particular type of actiation to use
    """
    import keras.layers
    
    if (type_name == 'leaky_rectify'):
        return keras.layers.LeakyReLU(alpha=0.01)
    if (type_name == 'relu'):
        return Activation('relu')
    if (type_name == 'tanh'):
        return Activation('tanh')
    if (type_name == 'linear'):
        return Activation('linear')
    if (type_name == 'elu'):
        return keras.layers.ELU(alpha=1.0)
    if (type_name == 'sigmoid'):
        return Activation('sigmoid')
    if (type_name == 'softplus'):
        return Activation('softplus')
    else:
        print ("Activation type unknown: ", type_name)
        sys.exit()

class DeepNNKerasAdaptive(ModelInterface):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(DeepNNKerasAdaptive,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        self._networkSettings = {}
        if ("network_settings" in settings_):
            self._networkSettings = settings_["network_settings"]
        ### data types for model
        # self._State = K.variable(value=np.random.rand(self._batch_size,self._state_length) ,name="State")
        self._State = keras.layers.Input(shape=(self._state_length,))
        # self._State.tag.test_value = np.random.rand(self._batch_size,self._state_length)
        # self._ResultState = K.variable(value=np.random.rand(self._batch_size,self._state_length), name="ResultState")
        self._ResultState = keras.layers.Input(shape=(self._state_length,))
        # self._ResultState.tag.test_value = np.random.rand(self._batch_size,self._state_length)
        # self._Reward = K.variable(value=np.random.rand(self._batch_size,1), name="Reward")
        self._Reward = keras.layers.Input(shape=(1,))
        # self._Reward.tag.test_value = np.random.rand(self._batch_size,1)
        # self._Action = K.variable(value=np.random.rand(self._batch_size, self._action_length), name="Action")
        self._Action = keras.layers.Input(shape=(self._action_length,))
        # self._Action.tag.test_value = np.random.rand(self._batch_size, self._action_length)
        
        
        ### Apparently after the first layer the patch axis is left out for most of the Keras stuff...
        input = Input(shape=(self._state_length,))
        self._stateInput = input
        input2 = Input(shape=(self._action_length,)) 
        self._actionInput = input2
        # input.trainable = True
        print ("Input ",  input)
        
        taskFeatures = Lambda(lambda x: x[:,0:self._settings['num_terrain_features']], output_shape=(self._settings['num_terrain_features'],))(input)
        # taskFeatures = Lambda(lambda x: x[:,0:self._settings['num_terrain_features']])(input)
        characterFeatures = Lambda(lambda x: x[:,self._settings['num_terrain_features']:self._state_length], output_shape=(self._state_length-self._settings['num_terrain_features'],))(input)

        perform_pooling=True
        if ( "perform_convolution_pooling" in self._networkSettings):
            perform_pooling = self._networkSettings["perform_convolution_pooling"]
        if ( ( "use_single_network" in self._settings and
               (self._settings['use_single_network'] == True)
             )
            ):
            pass
        else:
            ### Number of layers and sizes of layers        
            layer_sizes = self._settings['policy_network_layer_sizes']
            print ("Actor Network layer sizes: ", layer_sizes)
            networkAct = self._stateInput
            for i in range(len(layer_sizes)):
                # networkAct = Dense(layer_sizes[i], init='uniform')(inputAct)
                if type(layer_sizes[i]) is list:
                    if ( len(layer_sizes[i][1])> 1):
                        if (i == 0):
                            networkAct = Reshape((1, self._settings['num_terrain_features'], self._settings['num_terrain_features']))(taskFeatures)
                        networkAct = keras.layers.Conv2D(layer_sizes[i][0], kernel_size=layer_sizes[i][1], strides=(1,1),
                                                         kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(networkAct)
                    else:
                        if (i == 0):
                            networkAct = Reshape((self._settings['num_terrain_features'], 1))(taskFeatures)
                        networkAct = keras.layers.Conv1D(layer_sizes[i][0], kernel_size=layer_sizes[i][1], strides=1,
                                                         kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(networkAct)
                        if (perform_pooling):
                            networkAct = keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid')(networkAct)
                    # networkAct = Dense(layer_sizes[i], 
                    #              kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(networkAct)
                elif ( layer_sizes[i] == "merge_features"):
                    networkAct = Flatten()(networkAct)
                    networkAct = Concatenate(axis=1)([networkAct, characterFeatures])
                else:
                    networkAct = Dense(layer_sizes[i], 
                                       kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(networkAct)
                    networkAct = getKerasActivation(self._settings['policy_activation_type'])(networkAct)
            # inputAct.trainable = True
            print ("Network: ", networkAct)         
            networkAct_ = networkAct
            networkAct = Dense(self._action_length, kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(networkAct)
            networkAct = getKerasActivation(self._settings['last_policy_layer_activation_type'])(networkAct)
    
            self._actor = networkAct
                    
            if (self._settings['use_stocastic_policy']):
                with_std = Dense(self._action_length, 
                                kernel_regularizer=regularizers.l2(self._settings['regularization_weight']),
                                kernel_initializer=(keras.initializers.VarianceScaling(scale=0.01,
                               mode='fan_avg', distribution='uniform', seed=None) ))(networkAct_)
                with_std = getKerasActivation(self._settings['_last_std_policy_layer_activation_type'])(with_std)
                # with_std = networkAct = Dense(self._action_length, kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(networkAct)
                self._actor = keras.layers.concatenate(inputs=[self._actor, with_std], axis=-1)
                
            self._actor = Model(input=self._stateInput, output=self._actor)
            # self._actor = Model(input=[self._stateInput, self._actionInput], output=self._actor)
            print("Actor summary: ", self._actor.summary())
            
        
        layer_sizes = self._settings['critic_network_layer_sizes']
        if ( self._settings["agent_name"] == "algorithm.DPGKeras.DPGKeras" 
             or (self._settings["agent_name"] == "algorithm.QPropKeras.QPropKeras")
             ):
            
            if ( ('train_extra_value_function' in settings_ and (settings_['train_extra_value_function']) )
                 or (settings_['agent_name'] == 'algorithm.QPropKeras.QPropKeras') # A must for Q-Prop 
                 ):
                
                print ("Value Network layer sizes: ", layer_sizes)
                network = input
                if ( self._dropout_p > 0.001 ):
                    network = Dropout(rate=self._dropout_p)(network)
                for i in range(len(layer_sizes)):
                    # network = Dense(layer_sizes[i], init='uniform')(input)
                    if type(layer_sizes[i]) is list:
                        if ( len(layer_sizes[i][1])> 1):
                            if (i == 0):
                                network = Reshape((1, self._settings['num_terrain_features'], self._settings['num_terrain_features']))(network)
                            network = keras.layers.Conv2D(layer_sizes[i][0], kernel_size=layer_sizes[i][1], strides=(1,1),
                                                          kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(network)
                        else:
                            if (i == 0):
                                network = Reshape((self._settings['num_terrain_features'], 1))(taskFeatures)
                            network = keras.layers.Conv1D(layer_sizes[i][0], kernel_size=layer_sizes[i][1], strides=1,
                                                          kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(network)
                            if (perform_pooling):
                                network = keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid')(network)
                        # network = keras.layers.Conv2D(4, (8,1), strides=(1, 1))(network)
                        # networkAct = Dense(layer_sizes[i], 
                        #              kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(networkAct)
                    elif ( layer_sizes[i] == "integrate_actor_part"):
                        pass
                    elif ( layer_sizes[i] == "merge_features"):
                        network = Flatten()(network)
                        network = Concatenate(axis=1)([network, characterFeatures])
                    else:
                        network = Dense(layer_sizes[i],
                                        kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(network)
                        network = getKerasActivation(self._settings['activation_type'])(network)
                        if ( self._dropout_p > 0.001 ):
                            network = Dropout(rate=self._dropout_p)(network)
                    
                network= Dense(1,
                               kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(network)
                network = Activation('linear')(network)
                self._value_function = Model(input=input, output=network)
                print("Value Function summary: ", self._value_function.summary())
                
                
            print ("Creating DPG network")
            # input = Concatenate()([characterFeatures, self._actionInput])
        
        print ("Critic Network layer sizes: ", layer_sizes)
        network = input
        if ( self._dropout_p > 0.001 ):
            network = Dropout(rate=self._dropout_p)(network)
        for i in range(len(layer_sizes)):
            # network = Dense(layer_sizes[i], init='uniform')(input)
            if type(layer_sizes[i]) is list:
                if ( len(layer_sizes[i][1])> 1):
                    if (i == 0):
                        network = Reshape((1, self._settings['num_terrain_features'], self._settings['num_terrain_features']))(network)
                    network = keras.layers.Conv2D(layer_sizes[i][0], kernel_size=layer_sizes[i][1], strides=(1,1))(network)
                else:
                    if (i == 0):
                        network = Reshape((self._settings['num_terrain_features'], 1))(taskFeatures)
                    network = keras.layers.Conv1D(layer_sizes[i][0], kernel_size=layer_sizes[i][1], strides=1)(network)
                    if (perform_pooling):
                        network = keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid')(network)
                # network = keras.layers.Conv2D(4, (8,1), strides=(1, 1))(network)
                # networkAct = Dense(layer_sizes[i], 
                #              kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(networkAct)
            elif ( layer_sizes[i] == "integrate_actor_part"):
                network = Concatenate()([network, self._actionInput])
            elif ( layer_sizes[i] == "merge_features"):
                network = Flatten()(network)
                network = Concatenate(axis=1)([network, characterFeatures])
            else:
                
                network = Dense(layer_sizes[i],
                                kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(network)
                network = getKerasActivation(self._settings['activation_type'])(network)
                if ( self._dropout_p > 0.001 ):
                    network = Dropout(rate=self._dropout_p)(network)
            
        if ( "use_single_network" in self._settings and 
             (self._settings['use_single_network'] == True)):
            networkAct = network       
            networkAct_ = networkAct
            networkAct = Dense(self._action_length, kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(networkAct)
            networkAct = getKerasActivation(self._settings['last_policy_layer_activation_type'])(networkAct)
    
            self._actor = networkAct
                    
            if (self._settings['use_stocastic_policy']):
                with_std = Dense(self._action_length, 
                                kernel_regularizer=regularizers.l2(self._settings['regularization_weight']),
                                kernel_initializer=(keras.initializers.VarianceScaling(scale=0.01,
                               mode='fan_avg', distribution='uniform', seed=None) ))(networkAct_)
                with_std = getKerasActivation(self._settings['_last_std_policy_layer_activation_type'])(with_std)
                # with_std = networkAct = Dense(self._action_length, kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(networkAct)
                self._actor = keras.layers.concatenate(inputs=[self._actor, with_std], axis=-1)
                
            self._actor = Model(input=self._stateInput, output=self._actor)
            print("Actor summary: ", self._actor.summary())
        network= Dense(1,
                       kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(network)
        network = Activation('linear')(network)
            
        if ( self._settings["agent_name"] == "algorithm.DPGKeras.DPGKeras"
             or (self._settings["agent_name"] == "algorithm.QPropKeras.QPropKeras") 
             ):
            print ( "Creating DPG Keras Model")
            self._critic = Model(input=[self._stateInput, self._actionInput], output=network)
        else:
            self._critic = Model(input=input, output=network)
            
        print("Critic summary: ", self._critic.summary())


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
        # self._targets_shared.set_value(targets)ModelInterface):
    
    def getValueFunction(self):
        return self._value_function
    
    ######### Symbolic Variables ######
    def getStateSymbolicVariable(self):
        return self._stateInput
    def getActionSymbolicVariable(self):
        return self._actionInput
    def getResultStateSymbolicVariable(self):
        return self._ResultState
    def getRewardSymbolicVariable(self):
        return self._Reward
    def getTargetsSymbolicVariable(self):
        return self._Target