
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
from keras.layers import LSTM, LSTMCell, GRU
from keras.layers.convolutional import Conv1D
from keras.layers.merge import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
import keras
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K
import tensorflow as tf
from keras.layers import Layer
from util.coordconv import *
# For debugging
# theano.config.mode='FAST_COMPILE'
from model.ModelInterface import ModelInterface

class CoordConv2D(Layer):
    """
        https://gist.github.com/Dref360/b330e75cb121c03a0066d9587a7bfee5
    """
    def __init__(self, channel, kernel_size, padding='valid', **kwargs):
        self.layer = keras.layers.Conv2D(channel, kernel_size, padding=padding)
        self.name = 'CoordConv2D'
        super(CoordConv2D, self).__init__(**kwargs)

    def call(self, input):
        input_shape = tf.unstack(K.shape(input))
        if K.image_data_format() == 'channel_first':
            bs, channel, w, h = input_shape
        else:
            bs, w, h, channel = input_shape

        # Get indices
        indices = tf.to_float(tf.where(K.ones([bs, w, h])))
        canvas = K.reshape(indices, [bs, w, h, 3])[..., 1:]
        # Normalize the canvas
        canvas = canvas / tf.to_float(K.reshape([w, h], [1, 1, 1, 2]))
        canvas = (canvas * 2) - 1

        # If channel_first, we swap
        if K.image_data_format() == 'channel_first':
            canvas = K.swap_axes(canvas, [0, 3, 1, 2])

        # Concatenate channel-wise
        input = K.concatenate([input, canvas], -1)
        return self.layer(input)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

def getKerasActivation(type_name):
    """
        Compute a particular type of actiation to use
    """
    import keras.layers
    
    if (type_name == 'leaky_rectify'):
        return keras.layers.LeakyReLU(alpha=0.1)
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
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=False):

        super(DeepNNKerasAdaptive,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=print_info)
        self._networkSettings = {}
        if ("network_settings" in settings_):
            self._networkSettings = settings_["network_settings"]
            
        self._sequence_length = 1
        self._lstm_batch_size = 32
        if ( "lstm_batch_size" in settings_):
            self._lstm_batch_size = settings_["lstm_batch_size"]
        self._stateful_lstm = False
        if ("train_LSTM_stateful" in self._settings
            and (self._settings["train_LSTM_stateful"])):
            self._stateful_lstm = True
        ### data types for model
        # self._State = K.variable(value=np.random.rand(self._batch_size,self._state_length) ,name="State")
        # self._State = keras.layers.Input(shape=(self._state_length,), name="State", batch_shape=(32,self._state_length))
        if (("train_LSTM" in self._settings)
                and (self._settings["train_LSTM"] == True)):
            if ("simulation_model" in self._settings and
                (self._settings["simulation_model"] == True)):
                if (self._stateful_lstm):
                    self._State = keras.layers.Input(shape=(self._sequence_length, self._state_length), batch_shape=(1, 1, self._state_length), name="State")
                else:
                    self._State = keras.layers.Input(shape=(31, self._state_length), name="State")
            else:
                if (self._stateful_lstm):
                    self._State = keras.layers.Input(shape=(self._sequence_length, self._state_length), batch_shape=(self._lstm_batch_size, self._sequence_length, self._state_length), name="State")
                else:
                    self._State = keras.layers.Input(shape=(31, self._state_length), name="State")
        else:
            self._State = keras.layers.Input(shape=(self._state_length,), name="State")
            
        
        # self._State.tag.test_value = np.random.rand(self._batch_size,self._state_length)
        # self._ResultState = K.variable(value=np.random.rand(self._batch_size,self._state_length), name="ResultState")
        # self._ResultState = keras.layers.Input(shape=(self._state_length,), name="ResultState", batch_shape=(32,self._state_length))
        if (("train_LSTM" in self._settings)
                and (self._settings["train_LSTM"] == True)):
            if ("simulation_model" in self._settings and
                (self._settings["simulation_model"] == True)):
                if (self._stateful_lstm):
                    self._ResultState = keras.layers.Input(shape=(self._sequence_length, self._result_state_length), batch_shape=(1, 1, self._state_length), name="ResultState")
                else:
                    self._ResultState = keras.layers.Input(shape=(31, self._result_state_length), name="ResultState")
            else:
                if (self._stateful_lstm):
                    self._ResultState = keras.layers.Input(shape=(self._sequence_length, self._result_state_length), batch_shape=(self._lstm_batch_size, self._sequence_length, self._state_length), name="ResultState")
                else:
                    self._ResultState = keras.layers.Input(shape=(31, self._result_state_length), name="ResultState")
        else:
            self._ResultState = keras.layers.Input(shape=(self._result_state_length,), name="ResultState")
        # self._ResultState.tag.test_value = np.random.rand(self._batch_size,self._state_length)
        # self._Reward = K.variable(value=np.random.rand(self._batch_size,1), name="Reward")
        # self._Reward = keras.layers.Input(shape=(1,), name="Reward", batch_shape=(32,1))
        if (("train_LSTM" in self._settings)
                and (self._settings["train_LSTM"] == True)):
            # if ("simulation_model" in self._settings and
            #     (self._settings["simulation_model"] == True)):
            self._Reward = keras.layers.Input(shape=(self._sequence_length,1), name="Reward")
            # else:
            # self._Reward = keras.layers.Input(shape=(10, self._sequence_length,1), name="Reward")
        else:
            self._Reward = keras.layers.Input(shape=(1,), name="Reward")
        # self._Reward.tag.test_value = np.random.rand(self._batch_size,1)
        # self._Action = K.variable(value=np.random.rand(self._batch_size, self._action_length), name="Action")
        # self._Action = keras.layers.Input(shape=(self._action_length,), name="Action", batch_shape=(32,self._action_length))
        self._Action = keras.layers.Input(shape=(self._action_length,), name="Action")
        # self._Action.tag.test_value = np.random.rand(self._batch_size, self._action_length)
        
        self._data_format_ = keras.backend.image_data_format()
        if ("image_data_format" in self._networkSettings ):
            self._data_format_ = self._networkSettings["image_data_format"]
        ### Apparently after the first layer the patch axis is left out for most of the Keras stuff...
        # input = Input(shape=(self._state_length,))
        self._stateInput = self._State
        # input2 = Input(shape=(self._action_length,)) 
        self._actionInput = self._Action
        # input.trainable = True
        
        print ("self._stateInput ",  self._stateInput)
        inputAct = self._State
        
        ### It is complicated to serialize lambda functions, better to define a function
        def keras_slice(x, begin,end):
            return x[:,begin:end]
        if ('split_terrain_input' in self._networkSettings 
            and self._networkSettings['split_terrain_input']):
            mid = int((self._settings['num_terrain_features']/3) * 2)
            velFeatures_x = Lambda(keras_slice, output_shape=(int(self._settings['num_terrain_features']/3),),
                              arguments={'begin': 0, 'end': int(mid/2)})(inputAct)
            velFeatures_y = Lambda(keras_slice, output_shape=(int(self._settings['num_terrain_features']/3),),
                              arguments={'begin': int(mid/2), 'end': mid})(inputAct)
            taskFeatures = Lambda(keras_slice, output_shape=(int(self._settings['num_terrain_features']/3),),
                              arguments={'begin': mid, 'end': self._settings['num_terrain_features']})(inputAct)
        else:
            taskFeatures = Lambda(keras_slice, output_shape=(self._settings['num_terrain_features'],),
                              arguments={'begin': 0, 'end': self._settings['num_terrain_features']})(inputAct)
        # taskFeatures = Lambda(lambda x: x[:,0:self._settings['num_terrain_features']])(inputAct)
        characterFeatures = Lambda(keras_slice, output_shape=(self._state_length-self._settings['num_terrain_features'],),
                                   arguments={'begin': self._settings['num_terrain_features'], 
                                              'end': self._state_length})(inputAct)
        
        if (("train_LSTM" in self._settings)
                and (self._settings["train_LSTM"] == True)):
            taskFeatures = self._State
        """
        taskFeatures = inputAct
        characterFeatures = inputAct
        """
        self._perform_pooling=False
        if ( "perform_convolution_pooling" in self._networkSettings):
            self._perform_pooling = self._networkSettings["perform_convolution_pooling"]
        if ( ( "use_single_network" in self._settings and
               (self._settings['use_single_network'] == True)
             )
            ):
            pass
        else:
            ### Number of layers and sizes of layers        
            layer_sizes = self._settings['policy_network_layer_sizes']
            print ("Actor Network layer sizes: ", layer_sizes)
            networkAct = inputAct
            
            if ( self._dropout_p > 0.001 
                 and ("use_dropout_in_actor" in self._settings 
                      and (self._settings["use_dropout_in_actor"] == True)) ):
                networkAct = Dropout(rate=self._dropout_p)(networkAct)
            
            networkAct = self.createSubNetwork(networkAct, layer_sizes)
            
            # inputAct.trainable = True
            networkAct_ = networkAct
            if (layer_sizes[-1] != "merge_state_types"
                and ( not ("network_leave_off_end" in self._settings 
                           and (self._settings["network_leave_off_end"] == True )))):
                networkAct = Dense(n_out, 
                                   kernel_regularizer=regularizers.l2(self._settings['regularization_weight']),
                                   bias_regularizer=regularizers.l2(self._settings['regularization_weight']))(networkAct)
                networkAct = getKerasActivation(self._settings['last_policy_layer_activation_type'])(networkAct)
            """
            if (("train_LSTM" in self._settings)
                and (self._settings["train_LSTM"] == True)):
                networkAct = networkAct = Reshape((1, 64))(networkAct)
            """
            self._actor = networkAct
                    
            if (self._settings['use_stochastic_policy']):
                if ("split_single_net_earlier" in self._networkSettings and 
                    self._networkSettings["split_single_net_earlier"] == True):
                    second_last_layer = Dense(layer_sizes[len(layer_sizes)-1], 
                                       kernel_regularizer=regularizers.l2(self._settings['regularization_weight']),
                                       bias_regularizer=regularizers.l2(self._settings['regularization_weight']))(second_last_layer)
                    second_last_layer = getKerasActivation(self._settings['policy_activation_type'])(second_last_layer)
                    with_std = Dense(n_out, 
                                kernel_regularizer=regularizers.l2(self._settings['regularization_weight']),
                                bias_regularizer=regularizers.l2(self._settings['regularization_weight']),
                                kernel_initializer=(keras.initializers.VarianceScaling(scale=0.01,
                                mode='fan_avg', distribution='uniform', seed=None) )
                                     )(second_last_layer)
                else:
                    with_std = Dense(n_out, 
                                kernel_regularizer=regularizers.l2(self._settings['regularization_weight']),
                                bias_regularizer=regularizers.l2(self._settings['regularization_weight']),
                                kernel_initializer=(keras.initializers.VarianceScaling(scale=0.01,
                                mode='fan_avg', distribution='uniform', seed=None) )
                                     )(networkAct_)
                with_std = getKerasActivation(self._settings['_last_std_policy_layer_activation_type'])(with_std)
                # with_std = networkAct = Dense(self._action_length, kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(networkAct)
                self._actor = keras.layers.concatenate(inputs=[self._actor, with_std], axis=-1)
                
            # input_ = [self._stateInput, self._actionInput, self._Reward]
            # input_ = self._stateInput
            # self._actor = Model(input=input_, output=self._actor)
            # self._actor = Model(input=[self._stateInput, self._actionInput], output=self._actor)
            # print("Actor summary: ", self._actor.summary())
            
        layer_sizes = self._settings['critic_network_layer_sizes']
        
        print ("Critic Network layer sizes: ", layer_sizes)
        network = self._stateInput
        """
        if ( self._dropout_p > 0.001 ):
            network = Dropout(rate=self._dropout_p)(network)
        """
        network = self.createSubNetwork(network, layer_sizes)
        
            
        if ( "use_single_network" in self._settings and 
             (self._settings['use_single_network'] == True)):
            print ("Using a single network model")
            if ("split_single_net_earlier" in self._networkSettings and 
                    self._networkSettings["split_single_net_earlier"] == True):
                second_last_layer_ = Dense(layer_sizes[len(layer_sizes)-1],
                                kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(second_last_layer)
                second_last_layer_ = getKerasActivation(self._settings['activation_type'])(second_last_layer_)
                networkAct = second_last_layer_       
                # networkAct_ = networkAct
                networkAct = Dense(self._action_length, 
                                   kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                   bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(second_last_layer_)
                networkAct = getKerasActivation(self._settings['last_policy_layer_activation_type'])(networkAct)
            else:
                networkAct = network       
                networkAct_ = networkAct
                networkAct = Dense(self._action_length, 
                                   kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                   bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(networkAct)
                networkAct = getKerasActivation(self._settings['last_policy_layer_activation_type'])(networkAct)
    
            self._actor = networkAct
                    
            if (self._settings['use_stochastic_policy']):
                if ("split_single_net_earlier" in self._networkSettings and 
                    (self._networkSettings["split_single_net_earlier"] == True)):
                    second_last_layer = Dense(layer_sizes[len(layer_sizes)-1],
                                kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(second_last_layer)
                    second_last_layer = getKerasActivation(self._settings['activation_type'])(second_last_layer)
                    
                    with_std = Dense(self._action_length, 
                                kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                kernel_initializer=(keras.initializers.VarianceScaling(scale=0.01,
                               mode='fan_avg', distribution='uniform', seed=None) ))(second_last_layer)
                else:
                    with_std = Dense(self._action_length, 
                                    kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                    bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                    kernel_initializer=(keras.initializers.VarianceScaling(scale=0.01,
                                   mode='fan_avg', distribution='uniform', seed=None) ))(networkAct_)
                with_std = getKerasActivation(self._settings['_last_std_policy_layer_activation_type'])(with_std)
                # with_std = networkAct = Dense(self._action_length, kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(networkAct)
                self._actor = keras.layers.concatenate(inputs=[self._actor, with_std], axis=-1)
                
            if ("use_viz_for_policy" in self._settings 
                and self._settings["use_viz_for_policy"] == True):
                self._trans = Dense(self._settings["dense_state_size"],
                                kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(self._networkMiddle)
                self._trans = getKerasActivation(self._settings['last_fd_layer_activation_type'])(self._trans)
                
            if ("forward_dynamics_model_type" in self._settings 
                and (self._settings["forward_dynamics_model_type"] == "SingleNet")
                and (self._settings['use_single_network'] == True)):
                self._forward_dynamics_net = Dense(self._settings["fd_network_layer_sizes"][-1],
                                kernel_regularizer=regularizers.l2(self._settings['regularization_weight']),
                                bias_regularizer=regularizers.l2(self._settings['regularization_weight']))(self._networkMiddle)
                self._forward_dynamics_net = getKerasActivation(self._settings['last_fd_layer_activation_type'])(self._forward_dynamics_net)
                
                self._reward_net = Dense(self._settings["fd_network_layer_sizes"][-1],
                                kernel_regularizer=regularizers.l2(self._settings['regularization_weight']),
                                bias_regularizer=regularizers.l2(self._settings['regularization_weight']))(self._networkMiddle)
                self._reward_net = getKerasActivation(self._settings['last_fd_layer_activation_type'])(self._reward_net)
                
            # self._actor = Model(input=self._stateInput, output=self._actor)
            # print("Actor summary: ", self._actor.summary())
            ### Render a nice graph of the network
            # from keras.utils import plot_model
            # plot_model(self._actor, to_file='model.png')
        network= Dense(1,
                       kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                       bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(network)
        
        if ("last_critic_layer_activation_type" in self._settings):
            self._critic = getKerasActivation(self._settings['last_critic_layer_activation_type'])(network)
        else:
            self._critic = Activation('linear')(network)
            
        """
        if ( self._settings["agent_name"] == "algorithm.DPGKeras.DPGKeras"
             or (self._settings["agent_name"] == "algorithm.QPropKeras.QPropKeras") 
             ):
            print ( "Creating DPG Keras Model")
            self._critic = Model(input=[self._stateInput, self._actionInput], output=self._critic)
        else:
            self._critic = Model(input=self._stateInput, output=self._critic)
        """ 
        # print("Critic summary: ", self._critic.summary())
        
    def createSubNetwork(self, input, layer_info):
        
        network = input
        layer_sizes = layer_info
        for i in range(len(layer_sizes)):
            second_last_layer = network
            print ("layer_sizes[",i,"]: ", layer_sizes[i])
            print ("shape: ", repr(keras.backend.int_shape(network)))
            if type(layer_sizes[i]) is list:
                if (layer_sizes[i][0] == "LSTM"):
                    # print ("layer.output_shape: ", keras.backend.shape(network))
                    # network = Reshape((-1, layer_sizes[i][1]))(network)
                    network = LSTM(layer_sizes[i][2], stateful=self._stateful_lstm)(network)
                elif (layer_sizes[i][0] == "GRU"):
                    # print ("layer.output_shape: ", keras.backend.shape(network))
                    network = Reshape((1, layer_sizes[i][1]))(network)
                    network = GRU(layer_sizes[i][2], stateful=self._stateful_lstm)(network)
                elif (layer_sizes[i][0] == "Reshape"):
                    network = Reshape(layer_sizes[i][1])(network)
                elif (layer_sizes[i][0] == "TimeDistributedConv"):
                    
                    input_ = keras.layers.Input(shape=(1, self._state_length), name="State_Conv")
                    subnet = self.createSubNetwork(input_, layer_sizes[i][2])
                    subnet = Model(inputs=input_, outputs=subnet)
                    print("Subnet summary")
                    subnet.summary()
                    # subnet = Dense(8)
                    network = keras.layers.TimeDistributed(subnet, input_shape=(None, 31, 4096))(input)
                    # network = keras.layers.TimeDistributed(getKerasActivation(self._settings['activation_type'])(network))
                elif (layer_sizes[i][0] == "TimeDistributed"):
                    network = keras.layers.TimeDistributed(input_shape=(40, 31, 4096))(network)
                    
                elif (layer_sizes[i][0] == "integrate_actor_part"):
                    subnet_ = self.createSubNetwork(self._actionInput, layer_sizes[i][1])
                    network = Concatenate()([network, subnet_])
                elif (layer_sizes[i][0] == "max_pool"):
                        network = keras.layers.MaxPooling2D(pool_size=layer_sizes[i][1], strides=None, padding='valid', 
                                                                   data_format=self._data_format_)(network)  
                elif (layer_sizes[i][0] == "avg_pool"):
                        network = keras.layers.AveragePooling2D(pool_size=layer_sizes[i][1], strides=None, padding='valid', 
                                                                   data_format=self._data_format_)(network)  
                elif ( layer_sizes[i][0] == "dropout" ):
                    network = Dropout(rate=layer_sizes[i][1])(network)    
                elif ( len(layer_sizes[i][1])> 1 ):
                    if (i == 0):
                        if ('split_terrain_input' in self._networkSettings 
                        and self._networkSettings['split_terrain_input']):
                            if ("image_data_format" in self._networkSettings 
                                and (self._networkSettings["image_data_format"] == "channels_first")):
                                networkVel_x = Reshape((1, self._settings['terrain_shape'][1], self._settings['terrain_shape'][2]))(velFeatures_x)
                                networkVel_y = Reshape((1, self._settings['terrain_shape'][1], self._settings['terrain_shape'][2]))(velFeatures_y)
                                network = Reshape((1, self._settings['terrain_shape'][1], self._settings['terrain_shape'][2]))(taskFeatures)
                            else:
                                networkVel_x = Reshape((self._settings['terrain_shape'][0], self._settings['terrain_shape'][1], 1))(velFeatures_x)
                                networkVel_y = Reshape((self._settings['terrain_shape'][0], self._settings['terrain_shape'][1], 1))(velFeatures_y)
                                network = Reshape((self._settings['terrain_shape'][0], self._settings['terrain_shape'][1], 1))(taskFeatures)
                        else:    
                            network = Reshape(self._settings['terrain_shape'])(network)
                    stride = (1,1)
                    if (len(layer_sizes[i]) > 2):
                        stride = layer_sizes[i][2]
                        
                    if ("use_coordconv_layers" in self._networkSettings 
                            and (self._networkSettings["use_coordconv_layers"] == True)):
                        network = CoordinateChannel2D()(network)
                        # network = CoordConv2D(layer_sizes[i][0], kernel_size=layer_sizes[i][1], # strides=stride,
                        #                                  # kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight'])
                        #                                  )(network)
                    # else:
                    network = keras.layers.Conv2D(layer_sizes[i][0], kernel_size=layer_sizes[i][1], strides=stride,
                                                     kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                                     bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                                     data_format=self._data_format_)(network)
                    network = getKerasActivation(self._settings['activation_type'])(network)
                    if ('split_terrain_input' in self._networkSettings 
                            and self._networkSettings['split_terrain_input']):
                        if ("use_coordconv_layers" in self._networkSettings 
                            and (self._networkSettings["use_coordconv_layers"] == True)):
                            networkVel_x = CoordinateChannel2D()(networkVel_x)
                            networkVel_y = CoordinateChannel2D()(networkVel_y)
                        networkVel_x = keras.layers.Conv2D(layer_sizes[i][0], kernel_size=layer_sizes[i][1], strides=stride,
                                                     kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                                     bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                                     data_format=self._data_format_)(networkVel_x)
                        networkVel_x = getKerasActivation(self._settings['activation_type'])(networkVel_x)
                        networkVel_y = keras.layers.Conv2D(layer_sizes[i][0], kernel_size=layer_sizes[i][1], strides=stride,
                                                     kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                                     bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']), 
                                                     data_format=self._data_format_)(networkVel_y)   
                        networkVel_y = getKerasActivation(self._settings['activation_type'])(networkVel_y)         
                    if (self._perform_pooling):
                        network = keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='valid', 
                                                            data_format=self._data_format_)(network)
                        if ('split_terrain_input' in self._networkSettings 
                            and self._networkSettings['split_terrain_input']):
                            networkVel_x = keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='valid', 
                                                                     data_format=self._data_format_)(networkVel_x)
                            networkVel_y = keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='valid',
                                                                     data_format=self._data_format_)(networkVel_y)
                else:
                    if (i == 0):
                        # network = Reshape((self._state_length, 1))(taskFeatures)
                        network = Reshape((self._settings['num_terrain_features'], 1))(taskFeatures)
                    stride_ = 1
                    if (len(layer_sizes[i]) > 2):
                        stride_ = layer_sizes[i][2]
                    if ("use_coordconv_layers" in self._networkSettings 
                            and (self._networkSettings["use_coordconv_layers"] == True)):
                        network = CoordinateChannel1D()(network)
                    network = keras.layers.Conv1D(layer_sizes[i][0], kernel_size=layer_sizes[i][1], strides=stride_,
                                                  kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                                  bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(network)
                    network = getKerasActivation(self._settings['activation_type'])(network)
                    if (self._perform_pooling):
                        network = keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid')(network)
                # network = keras.layers.Conv2D(4, (8,1), strides=(1, 1))(network)
                # networkAct = Dense(layer_sizes[i], 
                #              kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(networkAct)
            elif ( layer_sizes[i] == "integrate_actor_part"):
                network = Concatenate()([network, self._actionInput])
            elif ( layer_sizes[i] == "integrate_gan_conditional_cnn"):
                # nextStateImg = Reshape((self._settings['terrain_shape'][0], self._settings['terrain_shape'][1], self._settings['terrain_shape'][2]))(self._ResultState)
                nextStateImg = self._ResultState
                network = Concatenate()([network, nextStateImg])
            elif ( layer_sizes[i] == "integrate_gan_conditional"):
                network = Concatenate()([network, self._ResultState])
            elif ( layer_sizes[i] == "mark_middle"):
                    self._networkMiddle = network
            elif ( layer_sizes[i] == "merge_features"):
                # network = Flatten()(network)
                if ('split_terrain_input' in self._networkSettings 
                    and self._networkSettings['split_terrain_input']):
                    network = Concatenate(axis=1)([networkVel_x, networkVel_y, network, characterFeatures])
                else:
                    network = Concatenate(axis=1)([network, characterFeatures])
            elif ( layer_sizes[i] == "flatten_features"):
                    network = Flatten()(network)
                    if ('split_terrain_input' in self._networkSettings 
                    and self._networkSettings['split_terrain_input']):
                        networkVel_x = Flatten()(networkVel_x)
                        networkVel_y = Flatten()(networkVel_y)
            else:
                
                network = Dense(layer_sizes[i],
                                kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(network)
                network = getKerasActivation(self._settings['activation_type'])(network)
                if ('split_terrain_input' in self._networkSettings 
                    and self._networkSettings['split_terrain_input']):
                    networkVel_x = Dense(layer_sizes[i],
                                kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(networkVel_x)
                    networkVel_x = getKerasActivation(self._settings['activation_type'])(networkVel_x)
                    networkVel_y = Dense(layer_sizes[i],
                                    kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                    bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(networkVel_y)
                    networkVel_y = getKerasActivation(self._settings['activation_type'])(networkVel_y)
            if ( self._dropout_p > 0.001 ):
                network = Dropout(rate=self._dropout_p)(network)
                
        return network
    
    def train(self, states, actions):
        score = self._actor.fit([states], actions,
              epochs=1, batch_size=32,
              verbose=0,
              shuffle=True
              # callbacks=[early_stopping],
              )
        return score.history['loss'][0]
        
    def compile(self):
        self._actor = Model(inputs=[self.getStateSymbolicVariable()], 
                            outputs=self._actor)
        print("Net summary: ", self._actor.summary())
        sgd = keras.optimizers.Adam(lr=np.float32(0.0001), beta_1=np.float32(0.95), beta_2=np.float32(0.999), epsilon=np.float32(0.000001), decay=np.float32(0.0))
        self._actor.compile(loss='mse', optimizer=sgd)
        
        self._f = self._actor([self.getStateSymbolicVariable()])
        self._forward = K.function([self.getStateSymbolicVariable(), K.learning_phase()], [self._f])
        
    def predict(self, state):
        state = [np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])]
        # print("state:", state)
        y = scale_state(self._forward([state,0])[0][0], self._action_bounds)
        # print("y: ", y)
        return y
    def predictWithDropout(self, state):
        state = [np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])]
        y = scale_state(self._forward([state,1])[0][0], self._action_bounds)
        return y    

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
    
    def getTransformationDynamicsNetwork(self):
        return self._trans
    
    def setTransformationDynamicsNetwork(self, net):
        self._trans = net
    
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
    
    def reset(self):
        self._actor.reset_states()
        self._critic.reset_states()