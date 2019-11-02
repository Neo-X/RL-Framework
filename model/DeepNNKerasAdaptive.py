
# import theano
# from theano import tensor as T
import numpy as np
# import lasagne
import sys
from dill.settings import settings
from tensorflow.python.layers.normalization import BatchNorm
from model.ModelUtil import *
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten, Lambda
from keras.layers import LSTM, LSTMCell, GRU, ZeroPadding1D
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
from pydoc import locate

        
### It is complicated to serialize lambda functions, better to define a function
def keras_slice(x, begin, end):
    # if (len(keras.backend.int_shape(x)) > 2):
    #     return x[:, :, begin:end]
    # else:
    return x[:,begin:end]

### It is complicated to serialize lambda functions, better to define a function
def keras_slice_3d(x, begin, end):
    # if (len(keras.backend.int_shape(x)) > 2):
    #     return x[:, :, begin:end]
    # else:
    return x[:,:,begin:end]

class DeepNNKerasAdaptive(ModelInterface):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=False, stateName="State", resultStateName="ResultState"):

        super(DeepNNKerasAdaptive,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=print_info)
        self._networkSettings = {}
        self._slices = {}
        if ("network_settings" in settings_):
            self._networkSettings = settings_["network_settings"]
            
        self._sequence_length = 1
        self._lstm_batch_size = 1
        if ( "lstm_batch_size" in settings_):
            self._lstm_batch_size = settings_["lstm_batch_size"][1]
        self._stateful_lstm = False
        if ("train_LSTM_stateful" in self._settings
            and (self._settings["train_LSTM_stateful"])):
            self._stateful_lstm = True
        isRNN = False
        ### data types for model
        # self._State = K.variable(value=np.random.rand(self._batch_size,self._state_length) ,name=stateName)
        # self._State = keras.layers.Input(shape=(self._state_length,), name=stateName, batch_shape=(32,self._state_length))
        if (("train_LSTM" in self._settings)
                and (self._settings["train_LSTM"] == True)):
            if ("simulation_model" in self._settings and
                (self._settings["simulation_model"] == True)):
                if (self._stateful_lstm):
                    self._State = keras.layers.Input(shape=(self._sequence_length, self._state_length), batch_shape=(1, 1, self._state_length), name=stateName)
                else:
                    self._State = keras.layers.Input(shape=(None, self._state_length), name=stateName)
            else:
                if (self._stateful_lstm):
                    self._State = keras.layers.Input(shape=(self._sequence_length, self._state_length), batch_shape=(self._lstm_batch_size, self._sequence_length, self._state_length), name=stateName)
                else:
                    self._State = keras.layers.Input(shape=(None, self._state_length), name=stateName)
        else:
            self._State = keras.layers.Input(shape=(self._state_length,), name=stateName)
            self._State_backup = self._State
            

        ### Apparently after the first layer the patch axis is left out for most of the Keras stuff...
        # input = Input(shape=(self._state_length,))
        self._stateInput = self._State
        # input2 = Input(shape=(self._action_length,))
        self._Action = keras.layers.Input(shape=(self._action_length,), name="Action") 
        self._actionInput = self._Action
        # input.trainable = True        
        if (("train_LSTM_Critic" in self._settings)
                and (self._settings["train_LSTM_Critic"] == True)):
            if ("simulation_model" in self._settings and
                (self._settings["simulation_model"] == True)):
                if (self._stateful_lstm):
                    self._ResultState = keras.layers.Input(shape=(self._sequence_length, 
                                                                  self._result_state_length), 
                                                           batch_shape=(1, 1, self._state_length), name=resultStateName+"_SIM")
                else:
                    self._ResultState = keras.layers.Input(shape=(None, self._result_state_length), name=resultStateName+"_SIM")
            else:
                if (self._stateful_lstm):
                    self._ResultState = keras.layers.Input(shape=(self._sequence_length, 
                                                                  self._result_state_length), 
                                                           batch_shape=(self._lstm_batch_size, self._sequence_length, self._state_length), name=resultStateName)
                else:
                    self._ResultState = keras.layers.Input(shape=(None, self._result_state_length), name=resultStateName)
            # self._stateInput = self._ResultState 
        else:
            self._ResultState = keras.layers.Input(shape=(self._result_state_length,), name=resultStateName)
            if (
                (("train_LSTM" in self._settings)
                and (self._settings["train_LSTM"] == True))
                and
                ( not (("train_LSTM_Critic" in self._settings)
                and (self._settings["train_LSTM_Critic"] == True)))
                ):
                print("Training a policy lstm but not a critic one.")
                self._stateInput = self._ResultState
                
        if ( (("train_LSTM" in self._settings)
                and (self._settings["train_LSTM"] == True)) 
            or
            (("train_LSTM_Critic" in self._settings)
                and (self._settings["train_LSTM_Critic"] == True))
            ):
            self._Reward = keras.layers.Input(shape=(self._sequence_length,1), name="Reward")
        else:
            self._Reward = keras.layers.Input(shape=(1,), name="Reward")
        
        self._data_format_ = keras.backend.image_data_format()
        if ("image_data_format" in self._networkSettings ):
            self._data_format_ = self._networkSettings["image_data_format"]
        
        if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
            print ("self._stateInput ",  self._stateInput)
        inputAct = self._State
        
        if (("train_LSTM" in self._settings)
                and (self._settings["train_LSTM"] == True)):
            # self._taskFeatures = self._State
            isRNN = True
        """
        if (self._settings['num_terrain_features'] > 0):
            inputAct = self._taskFeatures
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
            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                print ("Actor Network layer sizes: ", layer_sizes)
            networkAct = inputAct
            
            if ( self._dropout_p > 0.001 
                 and ("use_dropout_in_actor" in self._settings 
                      and (self._settings["use_dropout_in_actor"] == True)) ):
                networkAct = Dropout(rate=self._dropout_p)(networkAct)
                
            if ( "use_decoder" in self._settings
                  and (self._settings["use_decoder"] == True) ):
                networkAct = keras.layers.Input(shape=(self._settings["encoding_vector_size"],), name="FD_Encoding_State")
                # networkAct = Dropout(rate=self._dropout_p)(networkAct)
                self._State = networkAct 
            
            networkAct = self.createSubNetwork(networkAct, layer_sizes, isRNN=isRNN, stateName=stateName, resultStateName=resultStateName)
            
            # inputAct.trainable = True
            
            actor_out_init_layer_scale = 1.0
            if ("actor_out_init_layer_scale" in self._settings):
                actor_out_init_layer_scale = self._settings["actor_out_init_layer_scale"]
            networkAct_ = networkAct
            if (layer_sizes[-1] != "merge_state_types"
                and ( not ("network_leave_off_end" in self._settings 
                           and (self._settings["network_leave_off_end"] == True )))):
                
                networkAct = Dense(n_out, 
                                   kernel_regularizer=regularizers.l2(self._settings['regularization_weight']),
                                   bias_regularizer=regularizers.l2(self._settings['regularization_weight'])
                                   # , kernel_initializer=(keras.initializers.VarianceScaling(scale=actor_out_init_layer_scale,
                                   # mode='fan_avg', distribution='uniform', seed=None) )
                                   )(networkAct)
                networkAct = self.getActivationType(self._settings['last_policy_layer_activation_type'])(networkAct)
            """
            if (("train_LSTM" in self._settings)
                and (self._settings["train_LSTM"] == True)):
                networkAct = networkAct = Reshape((1, 64))(networkAct)
            """
            self._actor = networkAct
                    
            if (self._settings['use_stochastic_policy']):
                if ("split_single_net_earlier" in self._networkSettings and 
                    self._networkSettings["split_single_net_earlier"] == True):
                    self._second_last_layer = Dense(layer_sizes[len(layer_sizes)-1], 
                                       kernel_regularizer=regularizers.l2(self._settings['regularization_weight']),
                                       bias_regularizer=regularizers.l2(self._settings['regularization_weight']))(self._second_last_layer)
                    self._second_last_layer = self.getActivationType(self._settings['policy_activation_type'])(self._second_last_layer)
                    with_std = Dense(n_out, 
                                kernel_regularizer=regularizers.l2(self._settings['regularization_weight']),
                                bias_regularizer=regularizers.l2(self._settings['regularization_weight']),
                                kernel_initializer=(keras.initializers.VarianceScaling(scale=0.01,
                                mode='fan_avg', distribution='uniform', seed=None) )
                                     )(self._second_last_layer)
                else:
                    with_std = Dense(n_out, 
                                kernel_regularizer=regularizers.l2(self._settings['regularization_weight']),
                                bias_regularizer=regularizers.l2(self._settings['regularization_weight']),
                                kernel_initializer=(keras.initializers.VarianceScaling(scale=0.01,
                                mode='fan_avg', distribution='uniform', seed=None) )
                                     )(networkAct_)
                with_std = self.getActivationType(self._settings['_last_std_policy_layer_activation_type'])(with_std)
                # with_std = networkAct = Dense(self._action_length, kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(networkAct)
                self._actor = keras.layers.concatenate(inputs=[self._actor, with_std], axis=-1)
                
            # input_ = [self._stateInput, self._actionInput, self._Reward]
            # input_ = self._stateInput
            # self._actor = Model(input=input_, output=self._actor)
            # self._actor = Model(input=[self._stateInput, self._actionInput], output=self._actor)
            # print("Actor summary: ", self._actor.summary())
        # self._taskFeatures = self._ResultState
        # self._taskFeatures = self._ResultState
        isRNN = False
        network = self._stateInput
        if (("train_LSTM_Critic" in self._settings)
                and (self._settings["train_LSTM_Critic"] == True)):
            # self._taskFeatures = self._ResultState
            network = self._ResultState
            isRNN = True
            
        layer_sizes = self._settings['critic_network_layer_sizes']
        if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
            print ("Critic Network layer sizes: ", layer_sizes)

        if ("using_encoder_decoder" in self._settings
            and (self._settings["using_encoder_decoder"] == True)):
            print (self._actor)
            print ("keras.backend.int_shape(self._actor): ", keras.backend.int_shape(self._actor))
            if (self._stateful_lstm):
                network = keras.layers.Input(shape=(self._sequence_length, 
                      keras.backend.int_shape(self._actor)[1]), 
                       batch_shape=(self._lstm_batch_size, self._sequence_length, keras.backend.int_shape(self._actor)[-1]), name="Encoding_State")
            else:
                network = keras.layers.Input(shape=(None, keras.backend.int_shape(self._actor)[1]), name="Encoding_State")
            print ("Encoder input: ", network)
            self._ResultState = network
            
        if ( "use_centralized_critic" in self._settings
             and (self._settings["use_centralized_critic"] == True)
             and False):
            network = keras.layers.Input(shape=(len(self._settings["state_bounds"][0]),), name="Centralized_Critic_State")
            self._ResultState = network
        network = self.createSubNetwork(network, layer_sizes, isRNN=isRNN, stateName=stateName, resultStateName=resultStateName)
        
            
        if ( "use_single_network" in self._settings and 
             (self._settings['use_single_network'] == True)):
            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                print ("Using a single network model")
            if ("split_single_net_earlier" in self._networkSettings and 
                    self._networkSettings["split_single_net_earlier"] == True):
                self._second_last_layer_ = Dense(layer_sizes[len(layer_sizes)-1],
                                kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(self._second_last_layer)
                self._second_last_layer_ = self.getActivationType(self._settings['activation_type'])(self._second_last_layer_)
                networkAct = self._second_last_layer_       
                # networkAct_ = networkAct
                networkAct = Dense(self._action_length, 
                                   kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                   bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(self._second_last_layer_)
                networkAct = self.getActivationType(self._settings['last_policy_layer_activation_type'])(networkAct)
            else:
                networkAct = network       
                networkAct_ = networkAct
                networkAct = Dense(self._action_length, 
                                   kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                   bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(networkAct)
                networkAct = self.getActivationType(self._settings['last_policy_layer_activation_type'])(networkAct)
    
            self._actor = networkAct
                    
            if (self._settings['use_stochastic_policy']):
                if ("split_single_net_earlier" in self._networkSettings and 
                    (self._networkSettings["split_single_net_earlier"] == True)):
                    self._second_last_layer = Dense(layer_sizes[len(layer_sizes)-1],
                                kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(self._second_last_layer)
                    self._second_last_layer = self.getActivationType(self._settings['activation_type'])(self._second_last_layer)
                    
                    with_std = Dense(self._action_length, 
                                kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                kernel_initializer=(keras.initializers.VarianceScaling(scale=0.01,
                               mode='fan_avg', distribution='uniform', seed=None) ))(self._second_last_layer)
                else:
                    with_std = Dense(self._action_length, 
                                    kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                    bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                    kernel_initializer=(keras.initializers.VarianceScaling(scale=0.01,
                                   mode='fan_avg', distribution='uniform', seed=None) ))(networkAct_)
                with_std = self.getActivationType(self._settings['_last_std_policy_layer_activation_type'])(with_std)
                # with_std = networkAct = Dense(self._action_length, kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(networkAct)
                self._actor = keras.layers.concatenate(inputs=[self._actor, with_std], axis=-1)
                
            if ("use_viz_for_policy" in self._settings 
                and self._settings["use_viz_for_policy"] == True):
                self._trans = Dense(self._settings["dense_state_size"],
                                kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(self._networkMiddle)
                self._trans = self.getActivationType(self._settings['last_fd_layer_activation_type'])(self._trans)
                
            if ("forward_dynamics_model_type" in self._settings 
                and (self._settings["forward_dynamics_model_type"] == "SingleNet")
                and (self._settings['use_single_network'] == True)):
                self._forward_dynamics_net = Dense(self._settings["fd_network_layer_sizes"][-1],
                                kernel_regularizer=regularizers.l2(self._settings['regularization_weight']),
                                bias_regularizer=regularizers.l2(self._settings['regularization_weight']))(self._networkMiddle)
                self._forward_dynamics_net = self.getActivationType(self._settings['last_fd_layer_activation_type'])(self._forward_dynamics_net)
                
                self._reward_net = Dense(self._settings["fd_network_layer_sizes"][-1],
                                kernel_regularizer=regularizers.l2(self._settings['regularization_weight']),
                                bias_regularizer=regularizers.l2(self._settings['regularization_weight']))(self._networkMiddle)
                self._reward_net = self.getActivationType(self._settings['last_fd_layer_activation_type'])(self._reward_net)
                
            # self._actor = Model(input=self._stateInput, output=self._actor)
            # print("Actor summary: ", self._actor.summary())
            ### Render a nice graph of the network
        if ( not ("critic_network_leave_off_end" in self._settings
            and (self._settings["critic_network_leave_off_end"] == True))):

            if (len(keras.backend.int_shape(network)) > 2):
                ### THis is an LSTM use time distributed layer
                input_ = keras.layers.Input(shape=(keras.backend.int_shape(network)[-1],), name="Critic_subnet")
                print ("*** Critic subnet input shape: ", repr(keras.backend.int_shape(input_)))
                subnet = Dense(1,
                           kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                           bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(input_)
                subnet = Model(inputs=input_, outputs=subnet)
                if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                    print("Critic Subnet summary")
                    subnet.summary()
                # subnet = Dense(8)
                ### Create a model (set of layers to distribute) pass in the original input to that model
                network = keras.layers.TimeDistributed(subnet, input_shape=(None, 1, self._state_length))(network)
            else:
                network= Dense(1,
                           kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                           bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(network)
            
            if ("last_critic_layer_activation_type" in self._settings):
                self._critic = self.getActivationType(self._settings['last_critic_layer_activation_type'])(network)
            else:
                self._critic = Activation('linear')(network)
        else:
            self._critic = network
    
    def getActivationType(self, type_name):
        """
            Compute a particular type of actiation to use
        """
        import keras.layers
        # print ("Getting keras activation")
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
                
    def createSubNetwork(self, input, layer_info, isRNN=False, stateName="State", resultStateName="ResultState"):
        if ( "network_description_type" in self._settings and
             (self._settings["network_description_type"] == "json")):
            net = self._createSubNetworkFromJSON(input, layer_info, isRNN=False, stateName="State", resultStateName="ResultState")
        else:  
            net = self._createSubNetwork(input=input, layer_info=layer_info, 
                                         isRNN=isRNN, stateName=stateName, 
                                         resultStateName=resultStateName)
        
        return net
    
    def _createSubNetworkFromJSON(self, input, layer_info, isRNN=False, stateName="State", resultStateName="ResultState"):
        
        network = input
        for i in range(len(layer_info)):
            # layer_desc = dict(layer_info[i])
            self._second_last_layer = network
            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                print ("Layer info: ", type(layer_info[i]))
                print ("input: ", repr(network))
            layer_parms = copy.deepcopy(layer_info[i])
            layer_parms.pop("layer_type", None)
            layer_parms.pop("activation_type", None) 
            if (layer_info[i]["layer_type"] == "Dense"):
                network = Dense( kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                    bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                 **layer_parms)(network)
            elif (layer_info[i]["layer_type"] == "Reshape"):
                print(layer_parms)
                print(network)
                network = Reshape(**layer_parms)(network)
            elif (layer_info[i]["layer_type"] == "Padding"):
                network = Reshape([-1, 1])(network)
                network = ZeroPadding1D(
                    padding=(
                        layer_info[i]["before_pad_size"],
                        layer_info[i]["after_pad_size"]))(network)
                network = Reshape([-1])(network)
            elif (layer_info[i]["layer_type"] == "Flatten"):
                network = Flatten(**layer_parms)(network)
            elif (layer_info[i]["layer_type"] == "Dropout"):
                network = Dropout(**layer_parms)(network)
            elif (layer_info[i]["layer_type"] == "GaussianNoise"):
                network = keras.layers.GaussianNoise(**layer_parms)(network)
            elif (layer_info[i]["layer_type"] == "Input"):
                if ("slice_label" in layer_info[i]):  
                    input_ = self._slices[layer_info[i]["slice_label"]]
                    network = input_
                else:   
                    if ("flag" in layer_info[i] and
                        layer_info[i]["flag"] == "fd_input"):
                        input_ = keras.layers.Input(shape=(layer_info[i]["shape"][-1],), name=stateName)
                        network = input_
                        self._State_FD = input_
                    elif ("flag" in layer_info[i] and
                        layer_info[i]["flag"] == "action"):
                        input_act = keras.layers.Input(shape=(layer_info[i]["shape"][-1],), name="Action_Stacked")
                        self._slices["action"] = input_act
                        self._actionInput = input_act
                        self._Action = input_act
                    else:
                        if (("train_LSTM" in self._settings)
                                and (self._settings["train_LSTM"] == True)):
                            if ("simulation_model" in self._settings and
                            (self._settings["simulation_model"] == True)):
                                if (self._stateful_lstm):
                                    input_ = keras.layers.Input(shape=(self._sequence_length, layer_info[i]["shape"][-1]), batch_shape=(1, 1, self._state_length), name=stateName)
                                else:
                                    input_ = keras.layers.Input(shape=(None, layer_info[i]["shape"][-1]), name=stateName)
                            else:
                                if (self._stateful_lstm):
                                    input_ = keras.layers.Input(shape=(self._sequence_length, layer_info[i]["shape"][-1]), batch_shape=(self._lstm_batch_size, self._sequence_length, self._state_length), name=stateName)
                                else:
                                    input_ = keras.layers.Input(shape=(None, layer_info[i]["shape"][-1]), name=stateName)
                        else:
                            if (len(layer_info[i]["shape"]) > 1): ### Hack so that RNN layers don't complain about none shapes
                                input_ = keras.layers.Input(shape=(None, layer_info[i]["shape"][-1]), name=stateName)
                            else:
                                input_ = keras.layers.Input(shape=(layer_info[i]["shape"][0],), name=stateName)
                                self._State = input_ 
                        network = input_
                        self._State_ = input_   
                print ("self._State_: ", repr(network))
            elif (layer_info[i]["layer_type"] == "BatchNormalization"):
                network = keras.layers.BatchNormalization(**layer_parms)(network)
            elif (layer_info[i]["layer_type"] == "LayerNormalization"):
                from keras_layer_normalization import LayerNormalization
                network = LayerNormalization(**layer_parms)(network)
            elif ( layer_info[i]["layer_type"] == "activation"):
                network = self.getActivationType(layer_info[i]["activation_type"])(network)      
            elif ( layer_info[i]["layer_type"] == "integrate_actor_part"):
                if ("action" in self._slices):
                    input_act = self._slices["action"]
                else:
                    input_act = self._actionInput
                network = Concatenate()([network, input_act])          
            elif (layer_info[i]["layer_type"] == "Concatenate"):
                print ("concatenating: ", repr(network))
                if ("slice_label" in layer_info[i]):
                    print ("concatenating slice: ", repr(self._slices[layer_info[i]["slice_label"]]))
                    network = Concatenate(axis=-1)([network, self._slices[layer_info[i]["slice_label"]]])
                else:
                    print ("concatenating _characterFeatures: ", repr(_characterFeatures))
                    network = Concatenate(axis=-1)([network, _characterFeatures])
            elif (layer_info[i]["layer_type"] == "GRU"):
                network = GRU(
                              kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                    bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                    recurrent_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                    **layer_parms)(network)
            elif (layer_info[i]["layer_type"] == "LSTM"):
                network = LSTM(
                              kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                    bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                    recurrent_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                    **layer_parms)(network)
            elif (layer_info[i]["layer_type"] == "TimeDistributedConv"):
                    ### https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
                    if ( layer_info[i]["net_info"] == "fd" ):
                        subnet = self._actor
                        print ("self._State_backup: ", self._State_backup)
                        print ("*** subnet input shape: ", repr(keras.backend.int_shape(self._State_backup)))
                        subnet = Model(inputs=self._State_backup, outputs=subnet)
                        # print("Subnet summary")
                        # subnet.summary()
                    else:
                        print ("*** net input shape: ", repr(keras.backend.int_shape(network)))
                        input_ = keras.layers.Input(shape=(keras.backend.int_shape(network)[-1],), name="State_Conv")
                        print ("*** subnet input shape: ", repr(keras.backend.int_shape(input_)))
                        #if (layer_sizes[i]["isRNN"] == True):
                        #    subnet = self.createSubNetwork(input_, layer_info[i]["net_info"], isRNN=True)
                        #else:
                        subnet = self.createSubNetwork(input_, layer_info[i]["net_info"], isRNN=False)
                        subnet = Model(inputs=input_, outputs=subnet)
                        # print("Subnet summary")
                        # subnet.summary()
                        
                    ### Create a model (set of layers to distribute) pass in the original input to that model
                    print ("*** subnet input ", network, " shape: ", repr(keras.backend.int_shape(network)))
                    print ("self._state_length: ", self._state_length)
                    network = keras.layers.TimeDistributed(subnet, input_shape=(None, 1, self._state_length))(network)
            elif (layer_info[i]["layer_type"] == "slice"):
                ### Need to make sure to create end slice first to not overwrite network then try and slice from it again...
                if ("slice_index" in layer_info[i]):
                    
                    input__ = network
                    if ("slice_input" in layer_info[i]):
                        input__ = self._slices[layer_info[i]["slice_input"]]
                    state_length_ = keras.backend.int_shape(input__)[1]
                    print ("slice, state_length_: ", state_length_)
                    # sys.exit()
                    self._slices[layer_info[i]["slice_label"]] = Lambda(keras_slice, output_shape=(state_length_-layer_info[i]["slice_index"],),
                                       arguments={'begin': layer_info[i]["slice_index"], 
                                                  'end': state_length_},
                                       name=layer_info[i]["slice_label"])(input__)
                    print ("new slice network shape: ", repr(self._slices[layer_info[i]["slice_label"]]))
                    input__ = Lambda(keras_slice, output_shape=(layer_info[i]["slice_index"],),
                                  arguments={'begin': 0, 'end': layer_info[i]["slice_index"]}
                                  )(input__)
                    if ("slice_input" in layer_info[i]):
                        self._slices[layer_info[i]["slice_input"]] = input__
                        print ("new network shape: ", repr(input__))
                    else:
                        network = input__
                        print ("new network shape: ", repr(network))
                    # sys.exit()
                else:
                    _characterFeatures = Lambda(keras_slice, output_shape=(self._state_length-self._settings['num_terrain_features'],),
                                           arguments={'begin': self._settings['num_terrain_features'], 
                                                      'end': self._state_length})(network)
                    network = Lambda(keras_slice, output_shape=(self._settings['num_terrain_features'],),
                                            arguments={'begin': 0, 'end': self._settings['num_terrain_features']})(network)
                    # sys.exit()
            elif ( layer_info[i]["layer_type"] == "coordconv2d" ):
                  network = CoordinateChannel2D()(network)
            elif ( layer_info[i]["layer_type"] == "conv2d" ):
                network = keras.layers.Conv2D( kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                                 bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                                 data_format=self._data_format_,
                                                 **layer_parms)(network)
            elif ( layer_info[i]["layer_type"] == "deconv2d" ):
                network = keras.layers.Conv2DTranspose(
                                                     kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                                     bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                                     data_format=self._data_format_,
                                                     **layer_parms)(network)
            elif ( layer_info[i]["layer_type"] == "sepconv2d" ):
                network = keras.layers.SeparableConv2D( kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                                 bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                                 data_format=self._data_format_,
                                                 **layer_parms)(network)
            elif (layer_info[i]["layer_type"] == "subnet"):
                input_ = self._slices[layer_info[i]["input"]]
                subnet = self.createSubNetwork(input_, layer_info[i]["layer_info"], isRNN=False)
                ### build model, maybe?
                self._slices[layer_info[i]["output_label"]] = subnet
            else:
                print ("layer type: ", layer_info[i]["layer_type"])
                model_ = locate(layer_info[i]["layer_type"])
                print ("layer model: ", model_)
                if (issubclass(model_, keras.layers)):
                    network = model_(
                                     kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                     bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                     data_format=self._data_format_,
                                     **layer_parms)(network)
                    
                                                     
        return network
    
    def _createSubNetwork(self, input, layer_info, isRNN=False, stateName="State", resultStateName="ResultState"):
        
        network = input
        
        if (len(keras.backend.int_shape(input)) == 2
            and (self._settings['num_terrain_features'] > 0)
            and (not (isRNN))
            # and (not ("using_encoder_decoder" in self._settings
            #          and (self._settings["using_encoder_decoder"] == True)))
            ): ### Don't do this for RNNs...
            if ('split_terrain_input' in self._networkSettings 
                and self._networkSettings['split_terrain_input']):
                mid = int((self._settings['num_terrain_features']/3) * 2)
                velFeatures_x = Lambda(keras_slice, output_shape=(int(self._settings['num_terrain_features']/3),),
                                  arguments={'begin': 0, 'end': int(mid/2)})(input)
                velFeatures_y = Lambda(keras_slice, output_shape=(int(self._settings['num_terrain_features']/3),),
                                  arguments={'begin': int(mid/2), 'end': mid})(input)
                network = Lambda(keras_slice, output_shape=(int(self._settings['num_terrain_features']/3),),
                                  arguments={'begin': mid, 'end': self._settings['num_terrain_features']})(input)
            else:
                network = Lambda(keras_slice, output_shape=(self._settings['num_terrain_features'],),
                                  arguments={'begin': 0, 'end': self._settings['num_terrain_features']})(input)
                                  
            print ("input: ", repr(input))
            print ("Number charater features: ", self._state_length-self._settings['num_terrain_features'])
            print ("self._settings['num_terrain_features']: ", self._settings['num_terrain_features'], " self._state_length: ", self._state_length)
            
            if ("fd_use_multimodal_state" in self._settings
                and (self._settings["fd_use_multimodal_state"] == True)):
                ### Pull out just cam velocity features
                _characterFeatures = Lambda(keras_slice, output_shape=((self._state_length - self._settings["dense_state_size"])-self._settings['num_terrain_features'],),
                                       arguments={'begin': self._settings['num_terrain_features'], 
                                                  'end': self._state_length - self._settings["dense_state_size"]})(input)
            else:
                _characterFeatures = Lambda(keras_slice, output_shape=(self._state_length-self._settings['num_terrain_features'],),
                                       arguments={'begin': self._settings['num_terrain_features'], 
                                                  'end': self._state_length})(input)
            print ("*** _characterFeatures shape: ", repr(keras.backend.int_shape(_characterFeatures)))
            print ("*** _characterFeatures shape: ", repr(_characterFeatures))
            # sys.exit()
        # print ("**********************self._taskFeatures shape: ", repr(keras.backend.int_shape(input)))
        print ("**********************self._taskFeatures shape: ", repr(input))
        
        layer_sizes = layer_info
        for i in range(len(layer_sizes)):
            self._second_last_layer = network
            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                print ("layer_sizes[",i,"]: ", layer_sizes[i])
                # print ("shape[", i, "]: ", repr(keras.backend.int_shape(network)))
                print ("tensor ", network, "shape[", i, "]: ", repr(keras.backend.int_shape(network)))
            if type(layer_sizes[i]) is list:
                if (layer_sizes[i][0] == "GRU" or
                      layer_sizes[i][0] == "LSTM"):
                    # print ("layer.output_shape: ", keras.backend.shape(network))
                    # network = Reshape((1, layer_sizes[i][1]))(network)
                    rnn_dropout=0.0
                    if (len(layer_sizes[i]) >= 4):
                        rnn_dropout = float (layer_sizes[i][3])
                        print("Recurrent Dropout: ", rnn_dropout)
                    ### the RNN can return a sequence of outputs that can be used with a sequence of target values
                    rnn_return_sequence = False
                    rnn_return_state = False
                    if ((len(layer_sizes[i]) >= 5)
                        and (layer_sizes[i][4] == True)):
                        rnn_return_sequence = True
                        print("rnn_return_sequence: ", rnn_return_sequence)
                    if ((len(layer_sizes[i]) >= 6)
                        and (layer_sizes[i][5] == True)):
                        rnn_return_state = True
                        print("rnn_return_state: ", rnn_return_state)
                    if (layer_sizes[i][0] == "LSTM"):
                        network = LSTM(layer_sizes[i][2], stateful=self._stateful_lstm, 
                                  recurrent_dropout=rnn_dropout,
                                  return_sequences=rnn_return_sequence,
                                  return_state=rnn_return_state)(network)
                    else:
                        network = GRU(layer_sizes[i][2], stateful=self._stateful_lstm, 
                                  recurrent_dropout=rnn_dropout,
                                  return_sequences=rnn_return_sequence,
                                  return_state=rnn_return_state)(network)
                elif (layer_sizes[i][0] == "Reshape"):
                    network = Reshape(layer_sizes[i][1])(network)
                elif (layer_sizes[i][0] == "integrate_gan_conditional_cnn"):
                    subnet = self.createSubNetwork(self._ResultState, layer_sizes[i][1], isRNN=False)
                    # nextStateImg = self._ResultState
                    network = Concatenate()([network, subnet] )
                    
                elif (layer_sizes[i][0] == "Dense"):
                    if (len(layer_sizes[i]) > 3):
                        print ("Adding activity regularizer to dense layer")
                        network = Dense(layer_sizes[i][1],
                                    kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                    bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                    activity_regularizer=regularizers.l2(layer_sizes[i][3]))(network)
                    else:
                        network = Dense(layer_sizes[i][1],
                                    kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                    bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(network)
                                        
                    if (len(layer_sizes[i]) > 2):
                        network = self.getActivationType(layer_sizes[i][2])(network)
                    else:
                        network = self.getActivationType(self._settings['activation_type'])(network)
                elif (layer_sizes[i][0] == "TimeDistributedConv"):
                    ### https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
                    # input_ = keras.layers.Input(shape=(None, layer_sizes[i][1][-1]), name="State_Conv")
                    if ("fd" == layer_sizes[i][2]):
                        subnet = self._actor
                        print ("self._State_backup: ", self._State_backup)
                        print ("*** subnet input shape: ", repr(keras.backend.int_shape(self._State_backup)))
                        subnet = Model(inputs=self._State_backup, outputs=subnet)
                        #if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print("Subnet summary")
                        subnet.summary()
                    else:
                        print ("*** net input shape: ", repr(keras.backend.int_shape(network)))
                        # input_ = keras.layers.Input(shape=(1, keras.backend.int_shape(network)[-1]), name="State_Conv")
                        # if ("num_terrain_features" in self._settings
                        #     and (self._settings["num_terrain_features"] > 0)):
                        #     input_ = keras.layers.Input(shape=(1, self._settings["num_terrain_features"]), name="State_Conv")
                        # else:
                        input_ = keras.layers.Input(shape=(keras.backend.int_shape(network)[-1],), name="State_Conv")
                        print ("*** subnet input shape: ", repr(keras.backend.int_shape(input_)))
                        if (layer_sizes[i][1] == True):
                            subnet = self.createSubNetwork(input_, layer_sizes[i][2], isRNN=True)
                        else:
                            subnet = self.createSubNetwork(input_, layer_sizes[i][2], isRNN=False)
                        subnet = Model(inputs=input_, outputs=subnet)
                        # if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print("Subnet summary")
                        subnet.summary()
                        
                    ### Create a model (set of layers to distribute) pass in the original input to that model
                    print ("*** subnet input ", network, " shape: ", repr(keras.backend.int_shape(network)))
                    print ("self._state_length: ", self._state_length)
                    network = keras.layers.TimeDistributed(subnet, input_shape=(None, 1, self._state_length))(network)
                    
                    # network = keras.layers.TimeDistributed(self.getActivationType(self._settings['activation_type'])(network))
                elif (layer_sizes[i][0] == "TimeDistributed"):
                    
                    if ("fd" == layer_sizes[i][2]):
                        subnet = self._actor
                        subnet = Model(inputs=self._State_backup, outputs=subnet)
                    else:
                        input_ = keras.layers.Input(shape=(self._state_length,), name="State_subnet")
                        print ("*** subnet input shape: ", repr(keras.backend.int_shape(input_)))
                        subnet = self.createSubNetwork(input_, layer_sizes[i][2], isRNN=False)
                        subnet = Model(inputs=input_, outputs=subnet)
                        if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                            print("Subnet summary")
                            subnet.summary()
                    # subnet = Dense(8)
                    ### Create a model (set of layers to distribute) pass in the original input to that model
                    network = keras.layers.TimeDistributed(subnet, input_shape=(None, 1, self._state_length))(input)
                    
                elif (layer_sizes[i][0] == "Residual"):
                    
                    print ("*** Residual subnet input shape: ", repr(keras.backend.int_shape(network)))
                    subnet = self.createSubNetwork(network, layer_sizes[i][1], isRNN=True)
                    # subnet = Dense(256)(network)
                    # subnet = Model(inputs=network, outputs=subnet)
                    # network_ = Model(inputs=network, outputs=network)
                    network = keras.layers.Add()([network, subnet])
                    # network = network +
                    """ 
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print("Residual net summary")
                        network.summary()
                    """
                    # subnet = Dense(8)
                    ### Create a model (set of layers to distribute) pass in the original input to that model
                    # network = keras.layers.TimeDistributed(subnet, input_shape=(None, 1, self._state_length))(input)
                    
                elif (layer_sizes[i][0] == "integrate_actor_part"):
                    subnet_ = self.createSubNetwork(self._actionInput, layer_sizes[i][1], isRNN=False)
                    network = Concatenate()([network, subnet_])
                elif (layer_sizes[i][0] == "max_pool"):
                        network = keras.layers.MaxPooling2D(pool_size=layer_sizes[i][1], strides=None, padding='valid', 
                                                                   data_format=self._data_format_)(network)  
                elif (layer_sizes[i][0] == "avg_pool"):
                        network = keras.layers.AveragePooling2D(pool_size=layer_sizes[i][1], strides=None, padding='valid', 
                                                                   data_format=self._data_format_)(network)  
                elif ( layer_sizes[i][0] == "dropout" ):
                    network = Dropout(rate=layer_sizes[i][1])(network)
                elif ( layer_sizes[i][0] == "batchnorm" ):
                    network = keras.layers.BatchNormalization()(network)
                elif ( layer_sizes[i][0] == "input" ):
                    if ("simulation_model" in self._settings and
                        (self._settings["simulation_model"] == True)):
                        if (self._stateful_lstm):
                            input_ = keras.layers.Input(shape=(self._sequence_length, layer_sizes[i][1][-1]), batch_shape=(1, 1, self._state_length), name=stateName)
                        else:
                            input_ = keras.layers.Input(shape=(None, layer_sizes[i][1][-1]), name=stateName)
                    else:
                        if (self._stateful_lstm):
                            input_ = keras.layers.Input(shape=(self._sequence_length, layer_sizes[i][1][-1]), batch_shape=(self._lstm_batch_size, self._sequence_length, self._state_length), name=stateName)
                        else:
                            input_ = keras.layers.Input(shape=(None, layer_sizes[i][1][-1]), name=stateName)
                    network = input_
                    self._State_ = input_   
                    print ("self._State_: ", repr(self._State_)) 
                elif ( layer_sizes[i][0] == "deconv" ):
                    network = keras.layers.Conv2DTranspose(layer_sizes[i][1], kernel_size=layer_sizes[i][2], strides=layer_sizes[i][3],
                                                     kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                                     bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                                     data_format=self._data_format_,
                                                     padding=layer_sizes[i][4])(network)
                elif ( layer_sizes[i][0] == "conv" ):
                    network = keras.layers.Conv2D(layer_sizes[i][1], kernel_size=layer_sizes[i][2], strides=layer_sizes[i][3],
                                                     kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                                     bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                                     data_format=self._data_format_,
                                                     padding=layer_sizes[i][4])(network)
                # elif ( layer_sizes[i][0] == "json" ):
                    # network = self.createSubNetwork(network, layer_info=[layer_sizes[i][1]["layer_type"]])
                elif ( layer_sizes[i][0] == "slice_features" ):
                    # network = Concatenate(axis=1)([network, _characterFeatures])
                    _characterFeatures = Lambda(keras_slice, output_shape=(int(keras.backend.int_shape(network)[-1]) - layer_sizes[i][1],),
                                   arguments={'begin': layer_sizes[i][1], 
                                              'end': int(keras.backend.int_shape(network)[-1])})(network)
                    print ("slice feature extra: ", repr(_characterFeatures))
                    network = Lambda(keras_slice, output_shape=(layer_sizes[i][1],),
                                   arguments={'begin': 0, 
                                              'end': layer_sizes[i][1]})(network)
                elif ( len(layer_sizes[i][1])> 1 ):
                    if (i == 0):
                        print ("create self._taskFeatures shape: ", repr(keras.backend.int_shape(network)))
                        if ('split_terrain_input' in self._networkSettings 
                        and self._networkSettings['split_terrain_input']):
                            if ("image_data_format" in self._networkSettings 
                                and (self._networkSettings["image_data_format"] == "channels_first")):
                                networkVel_x = Reshape((1, self._settings['terrain_shape'][1], self._settings['terrain_shape'][2]))(velFeatures_x)
                                networkVel_y = Reshape((1, self._settings['terrain_shape'][1], self._settings['terrain_shape'][2]))(velFeatures_y)
                                network = Reshape((1, self._settings['terrain_shape'][1], self._settings['terrain_shape'][2]))(network)
                            else:
                                networkVel_x = Reshape((self._settings['terrain_shape'][0], self._settings['terrain_shape'][1], 1))(velFeatures_x)
                                networkVel_y = Reshape((self._settings['terrain_shape'][0], self._settings['terrain_shape'][1], 1))(velFeatures_y)
                                network = Reshape((self._settings['terrain_shape'][0], self._settings['terrain_shape'][1], 1))(network)
                        else:    
                            network = Reshape(self._settings['terrain_shape'])(network)
                    stride = (1,1)
                    if (len(layer_sizes[i]) > 2):
                        stride = layer_sizes[i][2]
                        
                    if ("use_coordconv_layers" in self._networkSettings 
                            and (self._networkSettings["use_coordconv_layers"] == True)):
                        network = CoordinateChannel2D()(network)
                    # else:
                    network = keras.layers.Conv2D(layer_sizes[i][0], kernel_size=layer_sizes[i][1], strides=stride,
                                                     kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                                     bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                                     data_format=self._data_format_)(network)
                    network = self.getActivationType(self._settings['activation_type'])(network)
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
                        networkVel_x = self.getActivationType(self._settings['activation_type'])(networkVel_x)
                        networkVel_y = keras.layers.Conv2D(layer_sizes[i][0], kernel_size=layer_sizes[i][1], strides=stride,
                                                     kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                                     bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']), 
                                                     data_format=self._data_format_)(networkVel_y)   
                        networkVel_y = self.getActivationType(self._settings['activation_type'])(networkVel_y)         
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
                        # network = Reshape((self._state_length, 1))(self._taskFeatures)
                        network = Reshape((self._settings['num_terrain_features'], 1))(network)
                    stride_ = 1
                    if (len(layer_sizes[i]) > 2):
                        stride_ = layer_sizes[i][2]
                    if ("use_coordconv_layers" in self._networkSettings 
                            and (self._networkSettings["use_coordconv_layers"] == True)):
                        network = CoordinateChannel1D()(network)
                    network = keras.layers.Conv1D(layer_sizes[i][0], kernel_size=layer_sizes[i][1], strides=stride_,
                                                  kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                                  bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(network)
                    network = self.getActivationType(self._settings['activation_type'])(network)
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
            elif ( layer_sizes[i] == "integrate_gan_noise"):
                network = Concatenate()([network, self._Noise])
            elif ( layer_sizes[i] == "mark_middle"):
                self._networkMiddle = network
            elif ( layer_sizes[i] == "merge_features"):
                # network = Flatten()(network)
                if ('split_terrain_input' in self._networkSettings 
                    and self._networkSettings['split_terrain_input']):
                    network = Concatenate(axis=1)([networkVel_x, networkVel_y, network, _characterFeatures])
                else:
                    network = Concatenate(axis=1)([network, _characterFeatures])
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
                network = self.getActivationType(self._settings['activation_type'])(network)
                if ('split_terrain_input' in self._networkSettings 
                    and self._networkSettings['split_terrain_input']):
                    networkVel_x = Dense(layer_sizes[i],
                                kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(networkVel_x)
                    networkVel_x = self.getActivationType(self._settings['activation_type'])(networkVel_x)
                    networkVel_y = Dense(layer_sizes[i],
                                    kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']),
                                    bias_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(networkVel_y)
                    networkVel_y = self.getActivationType(self._settings['activation_type'])(networkVel_y)
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
        if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
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
        return self._State
    def getActionSymbolicVariable(self):
        return self._Action
    def getResultStateSymbolicVariable(self):
        return self._ResultState
    def getRewardSymbolicVariable(self):
        return self._Reward
    def getTargetsSymbolicVariable(self):
        return self._Target
    
    def reset(self):
        self._actor.reset_states()
        self._critic.reset_states()