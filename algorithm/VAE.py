import numpy as np
# import lasagne
import sys
from dill.settings import settings
sys.path.append('../')
from model.ModelUtil import *
from model.LearningUtil import loglikelihood, loglikelihoodMEAN, kl, entropy, flatgrad, zipsame, get_params_flat, setFromFlat, likelihood, loglikelihoodMEAN
from model.LearningUtil import loglikelihood, likelihood, likelihoodMEAN, kl, kl_D, entropy, flatgrad, zipsame, get_params_flat, setFromFlat
from keras.optimizers import SGD
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K
import keras
from keras.layers import RepeatVector
from keras.models import Sequential, Model
from algorithm.SiameseNetwork import *
from util.SimulationUtil import createForwardDynamicsNetwork
from keras.losses import mse, binary_crossentropy

def euclidean_distance_fd2(vects):
    x, y = vects
    return K.sum(K.square(x - y), axis=-1, keepdims=True)

def l1_distance_fd2(vects):
    x, y = vects
    return K.sum(K.abs(x - y), axis=-1, keepdims=True)

def eucl_dist_output_shape_fd2(shapes):
    shape1, shape2 = shapes
    return (shape1[0], shape1[1], 1)


def vae_loss(network_vae, network_vae_log_var):
    def _vae_loss(y_true, y_pred):
        reconstruction_loss_a = mse(y_true, y_pred)
        reconstruction_loss_a *= 4096
        kl_loss = 1 + network_vae_log_var - K.square(network_vae) - K.exp(network_vae_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss_a + kl_loss)
        return vae_loss

# reparameterization trick from Keras example
# https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

class VAE(SiameseNetwork):
    
    def __init__(self, model, state_length, action_length, state_bounds, action_bounds, settings_, reward_bounds=0, print_info=False):

        super(SiameseNetwork,self).__init__(model, state_length, action_length, state_bounds, action_bounds, reward_bounds, settings_)
        self._model = model
        self._learning_rate = self.getSettings()["fd_learning_rate"]
        self._regularization_weight = 1e-6
        
        self._distance_func = euclidean_distance
        self._distance_func_np = euclidean_distance_np
        if ( "fd_distance_function" in self.getSettings()
             and (self.getSettings()["fd_distance_function"] == "l1")):
            print ("Using ", self.getSettings()["fd_distance_function"], " distance metric for siamese network.")
            self._distance_func = l1_distance
            self._distance_func_np = l1_distance_np
        condition_reward_on_result_state = False
        self._train_combined_loss = False
        
        inputs_ = [self._model.getStateSymbolicVariable()] 
        print ("farward dynamics shape: ", repr(self._model._forward_dynamics_net))
        self._model._forward_dynamics_z_mean = keras.layers.Dense(self.getSettings()["encoding_vector_size"], activation = 'linear')(self._model._forward_dynamics_net)
        self._model._forward_dynamics_z_log_var = keras.layers.Dense(self.getSettings()["encoding_vector_size"], activation = 'sigmoid')(self._model._forward_dynamics_net)
        self._model._forward_dynamics_z = keras.layers.Lambda(sampling, output_shape=(self.getSettings()["encoding_vector_size"],), name='z')([self._model._forward_dynamics_z_mean, 
                                                                   self._model._forward_dynamics_z_log_var])
        
        self._model._forward_dynamics_net = Model(inputs=inputs_, outputs=[self._model._forward_dynamics_z_mean, 
                                                                           self._model._forward_dynamics_z_log_var,
                                                                           self._model._forward_dynamics_z]
                                                                           )
        if (print_info):
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print("FD Net summary: ", self._model._forward_dynamics_net.summary())
        
        if ("force_use_actor_state_for_critic" in self._settings
            and (self._settings["force_use_actor_state_for_critic"] == True)):
            inputs_ = [self._model.getStateSymbolicVariable()]
        else:  
            inputs_ = [self._model.getResultStateSymbolicVariable()]
        print ("******** self._model._State_: ", repr(self._model._State_)) 
        self._model._reward_net = Model(inputs=self._model._State_, outputs=self._model._reward_net)
        if (print_info):
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print("FD Reward Net summary: ", self._model._reward_net.summary())

        settings__ = copy.deepcopy(self.getSettings())
        settings__["fd_network_layer_sizes"] = settings__["decoder_network_layer_sizes"]
        settings__["reward_network_layer_sizes"] = settings__["decoder_network_layer_sizes2"]
        settings__["fd_num_terrain_features"] = settings__["encoding_vector_size"] - 3
        settings__["dense_state_size"] = settings__["encoding_vector_size"]
        print ("****** Creating dense pose encoding network")
        print ("settings__[state_bounds]: ", len(settings__["state_bounds"][0]))
        if ("remove_character_state_features" in settings__):
            # settings__["state_bounds"][0] = settings__["state_bounds"][0][:-settings__["remove_character_state_features"]]
            # settings__["state_bounds"][1] = settings__["state_bounds"][1][:-settings__["remove_character_state_features"]]
            settings__["state_bounds"][0] = settings__["state_bounds"][0][:settings__["encoding_vector_size"]]
            print ("settings__[\"state_bounds\"][0]: ", len(settings__["state_bounds"][0]) )
            settings__["state_bounds"][1] = settings__["state_bounds"][1][:settings__["encoding_vector_size"]]
        self._modelTarget = createForwardDynamicsNetwork(settings__["state_bounds"], 
                                                         settings__["action_bounds"], settings__,
                                                         stateName="State_", resultStateName="ResultState_")
        # self._decode_state = keras.layers.Input(shape=(None,67), name="State_2")
        self._modelTarget._forward_dynamics_net = Model(inputs=[self._modelTarget.getStateSymbolicVariable()], outputs=self._modelTarget._forward_dynamics_net)
        if (print_info):
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print("FD Decoder Net summary: ", self._modelTarget._forward_dynamics_net.summary())
        self._modelTarget._reward_net = Model(inputs=self._modelTarget._State_, outputs=self._modelTarget._reward_net)
        if (print_info):
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print("Reward Decoder Net summary: ", self._modelTarget._reward_net.summary())
        
        VAE.compile(self)
    
    def compile(self):
        # sgd = SGD(lr=0.001, momentum=0.9)
        
        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        state_copy = keras.layers.Input(shape=keras.backend.int_shape(self._model.getStateSymbolicVariable())[1:], name="State_2")
        result_state_copy = keras.layers.Input(shape=keras.backend.int_shape(self._model.getResultStateSymbolicVariable())[1:]
                                                                              , name="ResultState_2"
                                                                              )
        print ("*** self._model.getStateSymbolicVariable() shape: ", repr(keras.backend.int_shape(self._model.getStateSymbolicVariable())))
        print ("*** self._model.getStateSymbolicVariable() shape: ", repr(self._model.getStateSymbolicVariable()))
        print ("*** self._model.getResultStateSymbolicVariable() shape: ", repr(keras.backend.int_shape(self._model.getResultStateSymbolicVariable())))
        print ("*** self._model.getResultStateSymbolicVariable() shape: ", repr(self._model.getResultStateSymbolicVariable()))
        processed_a = self._model._forward_dynamics_net(self._model.getStateSymbolicVariable())[0]
        self._model.processed_a = Model(inputs=[self._model.getStateSymbolicVariable()], outputs=processed_a)
        processed_b = self._model._forward_dynamics_net(state_copy)[0]
        self._model.processed_b = Model(inputs=[state_copy], outputs=processed_b)
        
        processed_a_log_var = self._model._forward_dynamics_net(self._model.getStateSymbolicVariable())[1]
        self._model.processed_a_log_var = Model(inputs=[self._model.getStateSymbolicVariable()], outputs=processed_a_log_var)
        processed_a_vae = self._model._forward_dynamics_net(self._model.getStateSymbolicVariable())[2]
        self._model.processed_a_vae = Model(inputs=[self._model.getStateSymbolicVariable()], outputs=processed_a_vae)
        
        if ("train_LSTM_Reward" in self.getSettings()
            and (self.getSettings()["train_LSTM_Reward"] == True)):
            network_ = keras.layers.TimeDistributed(self._model.processed_a, input_shape=(None, 1, self._state_length))(self._model.getResultStateSymbolicVariable())
            print ("network_: ", repr(network_))
            self._network_vae_log_var = keras.layers.TimeDistributed(self._model.processed_a_log_var, input_shape=(None, 1, self._state_length))(self._model.getResultStateSymbolicVariable())
            print ("_network_vae_log_var: ", repr(self._network_vae_log_var))
            self._network_vae = keras.layers.TimeDistributed(self._model.processed_a_vae, input_shape=(None, 1, self._state_length))(self._model.getResultStateSymbolicVariable())
            print ("network_vae: ", repr(self._network_vae))
        else:
            network_ = processed_a
            print ("network_mean: ", repr(network_))
            self._network_vae_log_var = processed_a_log_var
            print ("_network_vae_log_var: ", repr(self._network_vae_log_var))
            self._network_vae = processed_a_vae
            print ("_network_vae: ", repr(self._network_vae))
        
        
        """
        if ("condition_on_rnn_internal_state" in self.getSettings()
            and (self.getSettings()["condition_on_rnn_internal_state"] == True)):
            _, processed_a_r, processed_a_r_c  = self._model._reward_net(network_)
            _, processed_b_r, processed_b_r_c = self._model._reward_net(network_b)
            processed_a_r = keras.layers.concatenate(inputs=[processed_a_r, processed_a_r_c], axis=1)
            processed_b_r = keras.layers.concatenate(inputs=[processed_b_r, processed_b_r_c], axis=1)
            
            encode_input__ = keras.layers.Input(shape=keras.backend.int_shape(processed_b_r)[1:]
                                                                          , name="encoding_2"
                                                                          )
            last_dense = keras.layers.Dense(self.getSettings()["encoding_vector_size"], activation = 'linear')(encode_input__)
            self._last_dense = Model(inputs=[encode_input__], outputs=last_dense)
            
            processed_a_r = self._last_dense(processed_a_r)
            processed_b_r = self._last_dense(processed_b_r)
            
        else:
            
            processed_a_r_seq, processed_a_r = self._model._reward_net(network_)
            
            encode_input__ = keras.layers.Input(shape=keras.backend.int_shape(processed_a_r)[1:]
                                                                          , name="encoding_2"
                                                                          )
            last_dense = keras.layers.Dense(self.getSettings()["encoding_vector_size"], activation = 'linear')(encode_input__)
            self._last_dense = Model(inputs=[encode_input__], outputs=last_dense)
            processed_a_r = self._last_dense(processed_a_r)
        self._model.processed_a_r = Model(inputs=[self._model.getResultStateSymbolicVariable()], outputs=processed_a_r)
        """
        
        # distance_fd = keras.layers.Lambda(self._distance_func, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
        # distance_fd2 = keras.layers.Lambda(l1_distance_fd2, output_shape=eucl_dist_output_shape_fd2)([network_, network_b])
        # print ("distance_fd2: ", repr(distance_fd2))
        # distance_r = keras.layers.Lambda(self._distance_func, output_shape=eucl_dist_output_shape)([processed_a_r, processed_b_r])
        
        ### Decoding models
        ### https://github.com/keras-team/keras/issues/7949
        """
        def repeat_vector(args):
            # import keras
            ### sequence_layer is used to determine how long the repitition should be
            layer_to_repeat = args[0]
            sequence_layer = args[1]
            return RepeatVector(K.shape(sequence_layer)[1])(layer_to_repeat)
        ### Get a sequence as long as the state input
        encoder_a_outputs = keras.layers.Lambda(repeat_vector, output_shape=(None, self.getSettings()["encoding_vector_size"])) ([processed_a_r, self._model.getResultStateSymbolicVariable()])
        encoder_b_outputs = keras.layers.Lambda(repeat_vector, output_shape=(None, self.getSettings()["encoding_vector_size"])) ([processed_b_r, result_state_copy])
        print ("Encoder a output shape: ", encoder_a_outputs)
        print ("Encoder b output shape: ", encoder_b_outputs)
        
        ### Decode the sequence into another sequence
        decode_a_r = self._modelTarget._reward_net(encoder_a_outputs)
        print ("decode_a_r: ", repr(decode_a_r))
        # self._model.decode_a_r = Model(inputs=[encoder_a_outputs], outputs=decode_a_r)
        decode_b_r = self._modelTarget._reward_net(encoder_b_outputs)
        print ("decode_b_r: ", repr(decode_b_r))
        # self._model.decode_b_r = Model(inputs=[encoder_b_outputs], outputs=decode_b_r)
        
        """
        ### Decode sequences into images
        # state_copy = keras.layers.Input(shape=keras.backend.int_shape(self._model.getStateSymbolicVariable())[1:], name="State_2")
        if ("train_LSTM_Reward" in self.getSettings()
            and (self.getSettings()["train_LSTM_Reward"] == True)):
            decode_a = keras.layers.TimeDistributed(self._modelTarget._forward_dynamics_net, input_shape=(None, 1, 67))(processed_a_vae)
            print ("decode_a: ", repr(decode_a))
        else:
            decode_a = self._modelTarget._forward_dynamics_net(processed_a_vae)
            print ("decode_a: ", repr(decode_a))
        # decode_b = keras.layers.TimeDistributed(self._modelTarget._forward_dynamics_net, input_shape=(None, 1, 67))(decode_b_r)
        # print ("decode_b: ", repr(decode_b))
        # decode_a_vae = keras.layers.TimeDistributed(self._modelTarget._forward_dynamics_net, input_shape=(None, 1, 67))(self._network_vae)
        # print ("decode_a_vae: ", repr(decode_a_vae))

        """
        self._model._forward_dynamics_net = Model(inputs=[self._model.getStateSymbolicVariable()
                                                          ]
                                                  , outputs=processed_a_vae
                                                  )
        """
        self._modelTarget._forward_dynamics_net = Model(inputs=[self._model.getStateSymbolicVariable()], 
                                                        outputs=decode_a)
        """
        self._model._reward_net = Model(inputs=[self._model.getResultStateSymbolicVariable()
                                                          ]
                                                          , outputs=[
                                                                     decode_a_vae
                                                                     ]
                                                          )
        """
        # print ("encode_input__: ", repr(encode_input__))
        # distance_r_weighted = keras.layers.Dense(64, activation = 'sigmoid')(encode_input__)
        # self._distance_weighting_ = Model(inputs=[encode_input__], outputs=distance_r_weighted)
        # distance_r_weighted = self._distance_weighting_(distance_r)
        # print ("distance_r_weighted: ", repr(distance_r_weighted))
        
        if ( "return_rnn_sequence" in self.getSettings()
             and (self.getSettings()["return_rnn_sequence"])):
            distance_r_seq = keras.layers.Lambda(self._distance_func, output_shape=eucl_dist_output_shape_seq)([processed_a_r_seq, processed_b_r_seq])
            print ("distance_r_seq: ", repr(distance_r_seq))
            # distance_r_weighted_seq = keras.layers.TimeDistributed(self._distance_weighting_)(distance_r_seq)
            # print ("distance_r_weighted_seq: ", repr(distance_r_weighted_seq))
            self._model._reward_net_seq = Model(inputs=[self._model.getResultStateSymbolicVariable()
                                                              ,result_state_copy
                                                              ]
                                                              , outputs=distance_r_seq
                                                              )

        # sgd = SGD(lr=0.0005, momentum=0.9)
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), beta_1=np.float32(0.95), 
                                    beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0),
                                    clipnorm=2.5)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("sgd, actor: ", sgd)
            print ("Clipping: ", sgd.decay)
        # self._model._forward_dynamics_net.compile(loss=self.vae_loss_a, optimizer=sgd)
        self._modelTarget._forward_dynamics_net.compile(loss=self.vae_loss_a, optimizer=sgd)

        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), beta_1=np.float32(0.95), 
                                    beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0),
                                    clipnorm=2.5)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("sgd, actor: ", sgd)
            print ("Clipping: ", sgd.decay)
            
        # self._model._reward_net.compile(loss=self.vae_loss_a, optimizer=sgd)
        
    def vae_loss_a(self, action_true, action_pred):
        
        reconstruction_loss = mse(action_true, action_pred)
        # reconstruction_loss *= 4096
        kl_loss = 1 + self._network_vae_log_var - K.square(self._network_vae) - K.exp(self._network_vae_log_var)
        ### Using mean 
        kl_loss = K.mean(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss_a = K.mean(reconstruction_loss + kl_loss)
        return vae_loss_a

    def vae_loss_b(self, action_true, action_pred):
        
        reconstruction_loss = mse(action_true, action_pred)
        # reconstruction_loss *= 4096
        kl_loss = 1 + self._network_b_vae_log_var - K.square(self._network_b_vae) - K.exp(self._network_b_vae_log_var)
        ### Using mean 
        kl_loss = K.mean(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss_a = K.mean(reconstruction_loss + kl_loss)
        return vae_loss_a

    def reset(self):
        """
            Reset any state for the agent model
        """
        self._model.reset()
        self._model._reward_net.reset_states()
        self._model._forward_dynamics_net.reset_states()
        # self._model.processed_a.reset_states()
        # self._model.processed_b.reset_states()
        # self._model.processed_a_r.reset_states()
        # self._model.processed_b_r.reset_states()
        if not (self._modelTarget is None):
            self._modelTarget._forward_dynamics_net.reset_states()
            self._modelTarget._reward_net.reset_states()
            # self._modelTarget.reset()
            
    def getNetworkParameters(self):
        params = []
        params.append(copy.deepcopy(self._model._forward_dynamics_net.get_weights()))
        params.append(copy.deepcopy(self._model._reward_net.get_weights()))
        
        if ( "return_rnn_sequence" in self.getSettings()
             and (self.getSettings()["return_rnn_sequence"])):
            params.append(copy.deepcopy(self._model._reward_net_seq.get_weights()))
                
        return params
    
    def setNetworkParameters(self, params):
        self._model._forward_dynamics_net.set_weights(params[0])
        self._model._reward_net.set_weights(params[1])
        if ( "return_rnn_sequence" in self.getSettings()
             and (self.getSettings()["return_rnn_sequence"])):
            self._model._reward_net_seq.set_weights(params[2])
        
    def setGradTarget(self, grad):
        self._fd_grad_target_shared.set_value(grad)
        
    def getGrads(self, states, actions, result_states, v_grad=None, alreadyNormed=False):
        if ( alreadyNormed == False ):
            states = np.array(norm_state(states, self.getStateBounds()), dtype=self.getSettings()['float_type'])
            actions = np.array(norm_action(actions, self._action_bounds), dtype=self.getSettings()['float_type'])
            result_states = np.array(norm_state(result_states, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        # result_states = np.array(result_states, dtype=self.getSettings()['float_type'])
        # self.setData(states, actions, result_states)
        # if (v_grad != None):
        # print ("states shape: ", states.shape, " actions shape: ", actions.shape, " v_grad.shape: ", v_grad.shape)
        self.setGradTarget(v_grad)
        # print ("states shape: ", states.shape, " actions shape: ", actions.shape)
        # grad = self._get_grad([states, actions])[0]
        grad = np.zeros_like(states)
        # print ("grad: ", grad)
        return grad
    
    def getRewardGrads(self, states, actions, alreadyNormed=False):
        # states = np.array(states, dtype=self.getSettings()['float_type'])
        # actions = np.array(actions, dtype=self.getSettings()['float_type'])
        if ( alreadyNormed is False ):
            states = np.array(norm_state(states, self.getStateBounds()), dtype=self.getSettings()['float_type'])
            actions = np.array(norm_action(actions, self._action_bounds), dtype=self.getSettings()['float_type'])
            # rewards = np.array(norm_state(rewards, self._reward_bounds), dtype=self.getSettings()['float_type'])
        # self.setData(states, actions)
        return self._get_grad_reward([states, actions, 0])[0]
    
    def updateTargetModel(self):
        pass
                
    def train(self, states, actions, result_states, rewards, falls=None, updates=1, batch_size=None, p=1, lstm=True):
        """
            states will come for the agent and
            results_states can come from the imitation agent
        """
        # print ("fd: ", self)
        # print ("state length: ", len(self.getStateBounds()[0]))
        self.reset()
        states_ = states
        if ('anneal_learning_rate' in self.getSettings()
            and (self.getSettings()['anneal_learning_rate'] == True)):
            K.set_value(self._model._forward_dynamics_net.optimizer.lr, np.float32(self.getSettings()['fd_learning_rate']) * p)
        if ("replace_next_state_with_imitation_viz_state" in self.getSettings()
            and (self.getSettings()["replace_next_state_with_imitation_viz_state"] == True)):
            states_ = np.concatenate((states_, result_states), axis=0)
        if (((("train_LSTM_FD" in self._settings)
                and (self._settings["train_LSTM_FD"] == True))
            or
            (("train_LSTM_Reward" in self._settings)
                and (self._settings["train_LSTM_Reward"] == True))
            ) 
            and lstm):
            ### result states can be from the imitation agent.
            # print ("falls: ", falls)
            if (falls is None):
                sequences0, sequences1, targets_ = create_sequences2(states, result_states, self._settings)
                if ("include_agent_imitator_pairs" in self._settings
                    and (self._settings["include_agent_imitator_pairs"] == True)):
                    """
                    sequences0_, sequences1_, targets___ = create_advisarial_sequences(states, result_states, self._settings)
                    sequences0.extend(sequences0_)
                    sequences1.extend(sequences1_)
                    targets_.extend(targets___)
                    sequences0_, sequences1_, targets___ = create_advisarial_sequences(states, result_states, self._settings)
                    sequences0.extend(sequences0_)
                    sequences1.extend(sequences1_)
                    targets_.extend(targets___)
                    """
                    sequences0, sequences1, targets_ = create_advisarial_sequences(states, result_states, self._settings)
                    # sequences0.extend(sequences0_)
                    # sequences1.extend(sequences1_)
                    # targets_.extend(targets___)
            else:
                sequences0, sequences1, targets_ = create_multitask_sequences(states, result_states, falls, self._settings)
            sequences0 = np.array(sequences0)
            # print ("sequences0 shape: ", sequences0.shape)
            sequences1 = np.array(sequences1)
            targets_ = np.array(targets_)
            
            if ( "add_label_noise" in self._settings):
                if (np.random.rand() < self._settings["add_label_noise"]):
                    # print ("targets_[0]: ", targets_[0])
                    targets_ = 1.0 - targets_ ### Invert labels
                    # print ("Inverting label values this time")
                    # print ("targets_[0]: ", targets_[0])
            # print ("targets_ shape: ", targets_.shape)
            # te_pair1, te_pair2, te_y = seq
            # score = self._model._forward_dynamics_net.train_on_batch([sequences0, sequences1], targets_)
            loss_ = []
            if ("train_LSTM_FD_stateful" in self._settings
                and (self._settings["train_LSTM_FD_stateful"] == True)
                # and False
                ):
                for k in range(sequences0.shape[1]):
                    ### shaping data
                    x0 = np.array(sequences0[:,[k]])
                    x1 = np.array(sequences1[:,[k]])
                    y0 = np.array(targets_[:,k]) ### For now reduce the dimensionality of the target because my nets output (batch_size, target)
                    # print ("data: ", np.mean(x0), np.mean(x1), np.mean(y0))
                    # print (x0) 
                    # print ("x0 shape: ", x0.shape)
                    # print ("y0 shape: ", y0.shape)
                    if (("train_LSTM_FD" in self._settings)
                        and (self._settings["train_LSTM_FD"] == True)):
                        score = self._model._forward_dynamics_net.fit([x0, x1], [y0],
                                  epochs=1, 
                                  batch_size=sequences0.shape[0],
                                  verbose=0
                                  )
                        # print ("lstm train loss: ", score.history['loss'])
                        loss_.append(np.mean(score.history['loss']))
                    if (("train_LSTM_Reward" in self._settings)
                        and (self._settings["train_LSTM_Reward"] == True)):  
                        score = self._model._reward_net.fit([x0, x1], [y0],
                                  epochs=1, 
                                  batch_size=sequences0.shape[0],
                                  verbose=0
                                  )
                        # print ("lstm train loss: ", score.history['loss'])
                        loss_.append(np.mean(score.history['loss']))
            else:
                # print ("targets_[:,:,0]: ", np.mean(targets_, axis=1))
                targets__ = np.mean(targets_, axis=1)
                # print ("targets__: ", targets__)
                if (("train_LSTM_FD" in self._settings)
                    and (self._settings["train_LSTM_FD"] == True)):
                    score = self._model._forward_dynamics_net.fit([sequences0, sequences1], [targets__],
                                  epochs=1, 
                                  batch_size=sequences0.shape[0],
                                  verbose=0
                                  )
                    loss_.append(np.mean(score.history['loss']))
                    
                if (("train_LSTM_Reward" in self._settings)
                    and (self._settings["train_LSTM_Reward"] == True)):
                    
                    if (("train_lstm_fd_and_reward_and_decoder_together" in self._settings)
                        and (self._settings["train_lstm_fd_and_reward_and_decoder_together"] == True)):
                        
                        sequences0_ = sequences0
                        sequences1_ = sequences1
                        if ("remove_character_state_features" in self._settings):
                            sequences0_ = sequences0_[:, :, :-self._settings["remove_character_state_features"]]
                            sequences1_ = sequences1_[:, :, :-self._settings["remove_character_state_features"]]
                        # print ("sequences0 shape: ", sequences0.shape)
                        # print ("sequences1 shape: ", sequences1.shape)
                        # print ("targets__ shape: ", targets__.shape)
                        # print ("targets_ shape: ", targets_.shape)
                        ### separate data into positive and negative batches
                        # for k in range(len(sequences0)):
                        indecies_ = list(range(len(targets__)))
                        # print ("targets__: ", targets__)
                        # print("indecies_: ", indecies_)
                        if ("seperate_posandneg_pairs" in self._settings
                            and (self._settings["seperate_posandneg_pairs"] == True)):
                            less_ = np.less(targets__, 0.5)
                            negative_indecies = np.where(less_ == True)[0]
                            positive_indecies = np.where(less_ == False)[0]
                            # print ("negative_indecies: ", negative_indecies)
                            indecies_ = negative_indecies
                            # if (np.random.rand() > 0.5):
                            #     indecies_ = positive_indecies 
                                
                        
                        score = self._model._reward_net.fit([sequences0[indecies_], sequences1[indecies_]], 
                                      [
                                       sequences0_[indecies_], sequences1_[indecies_]],
                                      epochs=1, 
                                      batch_size=sequences0.shape[0],
                                      verbose=0
                                      )
                        
                    else:
                        score = self._model._reward_net.fit([sequences0, sequences1], [targets__],
                                      epochs=1, 
                                      batch_size=sequences0.shape[0],
                                      verbose=0
                                      )
                    loss_.append(np.mean(score.history['loss']))
            
            return np.mean(loss_)
        else:
            te_pair1, te_pair2, te_y = create_pairs2(states_, self._settings)
        self._updates += 1
        if (batch_size is None):
            batch_size_=states.shape[0]
        else:
            batch_size_=batch_size
        loss = 0
        # dist_ = np.array(self._contrastive_loss([te_pair1, te_pair2, 0]))[0]
        # dist = np.mean(dist_)
        te_y = np.array(te_y)
        # print("Distance: ", dist)
        # print("targets: ", te_y)
        # print("pairs: ", te_pair1)
        # print("Distance.shape, targets.shape: ", dist_.shape, te_y.shape)
        # print("Distance, targets: ", np.concatenate((dist_, te_y), axis=1))
        # if ( dist > 0):
        score = self._model._forward_dynamics_net.fit([te_pair1, te_pair2], te_y,
          epochs=updates, batch_size=batch_size_,
          verbose=0,
          shuffle=True
          )
        loss = np.mean(score.history['loss'])
            # print ("loss: ", loss)
        return loss
    
    def predict_encoding(self, state):
        """
            Compute distance between two states
        """
        # state = np.array(norm_state(state, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        if (("train_LSTM_FD" in self._settings)
                    and (self._settings["train_LSTM_FD"] == True)):
            h_a = self._model.processed_a.predict([np.array([state])])
        else:
            h_a = self._model._forward_dynamics_net.predict([state])[0]
        return h_a
    
    def predict(self, state, state2):
        """
            Compute distance between two states
        """
        # print("state shape: ", np.array(state).shape)
        state = np.array(norm_state(state, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        state2 = np.array(norm_state(state2, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        if ((("train_LSTM_FD" in self._settings)
                    and (self._settings["train_LSTM_FD"] == True))
                    # or
                    # settings["use_learned_reward_function"] == "dual"
                    ):
            ### Used because we need to keep two separate RNN networks and not mix the hidden states
            h_a = self._model.processed_a.predict([np.array([state])])
            h_b = self._model.processed_b.predict([np.array([state2])])
            state_ = self._distance_func_np((h_a, h_b))[0]
            # print ("siamese dist: ", state_)
            # state_ = self._model._forward_dynamics_net.predict([np.array([state]), np.array([state2])])[0]
        else:
            # state_ = self._model._forward_dynamics_net.predict([state, state2])[0]
            ### Want the mean of the forward model
            h_a = self._model._forward_dynamics_net.predict([np.array(state)])[0]
            h_b = self._model._forward_dynamics_net.predict([np.array(state2)])[0]
            state_ = self._distance_func_np((h_a, h_b))[0]
        # dist_ = np.array(self._contrastive_loss([te_pair1, te_pair2, 0]))[0]
        # print("state_ shape: ", np.array(state_).shape)
        return state_
    
    def predictWithDropout(self, state, action):
        # "dropout"
        state = np.array(norm_state(state, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        self._model.setStates(state)
        self._model.setActions(action)
        state_ = scale_state(self._forwardDynamics_drop()[0], self.getStateBounds())
        return state_
    
    def predict_std(self, state, action, p=1.0):
        state = np.array(norm_state(state, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        state_ = self._forwardDynamics_std() * (action_bound_std(self.getStateBounds()))
        return state_
    
    def predict_reward(self, state, state2):
        """
            Predict reward which is inverse of distance metric
        """
        # print ("state bounds length: ", self.getStateBounds())
        # print ("fd: ", self)
        state = np.array(norm_state(state, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        state2 = np.array(norm_state(state2, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        if (("train_LSTM_Reward" in self._settings)
            and (self._settings["train_LSTM_Reward"] == True)):
            ### Used because we need to keep two separate RNN networks and not mix the hidden states
            # print ("State shape: ", np.array([np.array([state])]).shape)
            # h_a = self._model.processed_a_r.predict([np.array([state])])
            # h_b = self._model.processed_b_r.predict([np.array([state2])])
            reward_ = [0]
            # print ("siamese dist: ", state_)
            # state_ = self._model._forward_dynamics_net.predict([np.array([state]), np.array([state2])])[0]
        else:
            predicted_reward = self._model._reward_net.predict([state, state2])[0]
            # reward_ = scale_reward(predicted_reward, self.getRewardBounds()) # * (1.0 / (1.0- self.getSettings()['discount_factor']))
            reward_ = predicted_reward
            
        return reward_
    
    def predict_reward_encoding(self, state):
        """
            Predict reward which is inverse of distance metric
        """
        # state = np.array(norm_state(state, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        if (("train_LSTM_Reward" in self._settings)
            and (self._settings["train_LSTM_Reward"] == True)):
            h_a = self._model.processed_a_r.predict([np.array([state])])
        else:
            h_a = self._model._reward_net.predict([state])[0]
            # reward_ = scale_reward(predicted_reward, self.getRewardBounds()) # * (1.0 / (1.0- self.getSettings()['discount_factor']))
            
        return h_a
    
    def predict_batch(self, states, actions):
        ## These input should already be normalized.
        return self.fd([states, actions, 0])[0]
    
    def predict_reward_batch(self, states, actions):
        """
            This data should already be normalized
        """
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        predicted_reward = self.reward([states, actions, 0])[0]
        return predicted_reward
    
    def predict_reward_(self, states, states2):
        """
            This data should already be normalized
        """
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        states = np.array(norm_state(states, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        actions = np.array(norm_state(states2, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        h_a = self._model.processed_a_r_seq.predict([states])
        h_b = self._model.processed_b_r_seq.predict([states2])
        # print ("h_b shape: ", h_b.shape) 
        predicted_reward = np.array([self._distance_func_np((np.array([h_a_]), np.array([h_b_])))[0] for h_a_, h_b_ in zip(h_a[0], h_b[0])])
        # print ("predicted_reward_: ", predicted_reward)
        # predicted_reward = self._model._reward_net_seq.predict([states, actions], batch_size=1)[0]
        return predicted_reward

    def bellman_error(self, states, actions, result_states, rewards):
        self.reset()
        if (("train_LSTM_FD" in self._settings)
                    and (self._settings["train_LSTM_FD"] == True)):
            sequences0, sequences1, targets_ = create_sequences2(states, result_states, self._settings)
            sequences0 = np.array(sequences0)
            sequences1 = np.array(sequences1)
            targets_ = np.array(targets_)
            errors=[]
            if ("train_LSTM_FD_stateful" in self._settings
                and (self._settings["train_LSTM_FD_stateful"] == True)
                # and False
                ):
                for k in range(sequences0.shape[1]):
                    ### shaping data
                    # print (k)
                    x0 = np.array(sequences0[:,[k]])
                    x1 = np.array(sequences1[:,[k]])
                    y0 = np.array(targets_[:,k]) ### For now reduce the dimensionality of the target because my nets output (batch_size, target)
                    predicted_y = self._model._forward_dynamics_net.predict([x0, x1], batch_size=x0.shape[0])
                    errors.append( compute_accuracy(predicted_y, y0) )
            else:
                predicted_y = self._model._forward_dynamics_net.predict([sequences0, sequences1], batch_size=sequences0.shape[0])
                # print ("fd error, predicted_y: ", predicted_y)
                targets__ = np.mean(targets_, axis=1)
                # print ("fd error, targets_ : ", targets_)
                # print ("fd error, targets__: ", targets__)
                errors.append( compute_accuracy(predicted_y, targets__) )
            # predicted_y = self._model._forward_dynamics_net.predict([np.array([[sequences0[0]]]), np.array([[sequences1[0]]])])
            # te_acc = compute_accuracy(predicted_y, np.array([targets_[0]]) )
            te_acc = np.mean(errors)
        else:
            states = np.concatenate((states, result_states), axis=0)
            te_pair1, te_pair2, te_y = create_pairs2(states, self._settings)
        
            # state_ = self._model._forward_dynamics_net.predict([state, state2])[0]
            predicted_y = self._model._forward_dynamics_net.predict([te_pair1, te_pair2])
            te_acc = compute_accuracy(predicted_y, te_y)
            
        # predicted_y = self._model._forward_dynamics_net.predict([te_pair1, te_pair2])
        return te_acc
    
    def reward_error(self, states, actions, result_states, rewards):
        # rewards = rewards * (1.0/(1.0-self.getSettings()['discount_factor'])) # scale rewards
        self.reset()
        if (("train_LSTM_Reward" in self._settings)
                    and (self._settings["train_LSTM_Reward"] == True)):
            sequences0, sequences1, targets_ = create_sequences2(states, result_states, self._settings)
            sequences0 = np.array(sequences0)
            sequences1 = np.array(sequences1)
            targets_ = np.array(targets_)
            errors=[]
            if ("train_LSTM_FD_stateful" in self._settings
                and (self._settings["train_LSTM_FD_stateful"] == True)
                # and False
                ):
                for k in range(sequences0.shape[1]):
                    ### shaping data
                    # print (k)
                    x0 = np.array(sequences0[:,[k]])
                    x1 = np.array(sequences1[:,[k]])
                    y0 = np.array(targets_[:,k]) ### For now reduce the dimensionality of the target because my nets output (batch_size, target)
                    predicted_y = self._model._reward_net.predict([x0, x1], batch_size=x0.shape[0])
                    errors.append( compute_accuracy(predicted_y, y0) )
            else:
                predicted_y = self._model._reward_net.predict([sequences0, sequences1], batch_size=sequences0.shape[0])
                if (("train_lstm_fd_and_reward_and_decoder_together" in self._settings)
                    and (self._settings["train_lstm_fd_and_reward_and_decoder_together"] == True)):
                    predicted_y = predicted_y[0]
                # print ("fd error, predicted_y: ", predicted_y)
                targets__ = np.mean(targets_, axis=1)
                # print ("fd error, targets_ : ", targets_)
                # print ("fd error, targets__: ", targets__)
                errors.append( compute_accuracy(predicted_y, targets__) )
            # predicted_y = self._model._forward_dynamics_net.predict([np.array([[sequences0[0]]]), np.array([[sequences1[0]]])])
            # te_acc = compute_accuracy(predicted_y, np.array([targets_[0]]) )
            te_acc = np.mean(errors)
        else:
            states = np.concatenate((states, result_states), axis=0)
            te_pair1, te_pair2, te_y = create_pairs2(states, self._settings)
        
            # state_ = self._model._forward_dynamics_net.predict([state, state2])[0]
            predicted_y = self._model._reward_net.predict([te_pair1, te_pair2])
            te_acc = compute_accuracy(predicted_y, te_y)
            
        # predicted_y = self._model._forward_dynamics_net.predict([te_pair1, te_pair2])
        return te_acc

    def saveTo(self, fileName):
        # print(self, "saving model")
        import h5py
        hf = h5py.File(fileName+"_bounds.h5", "w")
        hf.create_dataset('_state_bounds', data=self.getStateBounds())
        hf.create_dataset('_reward_bounds', data=self.getRewardBounds())
        hf.create_dataset('_action_bounds', data=self.getActionBounds())
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("fd save self.getStateBounds(): ", len(self.getStateBounds()[0]))
        # hf.create_dataset('_resultgetStateBounds()', data=self.getResultStateBounds())
        # print ("fd: ", self)
        hf.flush()
        hf.close()
        suffix = ".h5"
        ### Save models
        # self._model._actor_train.save(fileName+"_actor_train"+suffix, overwrite=True)
        self._model._forward_dynamics_net.save(fileName+"_FD"+suffix, overwrite=True)
        # self._model._reward_net.save(fileName+"_reward"+suffix, overwrite=True)
        self._model._reward_net.save_weights(fileName+"_reward"+suffix, overwrite=True)
        self._modelTarget._forward_dynamics_net.save(fileName+"_FD_T"+suffix, overwrite=True)
        # self._model._reward_net.save(fileName+"_reward"+suffix, overwrite=True)
        self._modelTarget._reward_net.save_weights(fileName+"_reward_T"+suffix, overwrite=True)
        # print ("self._model._actor_train: ", self._model._actor_train)
        try:
            from keras.utils import plot_model
            ### Save model design as image
            plot_model(self._model._forward_dynamics_net, to_file=fileName+"_FD"+'.svg', show_shapes=True)
            plot_model(self._model._reward_net, to_file=fileName+"_reward"+'.svg', show_shapes=True)
        except Exception as inst:
            ### Maybe the needed libraries are not available
            print ("Error saving diagrams for rl models.")
            print (inst)
        
    def loadFrom(self, fileName):
        import h5py
        from util.utils import load_keras_model
        # from keras.models import load_weights
        suffix = ".h5"
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Loading agent: ", fileName)
        # with K.get_session().graph.as_default() as g:
        ### Need to lead the model this way because the learning model's State expects batches...
        forward_dynamics_net = load_keras_model(fileName+"_FD"+suffix, custom_objects={'contrastive_loss': contrastive_loss})
        #reward_net = load_keras_model(fileName+"_reward"+suffix, custom_objects={'contrastive_loss': contrastive_loss,
        #                                                                         "vae_loss_a": self.vae_loss_a,
        #                                                                         "vae_loss_b": self.vae_loss_b})
        # if ("simulation_model" in self.getSettings() and
        #     (self.getSettings()["simulation_model"] == True)):
        if (True): ### Because the simulation and learning use different model types (statefull vs stateless lstms...)
            self._model._forward_dynamics_net.set_weights(forward_dynamics_net.get_weights())
            self._model._forward_dynamics_net.optimizer = forward_dynamics_net.optimizer
            # self._model._reward_net.set_weights(reward_net.get_weights())
            self._model._reward_net.load_weights(fileName+"_reward"+suffix)
            # self._model._reward_net.optimizer = reward_net.optimizer
        else:
            self._model._forward_dynamics_net = forward_dynamics_net
            self._model._reward_net = reward_net
            
        self._forward_dynamics_net = self._model._forward_dynamics_net
        self._reward_net = self._model._reward_net
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("******** self._forward_dynamics_net: ", self._forward_dynamics_net)
        if (self._modelTarget is not None):
            self._modelTarget._forward_dynamics_net = load_keras_model(fileName+"_FD_T"+suffix, custom_objects={'contrastive_loss': contrastive_loss})
            # self._modelTarget._reward_net = load_keras_model(fileName+"_reward_net_T"+suffix)
            self._modelTarget._reward_net.load_weights(fileName+"_reward_T"+suffix)
        # self._model._actor_train = load_keras_model(fileName+"_actor_train"+suffix, custom_objects={'loss': pos_y})
        # self._value = K.function([self._model.getStateSymbolicVariable(), K.learning_phase()], [self.__value])
        # self._value_Target = K.function([self._model.getResultStateSymbolicVariable(), K.learning_phase()], [self.__value_Target])
        hf = h5py.File(fileName+"_bounds.h5",'r')
        self.setStateBounds(np.array(hf.get('_state_bounds')))
        self.setRewardBounds(np.array(hf.get('_reward_bounds')))
        self.setActionBounds(np.array(hf.get('_action_bounds')))
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("fd load self.getStateBounds(): ", len(self.getStateBounds()[0]))
        # self._resultgetStateBounds() = np.array(hf.get('_resultgetStateBounds()'))
        hf.close()
        