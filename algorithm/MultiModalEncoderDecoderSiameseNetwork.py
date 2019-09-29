import numpy as np
import sys
from model.ModelUtil import *
from model.LearningUtil import loglikelihood, loglikelihoodMEAN, kl, entropy, flatgrad, zipsame, get_params_flat, setFromFlat, likelihood, loglikelihoodMEAN
from model.LearningUtil import loglikelihood, likelihood, likelihoodMEAN, kl, kl_D, entropy, flatgrad, zipsame, get_params_flat, setFromFlat
from keras.optimizers import SGD
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K
import keras
from keras.models import Sequential, Model
from keras.layers import RepeatVector

from util.SimulationUtil import createForwardDynamicsNetwork
from algorithm.SiameseNetwork import compute_accuracy, contrastive_loss_np
from algorithm.MultiModalSiameseNetwork import cosine_distance, cos_dist_output_shape, euclidean_distance, euclidean_distance_np, eucl_dist_output_shape, contrastive_loss, create_sequences, create_multitask_sequences, create_pairs2

# For debugging
# theano.config.mode='FAST_COMPILE'
from algorithm.KERASAlgorithm import KERASAlgorithm


class MultiModalEncoderDecoderSiameseNetwork(KERASAlgorithm):
    """
         This method uses two different types of data and learns a distance function between them.
         In this case the first type of data is pixles and the second is dense pose data.
         
         Notes:
         Maybe I can just let the first model ignore the pose features, i.e. not merge them...
    """
    def __init__(self, model, state_length, action_length, state_bounds, action_bounds, settings_, reward_bounds=0, print_info=False):

        super(MultiModalEncoderDecoderSiameseNetwork,self).__init__(model, state_length, action_length, state_bounds, action_bounds, reward_bounds, settings_)
        self._model = model
        self._learning_rate = self.getSettings()["fd_learning_rate"]
        self._regularization_weight = 1e-6
        
        condition_reward_on_result_state = False
        self._train_combined_loss = False
        
        ### Need to create a new model that uses a different network
        settings__ = copy.deepcopy(self.getSettings())
        settings__["fd_network_layer_sizes"] = settings__["encoder_network_layer_sizes"]
        settings__["reward_network_layer_sizes"] = settings__["decoder_network_layer_sizes"]
        settings__["fd_num_terrain_features"] = 0
        print ("****** Creating dense pose encoding network")
        if ("remove_character_state_features" in settings__):
            settings__["state_bounds"][0] = settings__["state_bounds"][0][:-settings__["remove_character_state_features"]]
            settings__["state_bounds"][1] = settings__["state_bounds"][1][:-settings__["remove_character_state_features"]]
        self._modelTarget = createForwardDynamicsNetwork(settings__["state_bounds"], 
                                                         settings__["action_bounds"], settings__,
                                                         stateName="State_", resultStateName="ResultState_")

        self._inputs_a = self._model.getStateSymbolicVariable()
        self._inputs_b = self._modelTarget.getStateSymbolicVariable() 
        self._model._forward_dynamics_net = Model(inputs=[self._inputs_a], outputs=self._model._forward_dynamics_net)
        self._modelTarget._forward_dynamics_net = Model(inputs=[self._inputs_b], outputs=self._modelTarget._forward_dynamics_net)
        if (print_info):
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print("FD Conv Net summary: ", self._model._forward_dynamics_net.summary())
                print("FD Target Net summary: ", self._modelTarget._forward_dynamics_net.summary())
        
        self._model._reward_net = Model(inputs=[self._model.getResultStateSymbolicVariable()], outputs=self._model._reward_net)
        self._modelTarget._reward_net = Model(inputs=[self._modelTarget.getResultStateSymbolicVariable()], outputs=self._modelTarget._reward_net)
        if (print_info):
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print("FD Reward Net summary: ", self._model._reward_net.summary())
                print("FD Target Reward Net summary: ", self._modelTarget._reward_net.summary())
                

        MultiModalEncoderDecoderSiameseNetwork.compile(self)
    
    def compile(self):
        # sgd = SGD(lr=0.001, momentum=0.9)
        
        print ("*** self._model.getStateSymbolicVariable() shape: ", repr(self._model.getStateSymbolicVariable()))
        print ("*** self._model.getResultStateSymbolicVariable() shape: ", repr(self._model.getResultStateSymbolicVariable()))
        print ("*** self._modelTarget.getStateSymbolicVariable() shape: ", repr(self._modelTarget.getStateSymbolicVariable()))
        print ("*** self._modelTarget.getResultStateSymbolicVariable() shape: ", repr(self._modelTarget.getResultStateSymbolicVariable()))
        # state_copy = keras.layers.Input(shape=keras.backend.int_shape(self._inputs_b)[1:], name="State_2")
        # result_state_copy = keras.layers.Input(shape=keras.backend.int_shape(self._inputs_bb)[1:]
        #                                                                       , name="ResultState_2"
        #                                                                       )
        
        encode_a = self._model._forward_dynamics_net(self._model.getStateSymbolicVariable())
        self._model.encode_a = Model(inputs=[self._model.getStateSymbolicVariable()], outputs=encode_a)
        encode_b = self._modelTarget._forward_dynamics_net(self._modelTarget.getStateSymbolicVariable())
        self._model.encode_b = Model(inputs=[self._modelTarget.getStateSymbolicVariable()], outputs=encode_b)
        
        # distance_fd = keras.layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
        # distance_r = keras.layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([encode_a, encode_b])
        """
        self._model._forward_dynamics_net = Model(inputs=[self._inputs_a
                                                          ,state_copy
                                                          ]
                                                  , outputs=distance_fd
                                                  )
        """
        """
        self._model._reward_net = Model(inputs=[self._inputs_aa
                                              ,result_state_copy
                                              ]
                                              , outputs=distance_r
                                              )
        """                                       
        ### https://github.com/keras-team/keras/issues/7949
        def repeat_vector(args):
            # import keras
            ### sequence_layer is used to determine how long the repitition should be
            layer_to_repeat = args[0]
            sequence_layer = args[1]
            return RepeatVector(K.shape(sequence_layer)[1])(layer_to_repeat)

        encoder_a_outputs = keras.layers.Lambda(repeat_vector, output_shape=(None, 32)) ([encode_a, self._model.getStateSymbolicVariable()])
        encoder_b_outputs = keras.layers.Lambda(repeat_vector, output_shape=(None, 32)) ([encode_b, self._modelTarget.getStateSymbolicVariable()])
        print ("Encoder a output shape: ", encoder_a_outputs)
        print ("Encoder b output shape: ", encoder_b_outputs)
        
        decode_a_r = self._model._reward_net(encoder_a_outputs)
        # self._model.decode_a_r = Model(inputs=[encoder_a_outputs], outputs=decode_a_r)
        decode_b_r = self._modelTarget._reward_net(encoder_b_outputs)
        # self._model.decode_b_r = Model(inputs=[encoder_b_outputs], outputs=decode_b_r)
        
        self._model._reward_net_a = Model(inputs=[self._model.getStateSymbolicVariable()
                                                          ]
                                                          , outputs=decode_a_r
                                                          )
        self._model._reward_net_b = Model(inputs=[self._modelTarget.getStateSymbolicVariable()
                                                          ]
                                                          , outputs=decode_b_r 
                                                          )
        
        distance_r = keras.layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([encode_a, encode_b])
        
        self._model._combination = Model(inputs=[self._model.getStateSymbolicVariable(),
                                                 self._modelTarget.getStateSymbolicVariable()
                                                          ]
                                                          , outputs=[decode_a_r,
                                                                     decode_b_r,
                                                                     distance_r] 
                                                          )
        
        # sgd = SGD(lr=0.0005, momentum=0.9)
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), beta_1=np.float32(0.95), 
                                    beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0),
                                    clipnorm=2.5)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("sgd, actor: ", sgd)
            print ("Clipping: ", sgd.decay)
        self._model._forward_dynamics_net.compile(loss=contrastive_loss, optimizer=sgd)

        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), beta_1=np.float32(0.95), 
                                    beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0),
                                    clipnorm=2.5)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("sgd, actor: ", sgd)
            print ("Clipping: ", sgd.decay)
        self._model._combination.compile(loss=['mse', 'mse', contrastive_loss], loss_weights=[0.01, 0.01, 0.98],optimizer=sgd)
        """
        self._contrastive_loss = K.function([self._inputs_a, 
                                             self._inputs_b,
                                             K.learning_phase()], 
                                            [distance_fd])
        
        self._contrastive_loss_r = K.function([self._inputs_aa, 
                                             self._inputs_bb,
                                             K.learning_phase()], 
                                            [distance_r])
        """
        # self.reward = K.function([self._model.getStateSymbolicVariable(), self._model.getActionSymbolicVariable(), K.learning_phase()], [self._reward])
        
    def getNetworkParameters(self):
        params = []
        params.append(copy.deepcopy(self._model._forward_dynamics_net.get_weights()))
        params.append(copy.deepcopy(self._model._reward_net.get_weights()))
        params.append(copy.deepcopy(self._modelTarget._forward_dynamics_net.get_weights()))
        params.append(copy.deepcopy(self._modelTarget._reward_net.get_weights()))
        return params
    
    def setNetworkParameters(self, params):
        self._model._forward_dynamics_net.set_weights(params[0])
        self._model._reward_net.set_weights(params[1])
        self._modelTarget._forward_dynamics_net.set_weights(params[2])
        self._modelTarget._reward_net.set_weights(params[3])
        
    def setGradTarget(self, grad):
        self._fd_grad_target_shared.set_value(grad)
        
    def getGrads(self, states, actions, result_states, v_grad=None, alreadyNormed=False):
        if ( alreadyNormed == False ):
            states = np.array(norm_state(states, self._state_bounds), dtype=self.getSettings()['float_type'])
            actions = np.array(norm_action(actions, self._action_bounds), dtype=self.getSettings()['float_type'])
            result_states = np.array(norm_state(result_states, self._state_bounds), dtype=self.getSettings()['float_type'])
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
            states = np.array(norm_state(states, self._state_bounds), dtype=self.getSettings()['float_type'])
            actions = np.array(norm_action(actions, self._action_bounds), dtype=self.getSettings()['float_type'])
            # rewards = np.array(norm_state(rewards, self._reward_bounds), dtype=self.getSettings()['float_type'])
        # self.setData(states, actions)
        return self._get_grad_reward([states, actions, 0])[0]
        
    def updateTargetModel(self):
        pass
          
    def train(self, states, actions, result_states, rewards, falls=None, updates=1, batch_size=None, p=1, lstm=True, datas=None):
        """
            states will come for the agent and
            results_states can come from the imitation agent
        """
        self.reset()
        if ('anneal_learning_rate' in self.getSettings()
            and (self.getSettings()['anneal_learning_rate'] == True)):
            K.set_value(self._model._forward_dynamics_net.optimizer.lr, np.float32(self.getSettings()['fd_learning_rate']) * p)
        if ("replace_next_state_with_imitation_viz_state" in self.getSettings()
            and (self.getSettings()["replace_next_state_with_imitation_viz_state"] == True)):
            states_ = np.concatenate((states, result_states), axis=0)
        if (((("train_LSTM_FD" in self._settings)
                and (self._settings["train_LSTM_FD"] == True))
            or
            (("train_LSTM_Reward" in self._settings)
                and (self._settings["train_LSTM_Reward"] == True))
            ) 
            and lstm):
            ### result states can be from the imitation agent.
            if (falls is None):
                sequences0, sequences1, targets_ = create_sequences(states, result_states, self._settings)
            else:
                sequences0, sequences1, targets_ = create_multitask_sequences(states, result_states, datas["task_id"], self._settings)
            """
            for jk in range(len(sequences0)):
                print ("sequences0 " , jk, ": len = ", len(sequences0[jk]))
            """
            # print ("sequences0 shape: ", np.array(sequences0).shape)
            sequences0 = np.array(sequences0)
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
                # print ("sequences0 shape: ", np.array(sequences0).shape)
                ### subtract dense state off
                vid_targets = sequences0[:,:, self._settings["dense_state_size"]:]
                if ("remove_character_state_features" in self._settings):
                    ### Remove ground reaction forces from state
                    sequences1 = sequences1[:, :, :-self._settings["remove_character_state_features"]]
                score = self._model._combination.fit([sequences0, sequences1], [vid_targets, sequences1, targets__],
                              epochs=1, 
                              batch_size=sequences0.shape[0],
                              verbose=0
                              )
                # print (score.history)
                loss_.append(score.history['loss'])
            
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
        # state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        if (("train_LSTM_FD" in self._settings)
                    and (self._settings["train_LSTM_FD"] == True)):
            h_a = self._model.processed_a.predict([np.array([state])])
        else:
            h_a = self._model._forward_dynamics_net.predict([state])[0]
        return h_a
    
    def predict(self, state):
        """
            Compute distance between two states
        """
        # print("state shape: ", np.array(state).shape)
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        # state2 = np.array(norm_state(state2, self._state_bounds), dtype=self.getSettings()['float_type'])
        if ((("train_LSTM_FD" in self._settings)
                    and (self._settings["train_LSTM_FD"] == True))
                    # or
                    # settings["use_learned_reward_function"] == "dual"
                    ):
            
            state2 = state[:, :self._settings["dense_state_size"]]
            
            if ("remove_character_state_features" in self._settings):
                ### Remove ground reaction forces from state
                state2 = state2[:, :-self._settings["remove_character_state_features"]]
            ### Used because we need to keep two separate RNN networks and not mix the hidden states
            h_a = self._model.encode_a.predict([np.array([state])])
            h_b = self._model.encode_b.predict([np.array([state2])])
            state_ = euclidean_distance_np((h_a, h_b))[0]
            # print ("siamese dist: ", state_)
            # state_ = self._model._forward_dynamics_net.predict([np.array([state]), np.array([state2])])[0]
        else:
            # print ("State shape: ", state.shape, " state2 shape: ", state2.shape)
            # state2 = np.array(norm_state(state2, self._state_bounds), dtype=self.getSettings()['float_type'])
            state2 = state[:, :self._settings["dense_state_size"]]
            state_ = self._model._forward_dynamics_net.predict([state, state2])[0]
        # dist_ = np.array(self._contrastive_loss([te_pair1, te_pair2, 0]))[0]
        # print("state_ shape: ", np.array(state_).shape)
        return state_
    
    def predictWithDropout(self, state, action):
        # "dropout"
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        self._model.setStates(state)
        self._model.setActions(action)
        state_ = scale_state(self._forwardDynamics_drop()[0], self._state_bounds)
        return state_
    
    def predict_std(self, state, action, p=1.0):
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        state_ = self._forwardDynamics_std() * (action_bound_std(self._state_bounds))
        return state_
    
    def predict_reward(self, state):
        """
            Predict reward which is inverse of distance metric
        """
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        # state2 = np.array(norm_state(state2, self._state_bounds), dtype=self.getSettings()['float_type'])
        if (("train_LSTM_Reward" in self._settings)
            and (self._settings["train_LSTM_Reward"] == True)):
            # print ("state shape: ", state.shape)
            state2 = state[:, :self._settings["dense_state_size"]]
            if ("remove_character_state_features" in self._settings):
                ### Remove ground reaction forces from state
                state2 = state2[:, :-self._settings["remove_character_state_features"]]
            ### Used because we need to keep two separate RNN networks and not mix the hidden states
            h_a = self._model.encode_a.predict([np.array([state])])
            h_b = self._model.encode_b.predict([np.array([state2])])
            reward_ = euclidean_distance_np((h_a, h_b))[0]
            # print ("siamese dist: ", state_)
            # state_ = self._model._forward_dynamics_net.predict([np.array([state]), np.array([state2])])[0]
        else:
            state2 = state[:, :self._settings["dense_state_size"]]
            predicted_reward = self._model._reward_net.predict([state, state2])[0]
            # reward_ = scale_reward(predicted_reward, self.getRewardBounds()) # * (1.0 / (1.0- self.getSettings()['discount_factor']))
            reward_ = predicted_reward
            
        return reward_
    
    def predict_reward_encoding(self, state):
        """
            Predict reward which is inverse of distance metric
        """
        # state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
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

    def bellman_error(self, states, actions, result_states, rewards):
        self.reset()
        if (("train_LSTM_FD" in self._settings)
                    and (self._settings["train_LSTM_FD"] == True)):
            sequences0, sequences1, targets_ = create_sequences(states, result_states, self._settings)
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
                if ("remove_character_state_features" in self._settings):
                    ### Remove ground reaction forces from state
                    sequences1 = sequences1[:, :, :-self._settings["remove_character_state_features"]]
                predicted_y = self._model._combination.predict([sequences0, sequences1], batch_size=sequences0.shape[0])
                # print ("fd error, predicted_y: ", predicted_y)
                targets__ = np.mean(targets_, axis=1)
                # print ("fd error, targets_ : ", targets_)
                # print ("fd error, targets__: ", targets__)
                errors.append( compute_accuracy(predicted_y[2], targets__) )
            # predicted_y = self._model._forward_dynamics_net.predict([np.array([[sequences0[0]]]), np.array([[sequences1[0]]])])
            # te_acc = compute_accuracy(predicted_y, np.array([targets_[0]]) )
            te_acc = np.mean(errors)
        else:
            states = np.concatenate((states, result_states), axis=0)
            te_pair1, te_pair2, te_y = create_pairs2(states, self._settings)
        
            # state_ = self._model._forward_dynamics_net.predict([state, state2])[0]
            predicted_y = self._model._forward_dynamics_net.predict([te_pair1, te_pair2])
            # print ("predicted_y: ", predicted_y)
            # print ("te_y: ", te_y)
            te_acc = compute_accuracy(predicted_y, te_y)
            # print ("te_acc: ", te_acc)
            
        # predicted_y = self._model._forward_dynamics_net.predict([te_pair1, te_pair2])
        return te_acc
    
    def reward_error(self, states, actions, result_states, rewards):
        # rewards = rewards * (1.0/(1.0-self.getSettings()['discount_factor'])) # scale rewards
        self.reset()
        if (("train_LSTM_Reward" in self._settings)
                    and (self._settings["train_LSTM_Reward"] == True)):
            sequences0, sequences1, targets_ = create_sequences(states, result_states, self._settings)
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
                if ("remove_character_state_features" in self._settings):
                    ### Remove ground reaction forces from state
                    sequences1 = sequences1[:, :, :-self._settings["remove_character_state_features"]]
                predicted_y = self._model._combination.predict([sequences0, sequences1], batch_size=sequences0.shape[0])
                # predicted_y = self._model._reward_net.predict([sequences0, sequences1], batch_size=sequences0.shape[0])
                # print ("fd error, predicted_y: ", predicted_y)
                targets__ = np.mean(targets_, axis=1) 
                print ("fd error, targets_ : ", np.mean(targets__))
                # print ("fd error, targets__: ", targets__)
                # print ("predicted_y: ", predicted_y[0].shape, predicted_y[1].shape, predicted_y[2].shape)
                # print ("predicted_y: ", predicted_y[2])
                # print ("targets__: ", targets__)
                error = contrastive_loss_np(predicted_y[2], targets__)
                # error = compute_accuracy(predicted_y[2], targets__)
                print ("error: ", error)
                errors.append(error )
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
        # hf.create_dataset('_result_state_bounds', data=self.getResultStateBounds())
        hf.flush()
        hf.close()
        suffix = ".h5"
        ### Save models
        # self._model._actor_train.save(fileName+"_actor_train"+suffix, overwrite=True)
        self._model._forward_dynamics_net.save(fileName+"_FD"+suffix, overwrite=True)
        self._model._reward_net.save(fileName+"_reward"+suffix, overwrite=True)
        self._modelTarget._forward_dynamics_net.save(fileName+"_FD_T"+suffix, overwrite=True)
        self._modelTarget._reward_net.save(fileName+"_reward_T"+suffix, overwrite=True)
        # print ("self._model._actor_train: ", self._model._actor_train)
        """
        from keras.utils import plot_model
        ### Save model design as image
        plot_model(self._model._forward_dynamics_net, to_file=fileName+"_FD"+'.svg', show_shapes=True)
        plot_model(self._model._reward_net, to_file=fileName+"_reward"+'.svg', show_shapes=True)
        plot_model(self._model._combination, to_file=fileName+"_fd_combination"+'.svg', show_shapes=True)
        """
        
    def loadFrom(self, fileName):
        import h5py
        from util.utils import load_keras_model
        suffix = ".h5"
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Loading agent: ", fileName)
        # with K.get_session().graph.as_default() as g:
        ### Need to lead the model this way because the learning model's State expects batches...
        forward_dynamics_net = load_keras_model(fileName+"_FD"+suffix, custom_objects={'contrastive_loss': contrastive_loss})
        reward_net = load_keras_model(fileName+"_reward"+suffix, custom_objects={'contrastive_loss': contrastive_loss})
        # if ("simulation_model" in self.getSettings() and
        #     (self.getSettings()["simulation_model"] == True)):
        if (True): ### Because the simulation and learning use different model types (statefull vs stateless lstms...)
            self._model._forward_dynamics_net.set_weights(forward_dynamics_net.get_weights())
            self._model._forward_dynamics_net.optimizer = forward_dynamics_net.optimizer
            self._model._reward_net.set_weights(reward_net.get_weights())
            self._model._reward_net.optimizer = reward_net.optimizer
        else:
            self._model._forward_dynamics_net = forward_dynamics_net
            self._model._reward_net = reward_net
            
        self._forward_dynamics_net = self._model._forward_dynamics_net
        self._reward_net = self._model._reward_net
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("******** self._forward_dynamics_net: ", self._forward_dynamics_net)
        if (self._modelTarget is not None):
            self._modelTarget._forward_dynamics_net = load_keras_model(fileName+"_FD_T"+suffix)
            self._modelTarget._reward_net = load_keras_model(fileName+"_reward_T"+suffix)
        # self._model._actor_train = load_keras_model(fileName+"_actor_train"+suffix, custom_objects={'loss': pos_y})
        # self._value = K.function([self._model.getStateSymbolicVariable(), K.learning_phase()], [self.__value])
        # self._value_Target = K.function([self._model.getResultStateSymbolicVariable(), K.learning_phase()], [self.__value_Target])
        hf = h5py.File(fileName+"_bounds.h5",'r')
        self.setStateBounds(np.array(hf.get('_state_bounds')))
        self.setRewardBounds(np.array(hf.get('_reward_bounds')))
        self.setActionBounds(np.array(hf.get('_action_bounds')))
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("fd self.getStateBounds(): ", self.getStateBounds())
        # self._result_state_bounds = np.array(hf.get('_result_state_bounds'))
        hf.close()
        