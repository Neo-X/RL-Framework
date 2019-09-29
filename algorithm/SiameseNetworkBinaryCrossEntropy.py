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

# For debugging
# theano.config.mode='FAST_COMPILE'
from algorithm.KERASAlgorithm import KERASAlgorithm
from algorithm.SiameseNetwork import *

def l1_distance_(vects):
    x, y = vects
    return K.abs(x - y)

def l1_distance_np_(vects):
    x, y = vects
    return K.abs(x - y)

class SiameseNetworkBinaryCrossEntropy(KERASAlgorithm):
    
    def __init__(self, model, state_length, action_length, state_bounds, action_bounds, settings_, reward_bounds=0, print_info=False):

        super(SiameseNetworkBinaryCrossEntropy,self).__init__(model, state_length, action_length, state_bounds, action_bounds, reward_bounds, settings_)
        self._model = model
        self._learning_rate = self.getSettings()["fd_learning_rate"]
        self._regularization_weight = 1e-6
        
        self._distance_func = l1_distance_
        self._distance_func_np = l1_distance_np_
        condition_reward_on_result_state = False
        self._train_combined_loss = False

        inputs_ = [self._model.getStateSymbolicVariable()] 
        self._model._forward_dynamics_net = Model(inputs=inputs_, outputs=self._model._forward_dynamics_net)
        if (print_info):
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print("FD Net summary: ", self._model._forward_dynamics_net.summary())
        
        if ("force_use_actor_state_for_critic" in self._settings
            and (self._settings["force_use_actor_state_for_critic"] == True)):
            inputs_ = [self._model.getStateSymbolicVariable()]
        else:  
            inputs_ = [self._model.getResultStateSymbolicVariable()] 
        self._model._reward_net = Model(inputs=inputs_, outputs=self._model._reward_net)
        if (print_info):
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print("FD Reward Net summary: ", self._model._reward_net.summary())

        self._modelTarget = None
        SiameseNetworkBinaryCrossEntropy.compile(self)
    
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
        processed_a = self._model._forward_dynamics_net(self._model.getStateSymbolicVariable())
        self._model.processed_a = Model(inputs=[self._model.getStateSymbolicVariable()], outputs=processed_a)
        processed_b = self._model._forward_dynamics_net(state_copy)
        self._model.processed_b = Model(inputs=[state_copy], outputs=processed_b)

        if ( "return_rnn_sequence" in self.getSettings()
             and (self.getSettings()["return_rnn_sequence"])):        
            processed_a_r_seq , processed_a_r, processed_a_r_c  = self._model._reward_net(self._model.getResultStateSymbolicVariable())
            processed_b_r_seq , processed_b_r, processed_b_r_c = self._model._reward_net(result_state_copy)
        else:
            processed_a_r = self._model._reward_net(self._model.getResultStateSymbolicVariable())
            processed_b_r = self._model._reward_net(result_state_copy)
        self._model.processed_a_r = Model(inputs=[self._model.getResultStateSymbolicVariable()], outputs=processed_a_r)
        print ("processed_b_r: ", repr(processed_b_r))
        self._model.processed_b_r = Model(inputs=[result_state_copy], outputs=processed_b_r)
        
        distance_fd = keras.layers.Lambda(self._distance_func, output_shape=eucl_dist_output_shape_seq)([processed_a, processed_b])
        distance_fd_weighted = keras.layers.Dense(1, activation = 'sigmoid')(distance_fd)
        
        distance_r = keras.layers.Lambda(self._distance_func, output_shape=eucl_dist_output_shape_seq)([processed_a_r, processed_b_r])
        print ("distance_r: ", repr(distance_r))
        encode_input__ = keras.layers.Input(shape=keras.backend.int_shape(distance_r)[1:]
                                                                              , name="encoding_2"
                                                                              )
        print ("encode_input__: ", repr(encode_input__))
        distance_r_weighted = keras.layers.Dense(1, activation = 'sigmoid')(encode_input__)
        self._distance_weighting_ = Model(inputs=[encode_input__], outputs=distance_r_weighted)
        distance_r_weighted = self._distance_weighting_(distance_r)
        print ("distance_r_weighted: ", repr(distance_r_weighted))
        
        # distance_r_seq = keras.layers.Lambda(self._distance_func, output_shape=eucl_dist_output_shape_seq)([processed_a_r_seq, processed_b_r_seq])
        # distance_r_seq = self._distance_func([processed_a_r_seq, processed_b_r_seq])
        self._model._forward_dynamics_net = Model(inputs=[self._model.getStateSymbolicVariable()
                                                          ,state_copy 
                                                          ]
                                                  , outputs=distance_fd_weighted
                                                  )
        
        self._model._reward_net = Model(inputs=[self._model.getResultStateSymbolicVariable()
                                                          ,result_state_copy
                                                          ]
                                                          , outputs=distance_r_weighted
                                                          )
        if ( "return_rnn_sequence" in self.getSettings()
             and (self.getSettings()["return_rnn_sequence"])):
            distance_r_seq = keras.layers.Lambda(self._distance_func, output_shape=eucl_dist_output_shape_seq)([processed_a_r_seq, processed_b_r_seq])
            print ("distance_r_seq: ", repr(distance_r_seq))
            distance_r_weighted_seq = keras.layers.TimeDistributed(self._distance_weighting_)(distance_r_seq)
            print ("distance_r_weighted_seq: ", repr(distance_r_weighted_seq))
            self._model._reward_net_seq = Model(inputs=[self._model.getResultStateSymbolicVariable()
                                                              ,result_state_copy
                                                              ]
                                                              , outputs=distance_r_weighted_seq
                                                              )

        # sgd = SGD(lr=0.0005, momentum=0.9)
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), beta_1=np.float32(0.95), 
                                    beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0),
                                    clipnorm=2.5)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("sgd, actor: ", sgd)
            print ("Clipping: ", sgd.decay)
        self._model._forward_dynamics_net.compile(loss='binary_crossentropy', optimizer=sgd)

        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), beta_1=np.float32(0.95), 
                                    beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0),
                                    clipnorm=2.5)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("sgd, actor: ", sgd)
            print ("Clipping: ", sgd.decay)
        self._model._reward_net.compile(loss='binary_crossentropy', optimizer=sgd)
        
        
    def reset(self):
        """
            Reset any state for the agent model
        """
        self._model.reset()
        self._model._reward_net.reset_states()
        self._model._forward_dynamics_net.reset_states()
        self._model.processed_a.reset_states()
        self._model.processed_b.reset_states()
        self._model.processed_a_r.reset_states()
        self._model.processed_b_r.reset_states()
        if not (self._modelTarget is None):
            self._modelTarget.reset()
            
    def getNetworkParameters(self):
        params = []
        params.append(copy.deepcopy(self._model._forward_dynamics_net.get_weights()))
        params.append(copy.deepcopy(self._model._reward_net.get_weights()))
        return params
    
    def setNetworkParameters(self, params):
        self._model._forward_dynamics_net.set_weights(params[0])
        self._model._reward_net.set_weights(params[1])
        
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
                
    def train(self, states, actions, result_states, rewards, falls=None, updates=1, batch_size=None, p=1, lstm=True, datas=None):
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
                sequences0, sequences1, targets_ = create_sequences(states, result_states, self._settings)
            else:
                sequences0, sequences1, targets_ = create_multitask_sequences(states, result_states, datas["task_id"], self._settings)
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
            state_ = 1 - self._model._forward_dynamics_net.predict([np.array([state]), np.array([state2])])
            """
            h_a = self._model.processed_a.predict([np.array([state])])
            h_b = self._model.processed_b.predict([np.array([state2])])
            state_ = self._distance_func_np((h_a, h_b))[0]
            """
            # print ("siamese dist: ", state_)
            # state_ = self._model._forward_dynamics_net.predict([np.array([state]), np.array([state2])])[0]
        else:
            state_ = 1 - self._model._forward_dynamics_net.predict([state, state2])[0]
        # dist_ = np.array(self._contrastive_loss([te_pair1, te_pair2, 0]))[0]
        # print("state_ shape: ", np.array(state_).shape)
        return [state_]
    
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
            reward_ = 1 - self._model._reward_net.predict([np.array([state]),np.array([state2])])
            """
            h_a = self._model.processed_a_r.predict([np.array([state])])
            h_b = self._model.processed_b_r.predict([np.array([state2])])
            reward_ = self._distance_func_np((h_a, h_b))[0]
            """
            # print ("siamese dist: ", reward_)
            # state_ = self._model._forward_dynamics_net.predict([np.array([state]), np.array([state2])])[0]
        else:
            predicted_reward = 1 - self._model._reward_net.predict([state, state2])[0]
            # reward_ = scale_reward(predicted_reward, self.getRewardBounds()) # * (1.0 / (1.0- self.getSettings()['discount_factor']))
            reward_ = predicted_reward
            
        return [reward_]
    
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
        return 1 - self._model._forward_dynamics_net.predict([states, actions])[0]
    
    def predict_reward_(self, states, actions):
        """
            This data should already be normalized
        """
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        states = np.array(norm_state(states, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        actions = np.array(norm_state(actions, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        predicted_reward = 1 - self._model._reward_net_seq.predict([states, actions], batch_size=1)[0]
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
                # print ("sequences0 shape: ", sequences0.shape)
                predicted_y = 1 - self._model._reward_net.predict([sequences0, sequences1], batch_size=sequences0.shape[0])
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
        self._model._reward_net.save(fileName+"_reward"+suffix, overwrite=True)
        # print ("self._model._actor_train: ", self._model._actor_train)
        
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
            self._modelTarget._forward_dynamics_net = load_keras_model(fileName+"_actor_T"+suffix)
            self._modelTarget._reward_net = load_keras_model(fileName+"_reward_net_T"+suffix)
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
        