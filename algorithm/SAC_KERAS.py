## Nothing for now...

# from modular_rl import *

# ================================================================
# Soft Actor Critic
# ================================================================


import numpy as np
import sys
import copy
from model.ModelUtil import *
from algorithm.KERASAlgorithm import KERASAlgorithm
from model.LearningUtil import  loglikelihood_keras, loglikelihood_np
from keras.optimizers import SGD
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from collections import OrderedDict


# For debugging
# theano.config.mode='FAST_COMPILE'
# from DeepCACLA import DeepCACLA

class SAC_KERAS(KERASAlgorithm):

    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=False):
        """
            In order to get this to work we need to be careful not to update the actor parameters
            when updating the critic. This can be an issue when the Concatenating networks together.
            The first first network becomes a part of the second. However you can still access the first
            network by itself but an updates on the second network will effect the first network.
            Care needs to be taken to make sure only the parameters of the second network are updated.
        """

        super(SAC_KERAS, self).__init__(model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_,
                                       print_info=False)

        assert("use_stochastic_policy" in self.getSettings() and self.getSettings()["use_stochastic_policy"]), "Must use a stochastic policy with SAC, set use_stochastic_policy=true"

        self._model._actor = Model(inputs=[self._model.getStateSymbolicVariable()], outputs=self._model._actor)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"][
            'train']):
            print("Actor summary: ", self._model._actor.summary())

        if ("use_centralized_critic" in self.getSettings()
                and (self.getSettings()["use_centralized_critic"] == True)
                and False):
            self._model._critic = Model(inputs=[self._model.getResultStateSymbolicVariable(),
                                                self._model.getActionSymbolicVariable()], outputs=self._model._critic)
        else:
            self._model._critic = Model(inputs=[self._model.getStateSymbolicVariable(),
                                                self._model.getActionSymbolicVariable()], outputs=self._model._critic)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"][
            'train']):
            print("Critic summary: ", self._model._critic.summary())

        self._modelTarget = type(self._model)(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_,
                                              print_info=False)
        self._modelTarget._actor = Model(inputs=[self._modelTarget.getStateSymbolicVariable()],
                                         outputs=self._modelTarget._actor)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"][
            'train']):
            print("Target Actor summary: ", self._modelTarget._actor.summary())
        if ("use_centralized_critic" in self.getSettings()
                and (self.getSettings()["use_centralized_critic"] == True)
                and False):
            self._modelTarget._critic = Model(inputs=[self._modelTarget.getResultStateSymbolicVariable(),
                                                      self._modelTarget.getActionSymbolicVariable()],
                                              outputs=self._modelTarget._critic)
        else:
            self._modelTarget._critic = Model(inputs=[self._modelTarget.getStateSymbolicVariable(),
                                                      self._modelTarget.getActionSymbolicVariable()],
                                              outputs=self._modelTarget._critic)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"][
            'train']):
            print("Target Critic summary: ", self._modelTarget._critic.summary())
            
        self._alpha = 0.01
        self._target_entropy = -len(action_bounds[0])

        self._discount_factor = self.getSettings()['discount_factor']
        self._rho = self.getSettings()['rho']
        self._rms_epsilon = self.getSettings()['rms_epsilon']

        sgd = keras.optimizers.Adam(lr=self.getSettings()['critic_learning_rate'], beta_1=0.9, beta_2=0.999,
                                    epsilon=self._rms_epsilon, decay=0.0)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"][
            'train']):
            print("Clipping: ", sgd.decay)
        self._model.getCriticNetwork().compile(loss='mse', optimizer=sgd)

        sgd = keras.optimizers.Adam(lr=self.getSettings()['critic_learning_rate'], beta_1=0.9, beta_2=0.999,
                                    epsilon=self._rms_epsilon, decay=0.0)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"][
            'train']):
            print("Clipping: ", sgd.decay)
        self._modelTarget.getCriticNetwork().compile(loss='mse', optimizer=sgd)

        SAC_KERAS.compile(self)

    def compile(self):

        self._q_valsActA = self._model.getActorNetwork()(self._model.getStateSymbolicVariable())[:,
                           :self._action_length]
        if ('use_stochastic_policy' in self.getSettings() and (self.getSettings()['use_stochastic_policy'])):
            # self._q_valsActASTD = (self._model.getActorNetwork()(self._model.getStateSymbolicVariable())[:,self._action_length:]) + 1e-2
            self._q_valsActASTD = ((self._model.getActorNetwork()(self._model.getStateSymbolicVariable())[:,
                                    self._action_length:]) * self.getSettings()['exploration_rate']) + 1e-2
        else:
            self._q_valsActASTD = (K.ones_like(self._q_valsActA)) * self.getSettings()['exploration_rate']

        self._q_valsActTarget_State = self._modelTarget.getActorNetwork()(self._model.getStateSymbolicVariable())[:,
                                      :self._action_length]
        if ('use_stochastic_policy' in self.getSettings() and (self.getSettings()['use_stochastic_policy'])):
            # self._q_valsActTargetSTD = (self._modelTarget.getActorNetwork()(self._model.getStateSymbolicVariable())[:,self._action_length:]) + 1e-2
            self._q_valsActTargetSTD = ((self._modelTarget.getActorNetwork()(self._model.getStateSymbolicVariable())[:,
                                         self._action_length:]) * self.getSettings()['exploration_rate']) + 1e-2
        else:

            self._q_valsActTargetSTD = (K.ones_like(self._q_valsActTarget_State)) * self.getSettings()['exploration_rate']

        self._logprob = loglikelihood_keras(self._modelTarget.getActionSymbolicVariable(), self._q_valsActA, K.exp(self._q_valsActASTD / 2.0),
                                      self._action_length)
        self._logprob_target = loglikelihood_keras(self._modelTarget.getActionSymbolicVariable(), self._q_valsActTarget_State,
                                             K.exp(self._q_valsActTargetSTD / 2.0), self._action_length)

        if ("use_centralized_critic" in self.getSettings()
                and (self.getSettings()["use_centralized_critic"] == True)
                and False):
            self._q_function = self._model.getCriticNetwork()(
                [self._model.getResultStateSymbolicVariable(), self._model.getActionSymbolicVariable()])
            self._q_function_Target = self._model.getCriticNetwork()(
                [self._modelTarget.getResultStateSymbolicVariable(), self._model.getActionSymbolicVariable()])
            self._q_func = K.function(
                [self._model.getResultStateSymbolicVariable(), self._model.getActionSymbolicVariable()],
                [self._q_function])
        else:
            self._q_function = self._model.getCriticNetwork()(
                [self._model.getStateSymbolicVariable(), self._model.getActionSymbolicVariable()])
            self._q_function_Target = self._model.getCriticNetwork()(
                [self._modelTarget.getStateSymbolicVariable(), self._model.getActionSymbolicVariable()])
            self._q_func = K.function([self._model.getStateSymbolicVariable(), self._model.getActionSymbolicVariable()],
                                      [self._q_function])

        q_vals_b = self._q_function_Target - self._logprob
        target_tmp_ = self._model.getRewardSymbolicVariable() + ((self._discount_factor * q_vals_b))
        diff = target_tmp_ - self._q_function
        self._q_loss = K.mean(K.mean(diff, axis=-1))
        # print ("Initial W " + str(self._w_o.get_value()) )

        # self._get_gradients = K.function(inputs=[self._model._stateInput], outputs=gradients)

        self._q_action_std = K.function([self._model._stateInput], [self._q_valsActASTD])

        # For the combined model we will only train the actor
        self._model.getCriticNetwork().trainable = False

        self._act = self._model.getActorNetwork()([self._model.getStateSymbolicVariable()])

        def sample_actions(_act):
            _act_mean, _act_std = _act[:, :self._action_length], _act[:, self._action_length:]
            taken_action = 0 * K.random_normal(K.shape(_act_mean)) * K.exp(_act_std) + _act_mean
            return taken_action

        sample_layer = keras.layers.Lambda(sample_actions)
        _act_unnormalized = sample_layer(self._act)

        def get_logprob(_act_unnormalized):
            return loglikelihood_keras(
                _act_unnormalized,
                self._act[:, :self._action_length],
                K.exp(self._act[:, self._action_length:] / 2.0),
                self._action_length)

        logprob_layer = keras.layers.Lambda(get_logprob)
        _act_logprob = logprob_layer(_act_unnormalized)

        def neg_y(true_y, pred_y):
            return K.mean(_act_logprob - pred_y)

        if ("use_centralized_critic" in self.getSettings()
                and (self.getSettings()["use_centralized_critic"] == True)
                and False):
            self._qFunc = (self._model.getCriticNetwork()(
                [self._model.getResultStateSymbolicVariable(),
                 _act_unnormalized]))
            self._combined = Model(input=[self._model.getStateSymbolicVariable(),
                                          self._model.getResultStateSymbolicVariable()],
                                   output=self._qFunc)
        else:
            self._qFunc = (self._model.getCriticNetwork()(
                [self._model.getStateSymbolicVariable(),
                 _act_unnormalized]))
            self._combined = Model(input=[self._model.getStateSymbolicVariable()],
                                   output=self._qFunc)

        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['learning_rate']),
                                    beta_1=np.float32(0.9), beta_2=np.float32(0.999),
                                    epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0000001),
                                    amsgrad=False)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"][
            'train']):
            print("Clipping: ", sgd.decay)
            print("sgd, critic: ", sgd)
        self._combined.compile(loss=[neg_y], optimizer=sgd)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"][
            'train']):
            print("combined qFun Net summary: ", self._combined.summary())

        if (self.getSettings()["regularization_weight"] > 0.0000001):
            self._actor_regularization = K.sum(self._model.getActorNetwork().losses)
        else:
            self._actor_regularization = K.sum(self._model.getActorNetwork().losses)

        if (self.getSettings()["critic_regularization_weight"] > 0.0000001):
            self._critic_regularization = K.sum(self._model.getCriticNetwork().losses)
        else:
            self._critic_regularization = K.sum(self._model.getCriticNetwork().losses)

        self._get_actor_regularization = K.function([], [self._actor_regularization])
        self._get_critic_regularization = K.function([], [self._critic_regularization])
        self._get_critic_loss = K.function([self._model.getStateSymbolicVariable(),
                                            self._model.getActionSymbolicVariable(),
                                            self._model.getRewardSymbolicVariable(),
                                            self._model.getResultStateSymbolicVariable(),
                                            K.learning_phase()
                                            ], [self._q_loss])
        self._get_actor_loss = K.function([self._model.getStateSymbolicVariable()
                                           # ,K.learning_phase()
                                           ], [self._qFunc])

    def getGrads(self, states, actions=None, alreadyNormed=False):
        """
            The states should be normalized
        """
        # self.setData(states, actions, rewards, result_states)
        if (alreadyNormed == False):
            states = norm_state(states, self._state_bounds)
        states = np.array(states, dtype=theano.config.floatX)
        grads = self._get_gradients([states])
        return grads

    def updateTargetModel(self):
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"][
            'debug']):
            print("Updating target Model")
        """
            Target model updates
        """
        # return
        ## I guess it is okay to lerp the entire network even though we only really want to
        ## lerp the value function part of the networks, the target policy is not used for anythings
        all_paramsA = self._model.getCriticNetwork().get_weights()
        all_paramsB = self._modelTarget.getCriticNetwork().get_weights()
        if ('target_net_interp_weight' in self.getSettings()):
            lerp_weight = self.getSettings()['target_net_interp_weight']
        else:
            lerp_weight = 0.001
        # vals = lasagne.layers.helper.get_all_param_values(self._l_outActA)

        all_params = []
        for paramsA, paramsB in zip(all_paramsA, all_paramsB):
            # print ("paramsA: " + str(paramsA))
            # print ("paramsB: " + str(paramsB))
            params = (lerp_weight * paramsA) + ((1.0 - lerp_weight) * paramsB)
            all_params.append(params)
        self._modelTarget.getCriticNetwork().set_weights(all_params)

        all_paramsA_Act = self._model.getActorNetwork().get_weights()
        all_paramsB_Act = self._modelTarget.getActorNetwork().get_weights()

        all_params = []
        for paramsA, paramsB in zip(all_paramsA_Act, all_paramsB_Act):
            # print ("paramsA: " + str(paramsA))
            # print ("paramsB: " + str(paramsB))
            params = (lerp_weight * paramsA) + ((1.0 - lerp_weight) * paramsB)
            all_params.append(params)
        self._modelTarget.getActorNetwork().set_weights(all_params)

    def updateTargetModelCritic(self):
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"][
            'debug']):
            print("Updating target critic Model")
        """
            Target model updates
        """
        # return
        ## I guess it is okay to lerp the entire network even though we only really want to
        ## lerp the value function part of the networks, the target policy is not used for anythings
        all_paramsA = self._model.getCriticNetwork().get_weights()
        all_paramsB = self._modelTarget.getCriticNetwork().get_weights()
        if ('target_net_interp_weight' in self.getSettings()):
            lerp_weight = self.getSettings()['target_net_interp_weight']
        else:
            lerp_weight = 0.001
        # vals = lasagne.layers.helper.get_all_param_values(self._l_outActA)

        all_params = []
        for paramsA, paramsB in zip(all_paramsA, all_paramsB):
            # print ("paramsA: " + str(paramsA))
            # print ("paramsB: " + str(paramsB))
            params = (lerp_weight * paramsA) + ((1.0 - lerp_weight) * paramsB)
            all_params.append(params)
        self._modelTarget.getCriticNetwork().set_weights(all_params)

    def updateTargetModelActor(self):
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"][
            'debug']):
            print("Updating target Actor Model")
        """
            Target model updates
        """
        # return
        ## I guess it is okay to lerp the entire network even though we only really want to
        ## lerp the value function part of the networks, the target policy is not used for anythings
        if ('target_net_interp_weight' in self.getSettings()):
            lerp_weight = self.getSettings()['target_net_interp_weight']
        else:
            lerp_weight = 0.001
        # vals = lasagne.layers.helper.get_all_param_values(self._l_outActA)

        all_paramsA_Act = self._model.getActorNetwork().get_weights()
        all_paramsB_Act = self._modelTarget.getActorNetwork().get_weights()

        all_params = []
        for paramsA, paramsB in zip(all_paramsA_Act, all_paramsB_Act):
            # print ("paramsA: " + str(paramsA))
            # print ("paramsB: " + str(paramsB))
            params = (lerp_weight * paramsA) + ((1.0 - lerp_weight) * paramsB)
            all_params.append(params)
        self._modelTarget.getActorNetwork().set_weights(all_params)

    def updateTargetModelValue(self):
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"][
            'debug']):
            print("Updating MBAE target Model")
        """
            Target model updates
        """
        all_paramsA = lasagne.layers.helper.get_all_param_values(self._model._value_function)
        # all_paramsActA = lasagne.layers.helper.get_all_param_values(self._model.getActorNetwork())
        lasagne.layers.helper.set_all_param_values(self._modelTarget._value_function, all_paramsA)
        # lasagne.layers.helper.set_all_param_values(self._modelTarget.getActorNetwork(), all_paramsActA)

    def getNetworkParameters(self):
        params = []
        params.append(copy.deepcopy(self._model.getCriticNetwork().get_weights()))
        params.append(copy.deepcopy(self._model.getActorNetwork().get_weights()))
        params.append(copy.deepcopy(self._modelTarget.getCriticNetwork().get_weights()))
        params.append(copy.deepcopy(self._modelTarget.getActorNetwork().get_weights()))
        return params

    def setNetworkParameters(self, params):
        """
        for i in range(len(params[0])):
            params[0][i] = np.array(params[0][i], dtype=theano.config.floatX)
            """
        self._model.getCriticNetwork().set_weights(params[0])
        self._model.getActorNetwork().set_weights(params[1])
        self._modelTarget.getCriticNetwork().set_weights(params[2])
        self._modelTarget.getActorNetwork().set_weights(params[3])

    def setData(self, states, actions, rewards, result_states, fallen):
        self._model.setStates(states)
        self._model.setResultStates(result_states)
        self._model.setActions(actions)
        self._model.setRewards(rewards)
        self._modelTarget.setStates(states)
        self._modelTarget.setResultStates(result_states)
        self._modelTarget.setActions(actions)
        self._modelTarget.setRewards(rewards)
        # print ("Falls: ", fallen)
        # self._fallen_shared.set_value(fallen)
        # diff_ = self.bellman_error(states, actions, rewards, result_states, falls)
        ## Easy fix for computing actor loss
        # diff = self._bellman_error2()
        # self._tmp_diff_shared.set_value(diff)

        # _targets = rewards + (self._discount_factor * self._q_valsTargetNextState )

    def trainCritic(self, states, actions, rewards, result_states, falls, G_t=[[0]], p=1.0):

        # self.setData(states, actions, rewards, result_states, falls)
        ## get actions for target policy
        actor_data = self._modelTarget.getActorNetwork().predict(result_states, batch_size=states.shape[0])
        target_means = actor_data[..., :self._action_length]
        target_stds = actor_data[..., self._action_length:]
        target_actions = np.random.normal(0, 1, target_means.shape) * np.exp(target_stds / 2.0) + target_means

        target_actions = target_actions.astype(np.float32)
        target_means = target_means.astype(np.float32)
        target_stds = target_stds.astype(np.float32)
        ## Get next q value
        q_vals_b = self._modelTarget.getCriticNetwork().predict([result_states, target_actions],
                                                                batch_size=states.shape[0])
        logprobs = loglikelihood_np(
            target_actions,
            target_means,
            np.exp(target_stds / 2.0),
            self._action_length)
        mean_probs = np.mean(logprobs)
        print ("q_vals_b: ", np.mean(q_vals_b), " logprobs: ", mean_probs, " self._alpha: ", self._alpha)
        step = 0.3 / self.getSettings()['critic_updates_per_actor_update']
        # steps = self.getSettings()['critic_updates_per_actor_update']
        if (mean_probs > self._target_entropy):
            self._alpha = np.clip(self._alpha * (1 - step), 0.0000001, 1)
        elif (mean_probs <= self._target_entropy):
            self._alpha = np.clip(self._alpha * (1.01 + step), 0.0000001, 1)
                
        q_vals_b = q_vals_b - (self._alpha * logprobs)
        # q_vals_b = self._q_val()
        ## Compute target values
        # target_tmp_ = rewards + ((self._discount_factor* q_vals_b )* falls)
        target_tmp_ = rewards + ((self._discount_factor * q_vals_b))
        # self.setData(states, actions, rewards, result_states, falls)
        # self._tmp_target_shared.set_value(target_tmp_)

        # self._target = T.mul(T.add(self._model.getRewardSymbolicVariable(), T.mul(self._discount_factor, self._q_valsB )), self._Fallen)

        if ('anneal_learning_rate' in self.getSettings()
                and (self.getSettings()['anneal_learning_rate'] == True)):
            K.set_value(self._model.getCriticNetwork().optimizer.lr,
                        np.float32(self.getSettings()['critic_learning_rate']) * p)
            # lr = K.get_value(self._model.getCriticNetwork().optimizer.lr)
            # print ("New critic learning rate: ", lr)

        loss = self._model.getCriticNetwork().fit([states, actions], target_tmp_,
                                                  batch_size=states.shape[0],
                                                  epochs=1,
                                                  verbose=False,
                                                  shuffle=False)

        self.updateTargetModelCritic()
        loss = loss.history['loss'][0]
        return loss

    def trainActor(self, states, actions, rewards, result_states, falls, advantage,
                   exp_actions, G_t=[[0]], forwardDynamicsModel=None, p=1.0):
        # self.setData(states, actions, rewards, result_states, falls)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"][
            'debug']):
            print("values: ", np.mean(self._q_val() * (1.0 / (1.0 - self.getSettings()['discount_factor']))), " std: ",
                  np.std(self._q_val() * (1.0 / (1.0 - self.getSettings()['discount_factor']))))
            print("Rewards: ", np.mean(rewards), " std: ", np.std(rewards), " shape: ", np.array(rewards).shape)
        # print("Policy mean: ", np.mean(self._q_action(), axis=0))
        loss = 0
        # loss = self._trainActor()
        if ('anneal_learning_rate' in self.getSettings()
                and (self.getSettings()['anneal_learning_rate'] == True)):
            K.set_value(self._combined.optimizer.lr, np.float32(self.getSettings()['learning_rate']) * p)

        ### The rewards are not used in this update, just a placeholder
        score = self._combined.fit([states], rewards,
                                   epochs=1, batch_size=states.shape[0],
                                   verbose=0
                                   # callbacks=[early_stopping],
                                   )
        q_fun = -score.history['loss'][0]

        # q_fun = np.mean(self._trainPolicy(states))

        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"][
            'debug']):
            # print("Actions mean:     ", np.mean(actions, axis=0))
            poli_mean = self._model.getActorNetwork().predict(states, batch_size=states.shape[0])
            print("Policy mean: ", np.mean(poli_mean, axis=0))
            # print("Actions std:  ", np.mean(np.sqrt( (np.square(np.abs(actions - np.mean(actions, axis=0))))/1.0), axis=0) )
            # print("Actions std:  ", np.std((actions - self._q_action()), axis=0) )
            # print("Actions std:  ", np.std((actions), axis=0) )
            # print("Policy std: ", np.mean(self._q_action_std(), axis=0))
            # print("Mean Next State Grad grad: ", np.mean(next_state_grads, axis=0), " std ", np.std(next_state_grads, axis=0))
            # print("Mean ation grad: ", np.mean(action_grads, axis=0), " std ", np.std(action_grads, axis=0))
            print("Actor Loss: ", q_fun)
        self.updateTargetModelActor()
        return q_fun

    def train(self, states, actions, rewards, result_states):
        loss = self.trainCritic(states, actions, rewards, result_states)
        lossActor = self.trainActor(states, actions, rewards, result_states)
        return loss

    def predict(self, state, deterministic_=True, evaluation_=False, p=None, sim_index=None, bootstrapping=False):
        # print ("self._state_bounds shape ", np.array(self._state_bounds).shape)
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=self._settings['float_type'])
        action = self._model.getActorNetwork().predict(state, batch_size=1)
        mean, std = action[:, self._action_length:], action[:, :self._action_length]
        action_ = scale_action(mean, self._action_bounds)
        return action_

    def predict_std(self, state, deterministic_=True, p=1.0):
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=self._settings['float_type'])
        action = self._model.getActorNetwork().predict(state, batch_size=1)
        mean, std = action[:, self._action_length:], action[:, :self._action_length]
        return softplus(std)

    def predict_target(self, state, deterministic_=True, evaluation_=False, p=None, sim_index=None,
                       bootstrapping=False):
        # print ("self._state_bounds shape ", np.array(self._state_bounds).shape)
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=self._settings['float_type'])
        action = self._modelTarget.getActorNetwork().predict(state, batch_size=1)
        mean, std = action[:, self._action_length:], action[:, :self._action_length]
        action_ = scale_action(mean, self._action_bounds)
        return action_

    def q_value(self, state):
        # states = np.zeros((self._batch_size, self._state_length), dtype=self._settings['float_type'])
        # states[0, ...] = state
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=self._settings['float_type'])
        # return scale_reward(self._q_valTarget(), self.getRewardBounds())[0]
        action = self._model.getActorNetwork().predict(state, batch_size=1)
        mean, std = action[:, self._action_length:], action[:, :self._action_length]
        value = (self._model.getCriticNetwork().predict([state, mean], batch_size=1) * action_bound_std(
            self.getRewardBounds())) * (1.0 / (1.0 - self.getSettings()['discount_factor']))

        # value = scale_reward(self._q_func(state), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
        return value
        # return self._q_val()[0]

    def q_values(self, state):
        """
            For returning a vector of q values, state should already be normalized
        """
        state = np.array(state, dtype=self._settings['float_type'])
        # print ("Getting q_values: ", state)
        action = self._model.getActorNetwork().predict(state, batch_size=state.shape[0])
        mean, std = action[:, self._action_length:], action[:, :self._action_length]
        value = self._model.getCriticNetwork().predict([state, mean], batch_size=state.shape[0])
        # print ("q_values: ", value)
        return value

    def q_values2(self, states):
        # states = np.zeros((self._batch_size, self._state_length), dtype=self._settings['float_type'])
        # states[0, ...] = state
        """
        if ( "use_centralized_critic" in self.getSettings()
             and (self.getSettings()["use_centralized_critic"] == True)):
            return np.zeros((states.shape[0], 1))
        """
        states = norm_state(states, self._state_bounds)
        states = np.array(states, dtype=self._settings['float_type'])
        # return scale_reward(self._q_valTarget(), self.getRewardBounds())[0]
        action = self._model.getActorNetwork().predict(states, batch_size=states.shape[0])
        mean, std = action[:, self._action_length:], action[:, :self._action_length]
        value = (self._model.getCriticNetwork().predict([states, mean],
                                                        batch_size=states.shape[0]) * action_bound_std(
            self.getRewardBounds())) * (1.0 / (1.0 - self.getSettings()['discount_factor']))

        # value = scale_reward(self._q_func(state), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
        return value
        # return self._q_val()[0]

    def bellman_error(self, states, actions, rewards, result_states, falls):
        """
            Computes the one step temporal difference.
        """
        action = self._model.getActorNetwork().predict(states, batch_size=states.shape[0])
        mean, std = action[:, self._action_length:], action[:, :self._action_length]
        y_ = self._modelTarget.getCriticNetwork().predict([result_states, actions], batch_size=states.shape[0])
        y_ = y_ - loglikelihood_np(
            actions,
            mean,
            np.exp(std / 2.0),
            self._action_length)
        target_ = rewards + ((self._discount_factor * y_))
        values = self._model.getCriticNetwork().predict([states, mean], batch_size=states.shape[0])
        bellman_error = target_ - values
        return bellman_error
        # return self._bellman_errorTarget()

    def get_actor_regularization(self):
        return self._get_actor_regularization([])

    def get_actor_loss(self, state, action, reward, nextState, advantage):
        return self._get_actor_loss([state])

    def get_critic_regularization(self):
        return self._get_critic_regularization([])

    def get_critic_loss(self, state, action, reward, nextState):
        return self._get_critic_loss([state, action, reward, nextState, 0])

    def saveTo(self, fileName):
        # print(self, "saving model")
        import h5py
        hf = h5py.File(fileName + "_bounds.h5", "w")
        hf.create_dataset('_state_bounds', data=self.getStateBounds())
        hf.create_dataset('_reward_bounds', data=self.getRewardBounds())
        hf.create_dataset('_action_bounds', data=self.getActionBounds())
        # hf.create_dataset('_result_state_bounds', data=self.getResultStateBounds())
        hf.flush()
        hf.close()
        suffix = ".h5"
        ### Save models
        # self._model._actor_train.save(fileName+"_actor_train"+suffix, overwrite=True)
        self._model._actor.save(fileName + "_actor" + suffix, overwrite=True)
        self._model._critic.save(fileName + "_critic" + suffix, overwrite=True)
        if (self._modelTarget is not None):
            self._modelTarget._actor.save(fileName + "_actor_T" + suffix, overwrite=True)
            self._modelTarget._critic.save(fileName + "_critic_T" + suffix, overwrite=True)

        try:
            from keras.utils import plot_model
            ### Save model design as image
            plot_model(self._model._actor, to_file=fileName + "_actor" + '.svg', show_shapes=True)
            plot_model(self._model._critic, to_file=fileName + "_critic" + '.svg', show_shapes=True)
        except Exception as inst:
            ### Maybe the needed libraries are not available
            print("Error saving diagrams for rl models.")
            print(inst)
        # print ("self._model._actor_train: ", self._model._actor_train)

    def loadFrom(self, fileName):
        from keras.models import load_model
        import h5py
        suffix = ".h5"
        print("Loading agent: ", fileName)
        # with K.get_session().graph.as_default() as g:
        self._model._actor = load_model(fileName + "_actor" + suffix)
        self._model._critic = load_model(fileName + "_critic" + suffix)
        if (self._modelTarget is not None):
            self._modelTarget._actor = load_model(fileName + "_actor_T" + suffix)
            self._modelTarget._critic = load_model(fileName + "_critic_T" + suffix)

        # self.compile()
        # self._model._actor_train = load_model(fileName+"_actor_train"+suffix, custom_objects={'loss': pos_y})
        # self._value = K.function([self._model.getStateSymbolicVariable(), K.learning_phase()], [self.__value])
        # self._value_Target = K.function([self._model.getResultStateSymbolicVariable(), K.learning_phase()], [self.__value_Target])
        hf = h5py.File(fileName + "_bounds.h5", 'r')
        self.setStateBounds(np.array(hf.get('_state_bounds')))
        self.setRewardBounds(np.array(hf.get('_reward_bounds')))
        self.setActionBounds(np.array(hf.get('_action_bounds')))
        print("critic self.getStateBounds(): ", self.getStateBounds())
        # self._result_state_bounds = np.array(hf.get('_result_state_bounds'))
        hf.close()



def softplus(x):
    return np.log(1.0 + np.exp(x))