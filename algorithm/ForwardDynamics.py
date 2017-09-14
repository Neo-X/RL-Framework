import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *
from model.LearningUtil import loglikelihood, kl, entropy, flatgrad, zipsame, get_params_flat, setFromFlat, likelihood, loglikelihoodMEAN

# For debugging
# theano.config.mode='FAST_COMPILE'
from algorithm.AlgorithmInterface import AlgorithmInterface

class ForwardDynamics(AlgorithmInterface):
    
    def __init__(self, model, state_length, action_length, state_bounds, action_bounds, settings_):

        super(ForwardDynamics,self).__init__(model, state_length, action_length, state_bounds, action_bounds, 0, settings_)
        self._model = model
        batch_size=32
        self._learning_rate = self.getSettings()["fd_learning_rate"]
        # data types for model
        # create a small convolutional neural network
        
        inputs_ = {
            self._model.getStateSymbolicVariable(): self._model.getStates(),
            self._model.getActionSymbolicVariable(): self._model.getActions(),
        }
        self._forward = lasagne.layers.get_output(self._model.getForwardDynamicsNetwork(), inputs_, deterministic=True)[:,:self._state_length]
        ## This drops to ~ 0 so fast.
        self._forward_std = (lasagne.layers.get_output(self._model.getForwardDynamicsNetwork(), inputs_, deterministic=True)[:,self._state_length:] * self.getSettings()['exploration_rate'] )+ 1e-2
        self._forward_std_drop = (lasagne.layers.get_output(self._model.getForwardDynamicsNetwork(), inputs_, deterministic=True)[:,self._state_length:] * self.getSettings()['exploration_rate']) + 1e-2
        self._forward_drop = lasagne.layers.get_output(self._model.getForwardDynamicsNetwork(), inputs_, deterministic=False)[:,:self._state_length]
        self._reward = lasagne.layers.get_output(self._model.getRewardNetwork(), inputs_, deterministic=True)
        self._reward_drop = lasagne.layers.get_output(self._model.getRewardNetwork(), inputs_, deterministic=False)
        
        l2_loss = True
        
        if ('use_stochastic_forward_dynamics' in self.getSettings() and 
            (self.getSettings()['use_stochastic_forward_dynamics'])):
            
            self._diff = loglikelihood(self._model.getResultStateSymbolicVariable(), self._forward_drop, self._forward_std_drop, self._state_length)
            self._policy_entropy = 0.5 * T.mean(T.log(2 * np.pi * self._forward_std ) + 1 )
            self._loss = -1.0 * (T.mean(self._diff) + (self._policy_entropy * 1e-3))
            # self._loss = -1.0 * (T.mean(self._diff) ) 
            
            ### Not used dropout stuff
            self._diff_NoDrop = loglikelihood(self._model.getResultStateSymbolicVariable(), self._forward, self._forward_std, self._state_length)
            # self._loss_NoDrop = -1.0 * (T.mean(self._diff_NoDrop) + (self._policy_entropy * 1e-4))
            self._loss_NoDrop = -1.0 * (T.mean(self._diff_NoDrop) )
        else:
            # self._target = (Reward + self._discount_factor * self._q_valsB)
            self._diff = self._model.getResultStateSymbolicVariable() - self._forward_drop
            ## mean across each sate
            if (l2_loss):
                self._loss = T.mean(T.pow(self._diff, 2),axis=1)
            else:
                self._loss = T.mean(T.abs_(self._diff),axis=1)
            ## mean over batch
            self._loss = T.mean(self._loss)
            ## Another version that does not have dropout
            self._diff_NoDrop = self._model.getResultStateSymbolicVariable() - self._forward
            ## mean across each sate
            if (l2_loss):
                self._loss_NoDrop = T.mean(T.pow(self._diff_NoDrop, 2),axis=1)
            else:
                self._loss_NoDrop = T.mean(T.abs_(self._diff_NoDrop),axis=1)
            ## mean over batch
            self._loss_NoDrop = T.mean(self._loss_NoDrop)
        
        
        self._reward_diff = self._model.getRewardSymbolicVariable() - self._reward_drop
        self._reward_loss = T.mean(T.pow(self._reward_diff, 2),axis=1)
        self._reward_loss = T.mean(self._reward_loss)
        
        self._reward_diff_NoDrop = self._model.getRewardSymbolicVariable() - self._reward
        self._reward_loss_NoDrop = T.mean(T.pow(self._reward_diff_NoDrop, 2),axis=1)
        self._reward_loss_NoDrop = T.mean(self._reward_loss_NoDrop)
        
        self._params = lasagne.layers.helper.get_all_params(self._model.getForwardDynamicsNetwork())
        self._reward_params = lasagne.layers.helper.get_all_params(self._model.getRewardNetwork())
        self._givens_ = {
            self._model.getStateSymbolicVariable() : self._model.getStates(),
            self._model.getResultStateSymbolicVariable() : self._model.getResultStates(),
            self._model.getActionSymbolicVariable(): self._model.getActions(),
        }
        
        self._reward_givens_ = {
            self._model.getStateSymbolicVariable() : self._model.getStates(),
            # self._model.getResultStateSymbolicVariable() : self._model.getResultStates(),
            self._model.getActionSymbolicVariable(): self._model.getActions(),
            self._model.getRewardSymbolicVariable() : self._model.getRewards(),
        }

        # SGD update
        if (self.getSettings()['optimizer'] == 'rmsprop'):
            print ("Optimizing Forward Dynamics with ", self.getSettings()['optimizer'], " method")
            self._updates_ = lasagne.updates.rmsprop(self._loss + (self._regularization_weight * lasagne.regularization.regularize_network_params(
                    self._model.getForwardDynamicsNetwork(), lasagne.regularization.l2)), self._params, self._learning_rate, self._rho, self._rms_epsilon)
            self._reward_updates_ = lasagne.updates.rmsprop(self._reward_loss + (self._regularization_weight * lasagne.regularization.regularize_network_params(
                    self._model.getRewardNetwork(), lasagne.regularization.l2)), self._reward_params, self._learning_rate, self._rho, self._rms_epsilon)
        elif (self.getSettings()['optimizer'] == 'momentum'):
            print ("Optimizing Forward Dynamics with ", self.getSettings()['optimizer'], " method")
            self._updates_ = lasagne.updates.momentum(self._loss + (self._regularization_weight * lasagne.regularization.regularize_network_params(
                self._model.getForwardDynamicsNetwork(), lasagne.regularization.l2)), self._params, self._learning_rate, momentum=self._rho)
            self._reward_updates_ = lasagne.updates.momentum(self._reward_loss + (self._regularization_weight * lasagne.regularization.regularize_network_params(
                self._model.getRewardNetwork(), lasagne.regularization.l2)), self._reward_params, self._learning_rate, momentum=self._rho)
        elif ( self.getSettings()['optimizer'] == 'adam'):
            print ("Optimizing Forward Dynamics with ", self.getSettings()['optimizer'], " method")
            self._updates_ = lasagne.updates.adam(self._loss + (self._regularization_weight * lasagne.regularization.regularize_network_params(
                self._model.getForwardDynamicsNetwork(), lasagne.regularization.l2)), self._params, self._learning_rate, beta1=0.9, beta2=0.999, epsilon=self._rms_epsilon)
            self._reward_updates_ = lasagne.updates.adam(self._reward_loss + (self._regularization_weight * lasagne.regularization.regularize_network_params(
                self._model.getRewardNetwork(), lasagne.regularization.l2)), self._reward_params, self._learning_rate, beta1=0.9, beta2=0.999, epsilon=self._rms_epsilon)
        elif ( self.getSettings()['optimizer'] == 'adagrad'):
            print ("Optimizing Forward Dynamics with ", self.getSettings()['optimizer'], " method")
            self._updates_ = lasagne.updates.adagrad(self._loss + (self._regularization_weight * lasagne.regularization.regularize_network_params(
                self._model.getForwardDynamicsNetwork(), lasagne.regularization.l2)), self._params, self._learning_rate, epsilon=self._rms_epsilon)
            self._reward_updates_ = lasagne.updates.adagrad(self._reward_loss + (self._regularization_weight * lasagne.regularization.regularize_network_params(
                self._model.getRewardNetwork(), lasagne.regularization.l2)), self._reward_params, self._learning_rate, epsilon=self._rms_epsilon)
        else:
            print ("Unknown optimization method: ", self.getSettings()['optimizer'])
        self._updates_ = lasagne.updates.rmsprop(self._loss + (self._regularization_weight * lasagne.regularization.regularize_network_params(
                self._model.getForwardDynamicsNetwork(), lasagne.regularization.l2)), self._params, self._learning_rate, self._rho,
                                            self._rms_epsilon)
        
        self._reward_updates_ = lasagne.updates.rmsprop(self._reward_loss + (self._regularization_weight * lasagne.regularization.regularize_network_params(
                self._model.getRewardNetwork(), lasagne.regularization.l2)), self._reward_params, self._learning_rate, self._rho,
                                            self._rms_epsilon)
     
        self._train = theano.function([], [self._loss], updates=self._updates_, givens=self._givens_)
        self._train_reward = theano.function([], [self._reward_loss], updates=self._reward_updates_, givens=self._reward_givens_)
        self._forwardDynamics = theano.function([], self._forward,
                                       givens={self._model.getStateSymbolicVariable() : self._model.getStates(),
                                                self._model.getActionSymbolicVariable(): self._model.getActions()})
        self._forwardDynamics_std = theano.function([], self._forward_std,
                                       givens={self._model.getStateSymbolicVariable() : self._model.getStates(),
                                                self._model.getActionSymbolicVariable(): self._model.getActions()})
        self._predict_reward = theano.function([], self._reward,
                                       givens={self._model.getStateSymbolicVariable() : self._model.getStates(),
                                                self._model.getActionSymbolicVariable(): self._model.getActions()})
        
        self._bellman_error = theano.function(inputs=[], outputs=self._diff, allow_input_downcast=True, givens=self._givens_)
        self._reward_error = theano.function(inputs=[], outputs=self._reward_diff, allow_input_downcast=True, givens=self._reward_givens_)
        # self._diffs = theano.function(input=[State])
        # self._get_grad = theano.function([], outputs=lasagne.updates.get_or_compute_grads(self._loss_NoDrop, [lasagne.layers.get_all_layers(self._model.getForwardDynamicsNetwork())[self.getSettings()['action_input_layer_index']].input_var] + self._params), allow_input_downcast=True, givens=self._givens_)
        self._get_grad = theano.function([], outputs=lasagne.updates.get_or_compute_grads(self._loss_NoDrop, [self._model._actionInputVar] + self._params), allow_input_downcast=True, givens=self._givens_)
        # self._get_grad_reward = theano.function([], outputs=lasagne.updates.get_or_compute_grads((self._reward_loss_NoDrop), [lasagne.layers.get_all_layers(self._model.getRewardNetwork())[0].input_var] + self._reward_params), allow_input_downcast=True,
        self._get_grad_reward = theano.function([], outputs=lasagne.updates.get_or_compute_grads(T.mean(self._reward), [self._model._actionInputVar] + self._reward_params), allow_input_downcast=True, 
                                                givens={
            self._model.getStateSymbolicVariable() : self._model.getStates(),
            # self._model.getResultStateSymbolicVariable() : self._model.getResultStates(),
            self._model.getActionSymbolicVariable(): self._model.getActions(),
            # self._model.getRewardSymbolicVariable() : self._model.getRewards(),
        })

    def getNetworkParameters(self):
        params = []
        params.append(lasagne.layers.helper.get_all_param_values(self._model.getForwardDynamicsNetwork()))
        params.append(lasagne.layers.helper.get_all_param_values(self._model.getRewardNetwork()))
        return params
    
    def setNetworkParameters(self, params):
        lasagne.layers.helper.set_all_param_values(self._model.getForwardDynamicsNetwork(), params[0])
        lasagne.layers.helper.set_all_param_values(self._model.getRewardNetwork(), params[1])
        
    def setData(self, states, actions, result_states, rewards=[]):
        self._model.setStates(states)
        self._model.setResultStates(result_states)
        self._model.setActions(actions)
        if (rewards != []):
            self._model.setRewards(rewards)
        
    def getGrads(self, states, actions, result_states):
        states = np.array(norm_state(states, self._state_bounds), dtype=self.getSettings()['float_type'])
        actions = np.array(norm_action(actions, self._action_bounds), dtype=self.getSettings()['float_type'])
        result_states = np.array(norm_state(result_states, self._state_bounds), dtype=self.getSettings()['float_type'])
        # result_states = np.array(result_states, dtype=self.getSettings()['float_type'])
        self.setData(states, actions, result_states)
        return self._get_grad()
    
    def getRewardGrads(self, states, actions, rewards):
        # states = np.array(states, dtype=self.getSettings()['float_type'])
        # actions = np.array(actions, dtype=self.getSettings()['float_type'])
        states = np.array(norm_state(states, self._state_bounds), dtype=self.getSettings()['float_type'])
        actions = np.array(norm_action(actions, self._action_bounds), dtype=self.getSettings()['float_type'])
        rewards = np.array(norm_state(rewards, self._reward_bounds), dtype=self.getSettings()['float_type'])
        self.setData(states, actions, None, rewards)
        return self._get_grad_reward()
                
    def train(self, states, actions, result_states, rewards):
        # rewards = rewards * (1.0/(1.0-self.getSettings()['discount_factor'])) # scale rewards
        self.setData(states, actions, result_states, rewards)
        # print ("Performing Critic trainning update")
        #if (( self._updates % self._weight_update_steps) == 0):
        #    self.updateTargetModel()
        self._updates += 1
        # all_paramsActA = lasagne.layers.helper.get_all_param_values(self._l_outActA)
        loss = self._train()
        if ( self.getSettings()['train_reward_predictor']):
            # print ("self._reward_bounds: ", self._reward_bounds)
            # print( "Rewards, predicted_reward, difference: ", np.concatenate((rewards, self._predict_reward(), rewards - self._predict_reward()), axis=1))
            lossReward = self._train_reward()
        # This undoes the Actor parameter updates as a result of the Critic update.
        # print (diff_)
        return loss
    
    def predict(self, state, action):
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        # print ("fd state: ", state)
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        self._model.setStates(state)
        self._model.setActions(action)
        # print ("State bounds: ", self._state_bounds)
        # print ("fd output: ", self._forwardDynamics()[0])
        state_ = scale_state(self._forwardDynamics()[0], self._state_bounds)
        return state_
    
    def predict_std(self, state, action):
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        # print ("fd state: ", state)
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        self._model.setStates(state)
        self._model.setActions(action)
        state_ = scale_state(self._forwardDynamics_std()[0], self._state_bounds)
        return state_
    
    def predict_reward(self, state, action):
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        self._model.setStates(state)
        self._model.setActions(action)
        predicted_reward = self._predict_reward()[0]
        reward_ = scale_state(predicted_reward, self._reward_bounds)
        # print ("reward, predicted reward: ", reward_, predicted_reward)
        return reward_
    
    def predict_batch(self, states, actions):
        ## These input should already be normalized.
        self._model.setStates(states)
        self._model.setActions(actions)
        return self._forwardDynamics()

    def bellman_error(self, states, actions, result_states, rewards):
        self.setData(states, actions, result_states, rewards)
        return self._bellman_error()
    
    def reward_error(self, states, actions, result_states, rewards):
        # rewards = rewards * (1.0/(1.0-self.getSettings()['discount_factor'])) # scale rewards
        self.setData(states, actions, result_states, rewards)
        return self._reward_error()
