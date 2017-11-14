import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *
from algorithm.AlgorithmInterface import AlgorithmInterface


# For debugging
# theano.config.mode='FAST_COMPILE'
from collections import OrderedDict

class GAN(AlgorithmInterface):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        
        super(GAN,self).__init__( model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)

        self._noise_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype=self.getSettings()['float_type']),
            broadcastable=(False, True))

                
        self._modelTarget = copy.deepcopy(model)
        
            
        # print ("Initial W " + str(self._w_o.get_value()) )
        
        self._learning_rate = self.getSettings()['learning_rate']
        self._discount_factor= self.getSettings()['discount_factor']
        self._rho = self.getSettings()['rho']
        self._rms_epsilon = self.getSettings()['rms_epsilon']
        
        self._weight_update_steps=self.getSettings()['steps_until_target_network_update']
        self._updates=0
        self._decay_weight=self.getSettings()['regularization_weight']
        self._critic_regularization_weight = self.getSettings()["critic_regularization_weight"]
        self._critic_learning_rate = self.getSettings()["critic_learning_rate"]
        
        # self._q_valsA = lasagne.layers.get_output(self._model.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=True)
        # self._q_valsA_drop = lasagne.layers.get_output(self._model.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=False)
        # self._q_valsNextState = lasagne.layers.get_output(self._model.getCriticNetwork(), self._model.getResultStateSymbolicVariable(), deterministic=True)
        # self._q_valsTargetNextState = lasagne.layers.get_output(self._modelTarget.getCriticNetwork(), self._model.getResultStateSymbolicVariable(), deterministic=True)
        # self._q_valsTarget = lasagne.layers.get_output(self._modelTarget.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=True)
        # self._q_valsTarget_drop = lasagne.layers.get_output(self._modelTarget.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=False)
        
        if ("train_gan_with_gaussian_noise" in self.getSettings() and (self.getSettings()["train_gan_with_gaussian_noise"])):
            inputs_1 = {
                self._model.getStateSymbolicVariable(): self._model.getStates(),
                self._model._Noise: self._noise_shared
            }
            self._generator = lasagne.layers.get_output(self._model.getActorNetwork(), inputs_1, deterministic=True)
        else:
            self._generator = lasagne.layers.get_output(self._model.getActorNetwork(), self._model.getStateSymbolicVariable(), deterministic=False)
        # self._q_valsActTarget = lasagne.layers.get_output(self._modelTarget.getActorNetwork(), self._model.getResultStateSymbolicVariable(), deterministic=True)
        # self._q_valsActA_drop = lasagne.layers.get_output(self._model.getActorNetwork(), self._model.getStateSymbolicVariable(), deterministic=False)
        
        
        self._discriminator = lasagne.layers.get_output(self._model.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=True)
        self._discriminator_drop = lasagne.layers.get_output(self._model.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=False)
        """
        inputs_2 = {
            self._modelTarget.getStateSymbolicVariable(): self._model.getResultStates(),
            self._modelTarget.getActionSymbolicVariable(): self._model.getActions()
        }
        """
        
        
        self._diff = self._model.getRewardSymbolicVariable() - self._discriminator_drop
        loss = T.pow(self._diff, 2)
        self._loss = T.mean(loss)
        
        
        self._diff_g = self._model.getStateSymbolicVariable() - self._generator
        loss_g = T.pow(self._diff_g, 2)
        self._loss_g = T.mean(loss_g)
    
        # assert len(lasagne.layers.helper.get_all_params(self._l_outA)) == 16
        # Need to remove the action layers from these params
        self._params = lasagne.layers.helper.get_all_params(self._model.getCriticNetwork()) 
        print ("******Number of Layers is: " + str(len(lasagne.layers.helper.get_all_params(self._model.getCriticNetwork()))))
        print ("******Number of Action Layers is: " + str(len(lasagne.layers.helper.get_all_params(self._model.getActorNetwork()))))
        self._actionParams = lasagne.layers.helper.get_all_params(self._model.getActorNetwork())
        self._givens_ = {
            self._model.getStateSymbolicVariable(): self._model.getStates(),
            self._model.getRewardSymbolicVariable(): self._model.getRewards(),
        }
        
        self._critic_regularization = (self._critic_regularization_weight * 
                                       lasagne.regularization.regularize_network_params(
                                            self._model.getCriticNetwork(), lasagne.regularization.l2))
        
        ## MSE update
        self._value_grad = T.grad(self._loss + self._critic_regularization
                                                     , self._params)
        print ("Optimizing Value Function with ", self.getSettings()['optimizer'], " method")
        self._updates_ = lasagne.updates.adam(self._value_grad
                    , self._params, self._critic_learning_rate , beta1=0.9, beta2=0.9, epsilon=self._rms_epsilon)
        
        if ("train_gan_with_gaussian_noise" in settings_ and (settings_["train_gan_with_gaussian_noise"])):
            self._actGivens = {
                self._model.getStateSymbolicVariable(): self._model.getStates(),
                self._model._Noise: self._noise_shared,
            }
        else:
            self._actGivens = {
                self._model.getStateSymbolicVariable(): self._model.getStates(),
            }
        
        self._actor_regularization = (self._regularization_weight * 
                                       lasagne.regularization.regularize_network_params(
                                            self._model.getActorNetwork(), lasagne.regularization.l2))
        ## MSE update
        self._gen_grad = T.grad(self._loss_g + self._actor_regularization
                                                     , self._actionParams)
        print ("Optimizing Value Function with ", self.getSettings()['optimizer'], " method")
        self._updates_generator = lasagne.updates.adam(self._gen_grad
                    , self._actionParams, self._learning_rate , beta1=0.9, beta2=0.9, epsilon=self._rms_epsilon)
        
        ## Some cool stuff to backprop action gradients
        
        self._state_grad = T.matrix("Action_Grad")
        self._state_grad.tag.test_value = np.zeros((self._batch_size,self._state_length), dtype=np.dtype(self.getSettings()['float_type']))
        
        self._state_grad_shared = theano.shared(
            np.zeros((self._batch_size, self._state_length),
                      dtype=self.getSettings()['float_type']))
        
        ### Maximize wrt q function
        
        self._state_mean_grads = T.grad(cost=None, wrt=self._actionParams,
                                                    known_grads={self._generator: self._state_grad_shared}),
        print ("Action grads: ", self._state_mean_grads[0])
        ## When passing in gradients it needs to be a proper list of gradient expressions
        self._state_mean_grads = list(self._state_mean_grads[0])
        # print ("isinstance(self._action_mean_grads, list): ", isinstance(self._action_mean_grads, list))
        # print ("Action grads: ", self._action_mean_grads)
        self._generatorGRADUpdates = lasagne.updates.adam(self._state_mean_grads, self._actionParams, 
                    self._learning_rate,  beta1=0.9, beta2=0.9, epsilon=self._rms_epsilon)
        
        self._givens_grad = {
                self._model.getStateSymbolicVariable(): self._model.getStates(),
            }
        
        GAN.compile(self)
        
    def compile(self):
        
        self._train = theano.function([], [self._loss, self._discriminator], updates=self._updates_, givens=self._givens_)
        
        # self._trainActor = theano.function([], [actLoss, self._q_valsActA], updates=actionUpdates, givens=actGivens)
        # self._trainActor = theano.function([], [self._q_func], updates=self._actionUpdates, givens=self._actGivens)
        self._trainGenerator  = theano.function([], [], updates=self._generatorGRADUpdates, givens=self._actGivens)
        self._trainGenerator_MSE = theano.function([], [], updates=self._updates_generator, givens=self._actGivens)
        self._q_val = theano.function([], self._discriminator,
                                       givens={self._model.getStateSymbolicVariable(): self._model.getStates()
                                               })
        
        #self._q_val_Target = theano.function([], self._q_valsB_, givens=self._givens_grad)
        if ("train_gan_with_gaussian_noise" in self.getSettings() and (self.getSettings()["train_gan_with_gaussian_noise"])):
            self._generate = theano.function([], self._generator,
                   givens={self._model.getStateSymbolicVariable(): self._model.getStates(),
                           self._model._Noise: self._noise_shared})
        else:
            self._generate = theano.function([], self._generator,
                                       givens={self._model.getStateSymbolicVariable(): self._model.getStates()})
        """
        inputs_ = [
                   self._model.getStateSymbolicVariable(), 
                   self._model.getRewardSymbolicVariable(), 
                   # ResultState
                   ]
        self._bellman_error = theano.function(inputs=inputs_, outputs=self._diff, allow_input_downcast=True)
        """
        # self._diffs = theano.function(input=[State])
        self._bellman_error2 = theano.function(inputs=[], outputs=self._diff, allow_input_downcast=True, givens=self._givens_)
        
        # self._get_action_grad = theano.function([], outputs=lasagne.updates.get_or_compute_grads(T.mean(self._discriminator), [self._model._actionInputVar] + self._params), allow_input_downcast=True, givens=self._givens_grad)
        
        self._get_state_grad = theano.function([], outputs=lasagne.updates.get_or_compute_grads(T.mean(self._discriminator), [self._model._stateInputVar] + self._params), allow_input_downcast=True, givens=self._givens_grad)
        
        
    def getStateGrads(self, states, actions=None, alreadyNormed=False):
        """
            The states should be normalized
        """
        # self.setData(states, actions, rewards, result_states)
        if ( alreadyNormed == False):
            states = norm_state(states, self._state_bounds)
        states = np.array(states, dtype=theano.config.floatX)
        self._model.setStates(states)
        
        return self._get_state_grad()
    

    def getNetworkParameters(self):
        params = []
        params.append(lasagne.layers.helper.get_all_param_values(self._model.getCriticNetwork()))
        params.append(lasagne.layers.helper.get_all_param_values(self._model.getActorNetwork()))
        params.append(lasagne.layers.helper.get_all_param_values(self._modelTarget.getCriticNetwork()))
        params.append(lasagne.layers.helper.get_all_param_values(self._modelTarget.getActorNetwork()))
        return params
        
    def setNetworkParameters(self, params):
        lasagne.layers.helper.set_all_param_values(self._model.getCriticNetwork(), params[0])
        lasagne.layers.helper.set_all_param_values(self._model.getActorNetwork(), params[1])
        lasagne.layers.helper.set_all_param_values(self._modelTarget.getCriticNetwork(), params[2])
        lasagne.layers.helper.set_all_param_values(self._modelTarget.getActorNetwork(), params[3])
        
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
        
    def trainCritic(self, states, actions, rewards, result_states, falls):
        
        self.setData(states, actions, rewards, result_states, falls)
        self._noise_shared.set_value(np.random.normal(0,0.5, size=(states.shape[0],1)))
        self._updates += 1
        ## Compute actions for TargetNet
        generated_samples = self._generate()
        tmp_states = copy.deepcopy(states)
        tmp_rewards = copy.deepcopy(rewards)
        ## replace half of the samples with generated ones...
        for i in range(int(states.shape[0]/2)):
            
            tmp_states[i] = generated_samples[i]
            tmp_rewards[i] = [0] 

        # print("Rewards: ", tmp_rewards)            
        self.setData(tmp_states, actions, tmp_rewards, result_states, falls)
        
        loss, _ = self._train()
        return loss
        
    def trainActor(self, states, actions, rewards, result_states, falls, advantage, exp_actions, forwardDynamicsModel=None):
        self.setData(states, actions, rewards, result_states, falls)
        
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['debug']):
            print("values: ", np.mean(self._q_val()* (1.0 / (1.0- self.getSettings()['discount_factor']))), " std: ", np.std(self._q_val()* (1.0 / (1.0- self.getSettings()['discount_factor']))) )
            print("Rewards: ", np.mean(rewards), " std: ", np.std(rewards), " shape: ", np.array(rewards).shape)
            
        self._noise_shared.set_value(np.random.normal(0,0.5, size=(states.shape[0],1)))
        ## Add MSE term
        self._trainGenerator_MSE()
        # print("Policy mean: ", np.mean(self._q_action(), axis=0))
        loss = 0
        # loss = self._trainActor()
        # print("******** Not learning actor right now *****")
        # return loss
        generated_samples = self.predict_batch(states)
        # print ("actions shape:", actions.shape)
        # next_states = forwardDynamicsModel.predict_batch(states, actions)
        # print ("next_states shape: ", next_states.shape)
        state_grads = self.getStateGrads(generated_samples, actions, alreadyNormed=True)[0] * 1.0
        # print ("next_state_grads shape: ", next_state_grads.shape)
        # action_grads = forwardDynamicsModel.getGrads(states, actions, next_states, v_grad=next_state_grads, alreadyNormed=True)[0] * 1.0
        # print ( "action_grads shape: ", action_grads.shape)
        discriminator_value = self._bellman_error2() 
        """
            From DEEP REINFORCEMENT LEARNING IN PARAMETERIZED ACTION SPACE
            Hausknecht, Matthew and Stone, Peter
            
            actions.shape == action_grads.shape
        """
                    
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['debug']):
            # print("Actions mean:     ", np.mean(actions, axis=0))
            print("Policy mean: ", np.mean(self._generate(), axis=0))
            # print("Actions std:  ", np.mean(np.sqrt( (np.square(np.abs(actions - np.mean(actions, axis=0))))/1.0), axis=0) )
            # print("Actions std:  ", np.std((actions - self._q_action()), axis=0) )
            # print("Actions std:  ", np.std((actions), axis=0) )
            # print("Policy std: ", np.mean(self._q_action_std(), axis=0))
            # print("Mean Next State Grad grad: ", np.mean(next_state_grads, axis=0), " std ", np.std(next_state_grads, axis=0))
            print("Mean action grad: ", np.mean(state_grads, axis=0), " std ", np.std(action_grads, axis=0))
        
        ## Set data for gradient
        self._model.setStates(states)
        self._modelTarget.setStates(states)
        ## Why the -1.0??
        ## Because the SGD method is always performing MINIMIZATION!!
        self._state_grad_shared.set_value(-1.0*state_grads)
        self._trainGenerator()
        
        return np.mean(discriminator_value)
        
    def train(self, states, actions, rewards, result_states, falls, advantage_, exp_actions__):
        loss = self.trainCritic(states, actions, rewards, result_states, falls)
        lossActor = self.trainActor(states, actions, rewards, result_states, falls, advantage_, exp_actions__)
        return (loss, lossActor)
    
    def predict(self, state, deterministic_=True):
        # states = np.zeros((self._batch_size, self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        """
        if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
            pass
        else:
        """
        # print ("Agent state bounds: ", self._state_bounds)
        state = norm_state(state, self._state_bounds)
        # print ("Agent normalized state: ", state)
        state = np.array(state, dtype=theano.config.floatX)
        self._model.setStates(state)
        self._noise_shared.set_value(np.random.normal(0,0.5, size=(1,1)))
        # if deterministic_:
        # action_ = self._generate()[0]
        action_ = scale_state(self._generate()[0], self._state_bounds)
        return action_
    
    def predict_batch(self, states, deterministic_=True):
        """
            These input and output do not need to be normalized/scalled
        """
        # state = norm_state(state, self._state_bounds)
        states = np.array(states, dtype=theano.config.floatX)
        self._model.setStates(states)
        self._noise_shared.set_value(np.random.normal(0,0.5, size=(states.shape[0],1)))
        actions_ = self._generate()
        return actions_
    
    def q_value(self, state):
        """
            For returning a vector of q values, state should NOT be normalized
        """
        # states = np.zeros((self._batch_size, self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        """
        if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
            pass
        else:
        """
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=theano.config.floatX)
        self._model.setStates(state)
        self._modelTarget.setStates(state)
        action = self._q_action()
        self._model.setActions(action)
        self._modelTarget.setActions(action)
        return scale_reward(self._q_val(), self.getRewardBounds())[0] * (1.0 / (1.0- self.getSettings()['discount_factor']))
        # return self._q_valTarget()[0]
        # return self._q_val()[0]
    
    def q_values(self, state):
        """
            For returning a vector of q values, state should already be normalized
        """
        """
        if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
            pass
        else:
        """
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=theano.config.floatX)
        self._model.setStates(state)
        self._modelTarget.setStates(state)
        action = self._q_action()
        self._model.setActions(action)
        self._modelTarget.setActions(action)
        return scale_reward(self._q_val(), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
        # return self._q_valTarget()
        # return self._q_val()