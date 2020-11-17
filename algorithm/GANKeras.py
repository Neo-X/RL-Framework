import numpy as np
import sys
from model.ModelUtil import *
from algorithm.AlgorithmInterface import AlgorithmInterface

from keras.optimizers import SGD
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K
import keras
from keras.models import Sequential, Model

# For debugging
# theano.config.mode='FAST_COMPILE'
from collections import OrderedDict
from util.ExperienceMemory import ExperienceMemory

class GANKeras(AlgorithmInterface):
    """
        0 is a generated sample
        1 is a true sample
        maximize D while minimizing G
    """
    
    def __init__(self,  model, state_length, action_length, state_bounds, action_bounds, settings_, reward_bounds=0, print_info=False):

        print("Building GAN Model")
        super(GANKeras,self).__init__(model, state_length, action_length, state_bounds, action_bounds, reward_bounds, settings_)
        self._noise_mean = 0.0
        self._noise_std = 1.0

        # if settings['action_space_continuous']:
        if ( 'size_of_result_state' in self.getSettings()):
            self._experience = ExperienceMemory(state_length, action_length, 
                                                self.getSettings()['experience_length'],
                                                continuous_actions=True, 
                                                settings=self.getSettings(), 
                                                result_state_length=self.getSettings()['size_of_result_state'])
        else:
            self._experience = ExperienceMemory(state_length, action_length, 
                                                self.getSettings()['experience_length'][0],
                                                continuous_actions=True, 
                                                settings=self.getSettings())
            
        self._experience.setStateBounds(copy.deepcopy(self.getStateBounds()))
        self._experience.setRewardBounds(copy.deepcopy(self.getRewardBounds()))
        self._experience.setActionBounds(copy.deepcopy(self.getActionBounds()))
                
        # self._modelTarget = copy.deepcopy(model)
        self._modelTarget = type(self._model)(state_length, action_length, state_bounds, action_bounds, reward_bounds, settings_, print_info=print_info)
            
        # print ("Initial W " + str(self._w_o.get_value()) )
        
        self._learning_rate = self.getSettings()["fd_learning_rate"]
        self._regularization_weight = 1e-5
        self._rho = self.getSettings()['rho']
        self._rms_epsilon = self.getSettings()['rms_epsilon']
        
        self._weight_update_steps=self.getSettings()['steps_until_target_network_update']
        self._updates=0
        self._decay_weight=self.getSettings()['regularization_weight']
        self._critic_regularization_weight = self.getSettings()["critic_regularization_weight"]
        
        # self._q_valsA = lasagne.layers.get_output(self._model.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=True)
        # self._q_valsA_drop = lasagne.layers.get_output(self._model.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=False)
        # self._q_valsNextState = lasagne.layers.get_output(self._model.getCriticNetwork(), self._model.getResultStateSymbolicVariable(), deterministic=True)
        # self._q_valsTargetNextState = lasagne.layers.get_output(self._modelTarget.getCriticNetwork(), self._model.getResultStateSymbolicVariable(), deterministic=True)
        # self._q_valsTarget = lasagne.layers.get_output(self._modelTarget.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=True)
        # self._q_valsTarget_drop = lasagne.layers.get_output(self._modelTarget.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=False)
        
        GANKeras.compile(self)
        
    def compile(self):
        
        self._discriminate = K.function([self._model.getStateSymbolicVariable(), 
                        self._model.getActionSymbolicVariable(),
                        self._model.getResultStateSymbolicVariable()],
                        [self._model.getCriticNetwork()])
        self._model._critic = Model(inputs=[self._model.getStateSymbolicVariable(), 
                            self._model.getActionSymbolicVariable(),
                            self._model.getResultStateSymbolicVariable()], 
                             outputs=[self._model._critic])
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), 
                                    beta_1=np.float32(0.9), beta_2=np.float32(0.999), 
                                    epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0000001),
                                    amsgrad=False)
        print ("Clipping: ", sgd.decay)
        print("sgd, critic: ", sgd)
        self._model.getCriticNetwork().compile(loss=['binary_crossentropy'], optimizer=sgd)
        print("Discriminator Net summary: ",  self._model.getCriticNetwork().summary())
        
        """
        self._model._reward_net = Model(input=[self._model.getStateSymbolicVariable(), 
                            self._model.getActionSymbolicVariable()],
                            output=self._model._reward_net)
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), 
                                    beta_1=np.float32(0.9), beta_2=np.float32(0.999), 
                                    epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0000001),
                                    amsgrad=False)
        print ("Clipping: ", sgd.decay)
        print("sgd, critic: ", sgd)
        self._model.getRewardNetwork().compile(loss='mse', optimizer=sgd)
        print("Reward Net summary: ",  self._model.getRewardNetwork().summary())
        """
        # For the combined model we will only train the generator
        self._model.getCriticNetwork().trainable = False
        
        def neg_y(true_y, pred_y):
            return -pred_y
        
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), 
                                    beta_1=np.float32(0.9), beta_2=np.float32(0.999), 
                                    epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0000001),
                                    amsgrad=False)
        
        self._model._forward_dynamics_net = Model(input=[self._model.getStateSymbolicVariable(), 
                        self._model.getActionSymbolicVariable(),
                        self._model._Noise], 
                        output=self._model._forward_dynamics_net)
        self._model._forward_dynamics_net.compile(loss=['mse'], optimizer=sgd)
        
        
        self.__generate = self._model.getForwardDynamicsNetwork()(
                                [self._model.getStateSymbolicVariable(), 
                                self._model.getActionSymbolicVariable(),
                                self._model._Noise]
                                )
        
        self._genloss = (self._model.getCriticNetwork()(
                            [self._model.getStateSymbolicVariable(), 
                            self._model.getActionSymbolicVariable(),
                            self.__generate]))
        
        self._combined = Model(input=[self._model.getStateSymbolicVariable(), 
                                self._model.getActionSymbolicVariable(),
                                self._model._Noise], 
                                output=self._genloss)
        
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), 
                                    beta_1=np.float32(0.9), beta_2=np.float32(0.999), 
                                    epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0000001),
                                    amsgrad=False)
        print ("Clipping: ", sgd.decay)
        print("sgd, critic: ", sgd)
        self._combined.compile(loss=['binary_crossentropy'], optimizer=sgd)
        print("FD Net summary: ",  self._combined.summary())
        
        self._generate = K.function([self._model.getStateSymbolicVariable(), 
                                self._model.getActionSymbolicVariable(),
                                self._model._Noise
                                , K.learning_phase()],
                                [self.__generate])
        
        
        self.__bellman_error = K.mean(K.abs(self.__generate - self._model.getResultStateSymbolicVariable()))
        self._bellman_error = K.function([self._model.getStateSymbolicVariable(), 
                                self._model.getActionSymbolicVariable(),
                                self._model._Noise,
                                self._model.getResultStateSymbolicVariable()],
                                [self.__bellman_error])
        
        
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
    
    def getResultStateGrads(self, result_states, actions=None, alreadyNormed=False):
        """
            The states should be normalized
        """
        # self.setData(states, actions, rewards, result_states)
        if ( alreadyNormed == False):
            result_states = norm_state(result_states, self._state_bounds)
        result_states = np.array(result_states, dtype=theano.config.floatX)
        self._model.setResultStates(result_states)
        
        return self._get_result_state_grad()
    
    def setGradTarget(self, grad):
        self._result_state_grad_shared.set_value(grad)
        
    def getGrads(self, states, actions, result_states, v_grad=None, alreadyNormed=False):
        if ( alreadyNormed == False ):
            states = np.array(norm_state(states, self._state_bounds), dtype=self.getSettings()['float_type'])
            actions = np.array(norm_action(actions, self._action_bounds), dtype=self.getSettings()['float_type'])
            result_states = np.array(norm_state(result_states, self._state_bounds), dtype=self.getSettings()['float_type'])
        # result_states = np.array(result_states, dtype=self.getSettings()['float_type'])
        self.setData(states, actions, result_states)
        ### I think this helps?
        noise = np.zeros((states.shape[0],1))
        self._noise_shared.set_value(noise)
        # if (v_grad != None):
        self.setGradTarget(v_grad)
        return self._get_action_grad()
    
    
    def getRewardGrads(self, states, actions, alreadyNormed=False):
        # states = np.array(states, dtype=self.getSettings()['float_type'])
        # actions = np.array(actions, dtype=self.getSettings()['float_type'])
        if ( alreadyNormed is False ):
            states = np.array(norm_state(states, self._state_bounds), dtype=self.getSettings()['float_type'])
            actions = np.array(norm_action(actions, self._action_bounds), dtype=self.getSettings()['float_type'])
            # rewards = np.array(norm_state(rewards, self._reward_bounds), dtype=self.getSettings()['float_type'])
        self.setData(states, actions)
        noise = np.zeros((states.shape[0],1))
        self._noise_shared.set_value(noise)
        return self._get_grad_reward()

    def getNetworkParameters(self):
        params = []
        params.append(copy.deepcopy(self._model.getCriticNetwork().get_weights()))
        # params.append(copy.deepcopy(self._model.getRewardNetwork().get_weights()))
        params.append(copy.deepcopy(self._model.getForwardDynamicsNetwork().get_weights()))
        # params.append(copy.deepcopy(self._modelTarget.getCriticNetwork().get_weights()))
        # params.append(copy.deepcopy(self._modelTarget.getRewardNetwork().get_weights()))
        # params.append(copy.deepcopy(self._modelTarget.getForwardDynamicsNetwork().get_weights()))
        return params
        
    def setNetworkParameters(self, params):
        
        # print("setting critic net params", )
        # print ("same nets:", len(params[0]), self._model.getCriticNetwork().get_weights())
        self._model.getCriticNetwork().set_weights(params[0])
        # print("setting actor net params")
        # self._model.getRewardNetwork().set_weights( params[1] )
        self._model.getForwardDynamicsNetwork().set_weights( params[1] )
        # print("setting critic target net params")
        # self._modelTarget.getCriticNetwork().set_weights( params[2])
        # print("setting actor target net params")
        # self._modelTarget.getRewardNetwork().set_weights( params[3])
        # self._modelTarget.getForwardDynamicsNetwork().set_weights( params[3])
        
    def setData(self, states, actions, result_states=None, rewards=None):
        # self._model.setStates(states)
        # self._model.setActions(actions)
        pass
        """
        if not (result_states is None):
            self._model.setResultStates(result_states)
        if not (rewards is None):
            self._model.setRewards(rewards)
        noise = np.random.normal(self._noise_mean,self._noise_std, size=(states.shape[0],1))
        self._noise_shared.set_value(noise)
        """
        # noise = np.zeros((states.shape[0],1))
        # self._noise_shared.set_value(noise)
            
        
    def trainCritic(self, states, actions, result_states, rewards, updates=1, batch_size=None):
        if (batch_size is None):
            batch_size_=states.shape[0]
        else:
            batch_size_=batch_size
        noise = np.random.normal(self._noise_mean,self._noise_std, size=(states.shape[0],1))
        rewards = (rewards * 0) + 1.0
        # print ("noise: ", noise)
        # print ("Shapes: ", states.shape, actions.shape, rewards.shape, result_states.shape)
        # self._noise_shared.set_value(noise)
        self._updates += 1
        ## Compute actions for TargetNet
        generated_samples = np.array(self._generate([states, actions, noise, 0]))[0]
        # print("generated_samples: ", repr(generated_samples))
        ### Put generated samples in memory
        
        for i in range(generated_samples.shape[0]):
            next_state__ = scale_state(generated_samples[i], self._state_bounds)
            # print("[states[i]]: ", repr([states[i]]))
            # print("next_state__: ", repr(next_state__))
            tup = ([states[i]], [actions[i]], [next_state__], [rewards[i]], [[0]], [[0]], [[0]], [[0]])
            self._experience.insertTuple(tup)
        tmp_result_states = copy.deepcopy(result_states)
        tmp_rewards = states__ = copy.deepcopy(rewards)
        tmp_actions = actions__ = copy.deepcopy(actions)
        tmp_states = copy.deepcopy(states)
        
        ## Pull out a batch of generated samples
        use_exp_mem = False
        if (use_exp_mem):
            states__, actions__, generated_samples, rewards__, falls__, G_ts__, exp_actions__ = self._experience.get_batch(min(states.shape[0], self._experience.samples()))
        """
        print("generated_samples: ", generated_samples.shape)
        print("tmp_result_states: ", tmp_result_states.shape)
        print("tmp_rewards: ", tmp_rewards.shape)
        print("states: ", states.shape)
        print("actions: ", actions.shape)
        """
        
        ## replace half of the samples with generated ones...
        """
        for i in range(int(states.shape[0]/2)):
            
            tmp_result_states[i] = generated_samples[i]
            tmp_rewards[i] = [0]
            tmp_states[i] = states__[i]
            tmp_actions[i] = actions__[i]
        """

        # print("Discriminator targets: ", tmp_rewards)
        
        ### Add noise to labels..
        tmp_rewards = tmp_rewards - np.fabs(np.random.normal(0,0.1, size=(states.shape[0],1)))
        self.setData(tmp_states, tmp_actions, tmp_result_states, tmp_rewards)
        
        ### True samples
        score = self._model.getCriticNetwork().fit([tmp_states, tmp_actions, tmp_result_states], tmp_rewards,
              epochs=updates, batch_size=batch_size_,
              verbose=0,
              shuffle=True
              # callbacks=[early_stopping],
              )
        ### Generated samples
        score = self._model.getCriticNetwork().fit([tmp_states, tmp_actions, generated_samples], np.fabs(np.random.normal(0,0.1, size=(states.shape[0],1))),
              epochs=updates, batch_size=batch_size_,
              verbose=0,
              shuffle=True
              # callbacks=[early_stopping],
              )
        loss = score.history['loss'][0]
        # print("Discriminator loss: ", loss)
        return loss
        
    def trainActor(self, states, actions, result_states, rewards, updates=1, batch_size=None):
        # self.setData(states, actions, result_states, rewards)
        if (batch_size is None):
            batch_size_=states.shape[0]
        else:
            batch_size_=batch_size
        noise = np.random.normal(self._noise_mean,self._noise_std, size=(states.shape[0],1))
        # self._noise_shared.set_value(np.random.normal(self._noise_mean,self._noise_std, size=(states.shape[0],1)))
        ## Add MSE term
        if ( 'train_gan_mse' in self.getSettings() and 
             (self.getSettings()['train_gan_mse'] == True)):
            # self._trainGenerator_MSE()
            self._model._forward_dynamics_net.fit([states, actions, noise * 0], result_states,
              epochs=1, batch_size=batch_size_,
              verbose=0,
              shuffle=True
              # callbacks=[early_stopping],
              )
        # print("Policy mean: ", np.mean(self._q_action(), axis=0))
        loss = 0
        # print("******** Not learning actor right now *****")
        # return loss
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['debug']):
            pass
        
        ### The rewards are not used in this update
        score = self._combined.fit([states, actions, noise], rewards,
              epochs=updates, batch_size=batch_size_,
              verbose=0,
              shuffle=True
              # callbacks=[early_stopping],
              )
        loss = score.history['loss'][0]
        error_MSE = self._bellman_error([states, actions, noise, result_states]) 
        return (np.mean(loss), error_MSE)
        
    def train(self, states, actions, result_states, rewards, updates=1, batch_size=None,
              lstm=False):
        print ("states: ", np.array(states).shape)
        if (batch_size is None):
            batch_size_ = states.shape[0]
        else:
            batch_size_=batch_size
        loss = self.trainCritic(states, actions, result_states, rewards, updates=updates, batch_size=batch_size_)
        # loss = 0
        lossActor = self.trainActor(states, actions, result_states, rewards, updates=updates, batch_size=batch_size_)
        if ( self.getSettings()['train_reward_predictor']):
            # print ("self._reward_bounds: ", self._reward_bounds)
            # print( "Rewards, predicted_reward, difference, model diff, model rewards: ", np.concatenate((rewards, self._predict_reward(), self._predict_reward() - rewards, self._reward_error(), self._reward_values()), axis=1))
            self.setData(states, actions, result_states, rewards)
            lossReward = self._train_reward()
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print ("Loss Reward: ", lossReward)
        return (loss, lossActor)
    
    def predict(self, state, deterministic_=True):
        pass
    
    def predict_batch(self, states, deterministic_=True):
        pass
    
    def predict(self, state, action):
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        # print ("fd state: ", state)
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        # self._model.setStates(state)
        # self._model.setActions(action)
        # self.setData(state,action)
        # self._noise_shared.set_value(np.random.normal(self._noise_mean,self._noise_std, size=(1,1)))
        noise = np.random.normal(self._noise_mean,self._noise_std, size=(1,1))
        # print ("State bounds: ", self._state_bounds)
        # print ("gen output: ", self._generate()[0])
        state_ = scale_state(self._generate([state, action, noise, 0])[0], self._state_bounds)
        # print( "self._state_bounds: ", self._state_bounds)
        # print ("scaled output: ", state_)
        return state_
    
    def predictWithDropout(self, state, action):
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        # print ("fd state: ", state)
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        # self._model.setStates(state)
        # self._model.setActions(action)
        # self.setData(state,action)
        # self._noise_shared.set_value(np.random.normal(self._noise_mean,self._noise_std, size=(1,1)))
        noise = np.random.normal(self._noise_mean,self._noise_std, size=(1,1))
        # print ("State bounds: ", self._state_bounds)
        # print ("gen output: ", self._generate()[0])
        state_ = scale_state(self._generate([state, action, noise, 1])[0], self._state_bounds)
        # print( "self._state_bounds: ", self._state_bounds)
        # print ("scaled output: ", state_)
        return state_
    
    def predict_batch(self, states, actions):
        ## These input should already be normalized.
        # self._model.setStates(states)
        # self._model.setActions(actions)
        self.setData(states,actions)
        # self._noise_shared.set_value(np.random.normal(self._noise_mean,self._noise_std, size=(states.shape[0],1)))
        noise = np.random.normal(self._noise_mean,self._noise_std, size=(states.shape[0],1))
        # print ("State bounds: ", self._state_bounds)
        # print ("fd output: ", self._forwardDynamics()[0])
        # state_ = scale_state(self._generate(), self._state_bounds)
        state_ = self._generate([states, actions, noise, 0])
        return state_
    
    def q_value(self, state):
        """
            For returning a vector of q values, state should NOT be normalized
        """
        # states = np.zeros((self._batch_size, self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=theano.config.floatX)
        self._model.setStates(state)
        self._modelTarget.setStates(state)
        action = self._q_action()
        self._model.setActions(action)
        self._modelTarget.setActions(action)
        return scale_reward(self._discriminate(), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
        # return self._q_valTarget()[0]
        # return self._q_val()[0]
        
    def q_value(self, state, action, next_state):
        """
            For returning a vector of q values, state should NOT be normalized
        """
        # states = np.zeros((self._batch_size, self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=theano.config.floatX)
        # action = self._q_action()
        action = norm_state(action, self.getActionBounds())
        nextState = norm_state(next_state, self.getStateBounds())
        # nextState = np.reshape(nextState, (1,20))
        
        # return scale_reward(self._discriminate(), self.getRewardBounds())[0] * (1.0 / (1.0- self.getSettings()['discount_factor']))
        return self._discriminate([state, action, nextState])
        # return self._q_valTarget()[0]
        # return self._q_val()[0]
    
    def q_values(self, state):
        """
            For returning a vector of q values, state should already be normalized
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
        
    def predict_std(self, state, deterministic_=True, p=1.0):
        """
            This does nothing for a GAN...
        """
        # states = np.zeros((self._batch_size, self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        action_std = np.array([0] * len(self._action_bounds))
        # np.zeros((state.shape[0], len(self._action_bounds)))
        # else:
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # action_ = q_valsActA[0]
        return action_std
    
    def predict_reward(self, state, action):
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        self._model.setStates(state)
        self._model.setActions(action)
        predicted_reward = self._predict_reward()
        reward_ = scale_reward(predicted_reward, self.getRewardBounds()) # * (1.0 / (1.0- self.getSettings()['discount_factor']))
        # reward_ = scale_reward(predicted_reward, self.getRewardBounds())[0] * (1.0 / (1.0- self.getSettings()['discount_factor']))
        # reward_ = scale_state(predicted_reward, self._reward_bounds)
        # print ("reward, predicted reward: ", reward_, predicted_reward)
        return reward_

    def predict_reward_batch(self, states, actions):
        
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        # state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        # action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        self._model.setStates(states)
        self._model.setActions(actions)
        predicted_reward = self._predict_reward()
        # reward_ = scale_reward(predicted_reward, self.getRewardBounds())[0] # * (1.0 / (1.0- self.getSettings()['discount_factor']))
        # reward_ = scale_reward(predicted_reward, self.getRewardBounds())[0] * (1.0 / (1.0- self.getSettings()['discount_factor']))
        # reward_ = scale_state(predicted_reward, self._reward_bounds)
        # print ("reward, predicted reward: ", reward_, predicted_reward)
        return predicted_reward
    
    def bellman_error(self, states, actions, result_states, rewards):
        noise = np.random.normal(self._noise_mean,self._noise_std, size=(states.shape[0],1))
        return self._bellman_error([states, actions, noise, result_states]) 
    
    def reward_error(self, states, actions, result_states, rewards):
        # rewards = rewards * (1.0/(1.0-self.getSettings()['discount_factor'])) # scale rewards
        # self.setData(states, actions, result_states, rewards)
        return 0

    
    def setStateBounds(self, state_bounds):
        super(GANKeras,self).setStateBounds(state_bounds)
        """
        print ("")
        print("Setting GAN state bounds: ", state_bounds)
        print("self.getStateBounds(): ", self.getStateBounds())
        print ("")
        """
        self._experience.setStateBounds(copy.deepcopy(self.getStateBounds()))
        
    def setActionBounds(self, action_bounds):
        super(GANKeras,self).setActionBounds(action_bounds)
        self._experience.setActionBounds(copy.deepcopy(self.getActionBounds()))
        
    def setRewardBounds(self, reward_bounds):
        super(GANKeras,self).setRewardBounds(reward_bounds)
        self._experience.setRewardBounds(copy.deepcopy(self.getRewardBounds()))
        
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
        self._model.getCriticNetwork().save(fileName+"_reward"+suffix, overwrite=True)
        # print ("self._model._actor_train: ", self._model._actor_train)
        
    def loadFrom(self, fileName):
        from keras.models import load_model
        import h5py
        suffix = ".h5"
        print ("Loading agent: ", fileName)
        # with K.get_session().graph.as_default() as g:
        self._model._forward_dynamics_net = load_model(fileName+"_FD"+suffix)
        self._model._critic = load_model(fileName+"_reward"+suffix)
        if (self._modelTarget is not None):
            self._modelTarget._forward_dynamics_net = load_model(fileName+"_FD"+suffix)
            self._modelTarget._reward_net = load_model(fileName+"_reward"+suffix)
        # self._model._actor_train = load_model(fileName+"_actor_train"+suffix, custom_objects={'loss': pos_y})
        # self._value = K.function([self._model.getStateSymbolicVariable(), K.learning_phase()], [self.__value])
        # self._value_Target = K.function([self._model.getResultStateSymbolicVariable(), K.learning_phase()], [self.__value_Target])
        hf = h5py.File(fileName+"_bounds.h5",'r')
        self.setStateBounds(np.array(hf.get('_state_bounds')))
        self.setRewardBounds(np.array(hf.get('_reward_bounds')))
        self.setActionBounds(np.array(hf.get('_action_bounds')))
        # self._result_state_bounds = np.array(hf.get('_result_state_bounds'))
        hf.close()
        
        