import numpy as np
import copy

import sys
from model.ModelUtil import *
import keras.backend as K
import keras
from keras.models import Sequential, Model

from algorithm.AlgorithmInterface import AlgorithmInterface

class DoubleDQN_KERAS(AlgorithmInterface):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=False):
        
        super(DoubleDQN_KERAS,self).__init__(model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)

        self._model._actor = Model(inputs=[self._model.getStateSymbolicVariable()], outputs=self._model._actor)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("Actor summary: ", self._model._actor.summary())
        
        if ( "use_centralized_critic" in self.getSettings()
             and (self.getSettings()["use_centralized_critic"] == True)
             and False):
            self._model._critic = Model(inputs=[self._model.getResultStateSymbolicVariable(),
                                              self._model.getActionSymbolicVariable()], outputs=self._model._critic)
        else:
            self._model._critic = Model(inputs=[self._model.getStateSymbolicVariable(),
                                              self._model.getActionSymbolicVariable()], outputs=self._model._critic)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("Critic summary: ", self._model._critic.summary())
        
        self._modelTarget = type(self._model)(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=False)
        self._modelTarget._actor = Model(inputs=[self._modelTarget.getStateSymbolicVariable()], outputs=self._modelTarget._actor)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("Target Actor summary: ", self._modelTarget._actor.summary())
        if ( "use_centralized_critic" in self.getSettings()
             and (self.getSettings()["use_centralized_critic"] == True)
             and False):
            self._modelTarget._critic = Model(inputs=[self._modelTarget.getResultStateSymbolicVariable(),
                                                  self._modelTarget.getActionSymbolicVariable()], outputs=self._modelTarget._critic)
        else:
            self._modelTarget._critic = Model(inputs=[self._modelTarget.getStateSymbolicVariable(),
                                                  self._modelTarget.getActionSymbolicVariable()], outputs=self._modelTarget._critic)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("Target Critic summary: ", self._modelTarget._critic.summary())

        self._discount_factor= self.getSettings()['discount_factor']
        self._rho = self.getSettings()['rho']
        self._rms_epsilon = self.getSettings()['rms_epsilon']

        sgd = keras.optimizers.Adam(lr=self.getSettings()['critic_learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=self._rms_epsilon, decay=0.0)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Clipping: ", sgd.decay)
        self._model.getCriticNetwork().compile(loss='mse', optimizer=sgd)
        
        sgd = keras.optimizers.Adam(lr=self.getSettings()['critic_learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=self._rms_epsilon, decay=0.0)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Clipping: ", sgd.decay)
        self._modelTarget.getCriticNetwork().compile(loss='mse', optimizer=sgd)

        DoubleDQN_KERAS.compile(self)


    def compile(self):       
        self._q_valsA = self._model.getActorNetwork()([self._model.getStateSymbolicVariable()])
        self._q_valsB = self._modelTarget.getActorNetwork()([self._model.getStateSymbolicVariable()])
        self._q_valsA_B = self._model.getActorNetwork()([self._model.getResultStateSymbolicVariable()])
        self._q_valsB_A = self._modelTarget.getActorNetwork()([self._model.getResultStateSymbolicVariable()])
        
        target = (self._model.getRewardSymbolicVariable() +
                #(T.ones_like(terminals) - terminals) *
                  self._discount_factor * K.max(self._q_valsB_A, axis=1, keepdims=True))
        # diff = target - self._q_valsA[K.arange(keras.backend.int_shape(self._model.getStateSymbolicVariable())[-1]),
        #                       self._model.getActionSymbolicVariable().reshape((-1,))].reshape((-1, 1))# Does some fancy indexing to get the column of interest
        diff = target - self._q_valsA[K.arange( self._model.getActionSymbolicVariable().reshape((-1,))]# Does some fancy indexing to get the column of interest
                               
        loss = diff
        
        targetB = (self._model.getRewardSymbolicVariable() +
                #(T.ones_like(terminals) - terminals) *
                  self._discount_factor * K.max(self._q_valsA_B, axis=1, keepdims=True))
        diffB = targetB - self._q_valsB[K.arange(len(self._modelTarget.getStateValues())),
                               self._model.getActionSymbolicVariable().reshape((-1,))].reshape((-1, 1))# Does some fancy indexing to get the column of interest
                               
        lossB = diffB 

        sgd = keras.optimizers.Adam(lr=self.getSettings()['critic_learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=self._rms_epsilon, decay=0.0)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Clipping: ", sgd.decay)
        self._model.getActorNetwork().compile(loss='mse', optimizer=sgd)

        sgd = keras.optimizers.Adam(lr=self.getSettings()['critic_learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=self._rms_epsilon, decay=0.0)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Clipping: ", sgd.decay)
        self._modelTarget.getActorNetwork().compile(loss='mse', optimizer=sgd)

        
    def train(self, states, actions, rewards, result_states):
        self.setData(states, actions, rewards, result_states)

        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModel()
        self._updates += 1
        import random
        r = random.choice([0,1])
        if r == 0:
            loss, _ = self._model.getActorNetwork().fit([states, actions, rewards, result_states])
            
            # diff_ = self.bellman_error(states, actions, rewards, result_states)
        else:
            loss, _ = self._modelTarget.getActorNetwork().fit([states, actions, rewards, result_states])
            
            # diff_ = self.bellman_errorB(states, actions, rewards, result_states)
        return loss
    
    def q_value(self, state):
        """
            Don't normalize here it is done in q_values
        """
        # state = [norm_state(state, self._state_bounds)]
        q_values = self.q_values(state)
        action_ = self.predict(state)
        # print ("q_values: " + str(q_values) + " Action: " + str(action_))
        original_val = q_values[action_]
        return original_val
    
    def predict(self, state):
        """
            Don't normalize here it is done in q_values
        """
        # state = [norm_state(state, self._state_bounds)]
        q_vals = self.q_values(state)
        return np.argmax(q_vals)
    
    def q_values(self, state):
        state = [norm_state(state, self._state_bounds)]
        # print ("Q value: ", state)
        self._model.setStates(state)
        return self._q_vals()[0]
    
    def bellman_error(self, states, actions, rewards, result_states):
        # print ("Bellman error 2 actions: ", len(actions) , " rewards ", len(rewards), " states ", len(states), " result_states: ", len(result_states))
        b = self._model.getActorNetwork().predict([states, actions, rewards, result_states])
        return self._bellman_error2()
        # return self._bellman_error(state, action, reward, result_state)
