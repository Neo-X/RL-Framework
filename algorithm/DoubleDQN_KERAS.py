import numpy as np
import copy

import sys
from model.ModelUtil import *
import keras.backend as K
import keras
from keras.models import Sequential, Model
from algorithm.KERASAlgorithm import *

from algorithm.AlgorithmInterface import AlgorithmInterface

class DoubleDQN_KERAS(KERASAlgorithm):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=False):
        
        super(DoubleDQN_KERAS,self).__init__(model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)

        self._model._actor = Model(inputs=[self._model.getStateSymbolicVariable()], outputs=self._model._actor)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("Actor summary: ", self._model._actor.summary())
        
        
        n_out = self.getSettings()["discrete_actions"]
        print ("n_out:", n_out)
        self._modelB = type(self._model)(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=False)
        self._modelB._actor = Model(inputs=[self._modelB.getStateSymbolicVariable()], outputs=self._modelB._actor)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("ActorB summary: ", self._modelB._actor.summary())
       
        self._modelTarget = type(self._model)(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=False)
        self._modelTarget._actor = Model(inputs=[self._modelTarget.getStateSymbolicVariable()], outputs=self._modelTarget._actor)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("Actor Target summary: ", self._modelTarget._actor.summary())
        
        self._modelBTarget = type(self._model)(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=False)
        self._modelBTarget._actor = Model(inputs=[self._modelBTarget.getStateSymbolicVariable()], outputs=self._modelBTarget._actor)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("Actor B Target summary: ", self._modelBTarget._actor.summary())
       

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
        # diff = target - self._q_valsA[K.max(self._model.getActionSymbolicVariable())]# Does some fancy indexing to get the column of interest
        # loss = diff
        
        # targetB = (self._model.getRewardSymbolicVariable() +
                #(T.ones_like(terminals) - terminals) *
        #          self._discount_factor * K.max(self._q_valsA_B, axis=1, keepdims=True))
        # diffB = targetB - self._q_valsB[K.arange(len(self._modelTarget.getStateValues())),
        #                        self._model.getActionSymbolicVariable().reshape((-1,))].reshape((-1, 1))# Does some fancy indexing to get the column of interest
                               
        # lossB = diffB 

        sgd = keras.optimizers.Adam(lr=self.getSettings()['critic_learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=self._rms_epsilon, decay=0.0)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Clipping: ", sgd.decay)
        self._model.getActorNetwork().compile(loss='mse', optimizer=sgd)

        sgd = keras.optimizers.Adam(lr=self.getSettings()['critic_learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=self._rms_epsilon, decay=0.0)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Clipping: ", sgd.decay)
        self._modelB.getActorNetwork().compile(loss='mse', optimizer=sgd)

    def reset(self):
        """
            Reset any state for the agent model
        """
        # self._model.reset()
        # if not (self._modelTarget is None):
        #     self._modelTarget.reset()

    def getNetworkParameters(self):
        params = []
        params.append(copy.deepcopy(self._model.getActorNetwork().get_weights()))
        params.append(copy.deepcopy(self._modelB.getActorNetwork().get_weights()))
        params.append(copy.deepcopy(self._modelTarget.getActorNetwork().get_weights()))
        params.append(copy.deepcopy(self._modelBTarget.getActorNetwork().get_weights()))
        return params
    
    def setNetworkParameters(self, params):
        self._model.getActorNetwork().set_weights(params[0])
        self._modelB.getActorNetwork().set_weights( params[1] )
        self._modelTarget.getActorNetwork().set_weights( params[2])
        self._modelBTarget.getActorNetwork().set_weights( params[3])

    def updateTargetModel(self):
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Updating target Model")
        """
            Target model updates
        """
        ### Some models don't have target networks. Mostly FD models.
        if (self._model is not None and (self._modelTarget is not None)):
            self._modelTarget.getActorNetwork().set_weights( copy.deepcopy(self._model.getActorNetwork().get_weights()))
            self._modelBTarget.getActorNetwork().set_weights( copy.deepcopy(self._modelB.getActorNetwork().get_weights()))
        
    def trainCritic(self, states, actions, rewards, result_states, falls, G_t=[[0]], p=1.0,
                    updates=1, batch_size=None):

        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModel()
        self._updates += 1
        import random
        r = random.choice([0,1])
        # if r == 0:
        targets = self._model.getActorNetwork().predict(states)
        # print ("targets", targets)
        maxQ = np.max(self._modelTarget.getActorNetwork().predict(result_states), axis=-1, keepdims=True)
        # print ("maxQ", maxQ)
        target = rewards + (self._discount_factor * maxQ)
        # print ("target", target)
        # print ("actions", actions)
        # print ("targets[actions]: ", targets[actions])
        # print ("targets[actions] 2: ", targets[actions.flatten()])
        for i in range(len(states)):
            targets[i][actions[i][0]] = target[i]
        # print ("targets", targets)
        score = self._model.getActorNetwork().fit([states], [targets], epochs=1, 
                            batch_size=states.shape[0],
                            verbose=0)
        """
        else:
            targets = self._modelB.getActorNetwork().predict(state)
            target = rewards + self._discount_factor * np.max(self._modelBTarget.getActorNetwork().predict(result_states), axis=-1)
            targets[actions] = target
            score = self._modelB.getActorNetwork().fit([states], [target], epochs=1, 
                              batch_size=states.shape[0],
                              verbose=0)
        """ 
            # diff_ = self.bellman_errorB(states, actions, rewards, result_states)
        loss = np.mean(score.history['loss'])
        return loss

    def trainActor(self, states, actions, rewards, result_states, falls, advantage, exp_actions=None, 
                   G_t=[[0]], forwardDynamicsModel=None, p=1.0, updates=1, batch_size=None):
        pass 

    def predict(self, state, deterministic_=True, evaluation_=False, p=None, sim_index=None, bootstrapping=False):
        """
            Don't normalize here it is done in q_values
        """
        state = norm_state(state, self._state_bounds)
        # q_vals = self.q_values2(state)
        r = random.choice([0,1])
        if r == 0:
            action = np.argmax(self._model.getActorNetwork().predict(state))
        else:
            action = np.argmax(self._modelB.getActorNetwork().predict(state))

        # print ("action: ", action)
        action = np.array([action])
        # print ("action2: ", action)
        return action

    def compute_q(self, state):
        # state = [state]
        # print ("state: ", state)
        state = np.array(state, dtype=self._settings['float_type'])
        # print ("Q value: ", state)
        values = self._model.getActorNetwork().predict(state)
        valuesB = self._modelB.getActorNetwork().predict(state)
        # values = np.concatenate((values, valuesB), axis=2)
        
        # print ("values: ", values)
        # print ("valuesB: ", valuesB)
        minQ = np.minimum(values, valuesB)
        # print ("minQ: ", minQ )
        values = minQ
        values = np.max(minQ, axis=-1, keepdims=True)
        # print  ("values: ", values)
        return values
        
    def q_value(self, state):
        # state = [state]
        state = norm_state(state, self._state_bounds)
        q = self.compute_q(state)
        values = (q * action_bound_std(self.getRewardBounds())) * (1.0 / (1.0- self.getSettings()['discount_factor']))
        return values
    
    def q_values(self, state):
        # state = [state]
        q = self.compute_q(state)
        return q

    def q_values2(self, states, wrap=True):
        ### These versions of states are NOT normalized yet
        # state = [state]
        state = norm_state(states, self._state_bounds)
        q = self.compute_q(state)
        values = (q * action_bound_std(self.getRewardBounds())) * (1.0 / (1.0- self.getSettings()['discount_factor']))
        return values
    
    def bellman_error(self, states, actions, rewards, result_states, falls):
        # print ("Bellman error 2 actions: ", len(actions) , " rewards ", len(rewards), " states ", len(states), " result_states: ", len(result_states))
        b = self._model.getActorNetwork().predict([states])
        return np.min(b, axis=-1, keepdims=True)
        # return self._bellman_error(state, action, reward, result_state)
