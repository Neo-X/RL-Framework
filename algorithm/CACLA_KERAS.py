import theano
from theano import tensor as T
from lasagne.layers import get_all_params
import numpy as np
# import lasagne
import sys
import copy
sys.path.append('../')
from model.ModelUtil import *
from algorithm.AlgorithmInterface import AlgorithmInterface
from model.LearningUtil import loglikelihood, kl, entropy, change_penalty
from keras.optimizers import SGD
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K

# For debugging
# theano.config.mode='FAST_COMPILE'
# from DeepCACLA import DeepCACLA

class CACLA_KERAS(AlgorithmInterface):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(CACLA_KERAS,self).__init__(model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        
        ## primary network
        self._model = model

        sgd = SGD(lr=0.01, momentum=0.9)
        print ("Clipping: ", sgd.decay)
        self._model.getCriticNetwork().compile(loss='mse', optimizer=sgd)
        sgd = SGD(lr=0.01, momentum=0.9)
        print ("Clipping: ", sgd.decay)
        self._model.getActorNetwork().compile(loss='mse', optimizer=sgd)
        
        print ("Loss ", self._model.getActorNetwork().total_loss)
        
        ## Target network
        self._modelTarget = copy.deepcopy(model)
        # self._modelTarget = model
        
        CACLA_KERAS.compile(self)
        
    def compile(self):
        
        pass
        
    def updateTargetModel(self):
        print ("Updating target Model")
        """
            Target model updates
        """
        for i in range(len(self._model.getCriticNetwork().layers)):
            self._modelTarget.getCriticNetwork().layers[i].set_weights(self._model.getCriticNetwork().layers[i].get_weights())
        
        for i in range(len(self._model.getActorNetwork().layers)):
            self._modelTarget.getActorNetwork().layers[i].set_weights(self._model.getActorNetwork().layers[i].get_weights())
    
    def setData(self, states, actions, rewards, result_states, fallen):
        pass
        # _targets = rewards + (self._discount_factor * self._q_valsTargetNextState )
        
    def getGrads(self, states):
        # self.setData(states, actions, rewards, result_states)
        states = np.array(states, dtype=theano.config.floatX)
        self._model.setStates(states)
        return self._get_grad()

    def trainCritic(self, states, actions, rewards, result_states, falls):
        self.setData(states, actions, rewards, result_states, falls)
        # print ("Performing Critic trainning update")
        
        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModel()
        self._updates += 1
        # print ("Falls:", falls)
        # print ("Rewards: ", rewards)
        # print ("Target Values: ", self._get_target())
        # print ("V Values: ", np.mean(self._q_val()))
        # print ("diff Values: ", np.mean(self._get_diff()))
        # data = np.append(falls, self._get_target()[0], axis=1)
        # print ("Rewards, Falls, Targets:", np.append(rewards, data, axis=1))
        # print ("Rewards, Falls, Targets:", [rewards, falls, self._get_target()])
        # print ("Actions: ", actions)
        y_ = self._modelTarget.getCriticNetwork().predict(result_states, batch_size=200)
        target_ = rewards + ((self._discount_factor * y_) * falls)
        print ("Critic Target: ", target_)
        score = self._model.getCriticNetwork().fit(states, target_,
              nb_epoch=1, batch_size=32
              # callbacks=[early_stopping],
              )
        loss = score.history['loss']
        print(" Critic loss: ", loss)
        
        return loss
    
    def trainActor(self, states, actions, rewards, result_states, falls, advantage):
        lossActor = 0
        
        diff_ = self.bellman_error(states, actions, rewards, result_states, falls)
        # print ("Diff")
        # print (diff_)
        tmp_states=[]
        tmp_result_states=[]
        tmp_actions=[]
        tmp_rewards=[]
        tmp_falls=[]
        tmp_diff=[]
        for i in range(len(diff_)):
            if ( diff_[i] > 0.0):
                tmp_diff.append(diff_[i])
                tmp_states.append(states[i])
                tmp_result_states.append(result_states[i])
                tmp_actions.append(actions[i])
                tmp_rewards.append(rewards[i])
                tmp_falls.append(falls[i])
                
        if (len(tmp_actions) > 0):
            # self._tmp_diff_shared.set_value(tmp_diff)
            # self.setData(tmp_states, tmp_actions, tmp_rewards, tmp_result_states, tmp_falls)
            
            score = self._model.getActorNetwork().fit(tmp_states, tmp_actions,
              nb_epoch=1, batch_size=len(tmp_actions)
              # callbacks=[early_stopping],
              )
            lossActor = score.history['loss']
        
            # print ("Actor diff: ", np.mean(np.array(self._get_diff()) / (1.0/(1.0-self._discount_factor))))
            # lossActor, _ = self._trainActor()
            print( "Length of positive actions: " , str(len(tmp_actions)), " Actor loss: ", lossActor)
            # print( " Actor loss: ", lossActor)
            # print("Diff for actor: ", self._get_diff())
            # print ("Tmp_diff: ", tmp_diff)
            # print ( "Action before diff: ", self._get_actor_diff_())
            # print( "Action diff: ", self._get_action_diff())
            # return np.sqrt(lossActor);
        return lossActor
    
    def train(self, states, actions, rewards, result_states, falls):
        loss = self.trainCritic(states, actions, rewards, result_states, falls)
        lossActor = self.trainActor(states, actions, rewards, result_states, falls)
        return loss
    
    def predict(self, state, deterministic_=True):
        # states = np.zeros((self._batch_size, self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        # state = np.array(state, dtype=theano.config.floatX)
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=theano.config.floatX)
        self._model.setStates(state)
        # action_ = lasagne.layers.get_output(self._model.getActorNetwork(), state, deterministic=deterministic_).mean()
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # if deterministic_:
        action_ = scale_action(self._modelTarget.getActorNetwork().predict(state, batch_size=1)[0], self._action_bounds)
        # action_ = scale_action(self._q_action_target()[0], self._action_bounds)
        # else:
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # action_ = q_valsActA[0]
        return action_
    
    def predictWithDropout(self, state, deterministic_=True):
        # states = np.zeros((self._batch_size, self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = np.array(state, dtype=theano.config.floatX)
        state = norm_state(state, self._state_bounds)
        self._model.setStates(state)
        # action_ = lasagne.layers.get_output(self._model.getActorNetwork(), state, deterministic=deterministic_).mean()
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # if deterministic_:
        action_ = scale_action(self._modelTarget.getActorNetwork().predict(states, batch_size=1)[0], self._action_bounds)
        # else:
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # action_ = q_valsActA[0]
        return action_
    
    def q_value(self, state):
        # states = np.zeros((self._batch_size, self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=theano.config.floatX)
        self._model.setStates(state)
        self._modelTarget.setStates(state)
        # return scale_reward(self._q_valTarget(), self.getRewardBounds())[0]
        return self._modelTarget.getCriticNetwork().predict(state, batch_size=1)[0]
        # return self._q_val()[0]
    
    def q_values(self, state):
        """
            For returning a vector of q values, state should already be normalized
        """
        state = np.array(state, dtype=theano.config.floatX)
        self._model.setStates(state)
        self._modelTarget.setStates(state)
        return self._modelTarget.getCriticNetwork().predict(state, batch_size=state.shape[0])
    
    def q_valueWithDropout(self, state):
        # states = np.zeros((self._batch_size, self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = np.array(state, dtype=theano.config.floatX)
        state = norm_state(state, self._state_bounds)
        self._model.setStates(state)
        return scale_reward(self._q_val_drop(), self.getRewardBounds())[0]
    
    def bellman_error(self, states, actions, rewards, result_states, falls):
        y_ = self._modelTarget.getCriticNetwork().predict(result_states, batch_size=32)
        target_ = rewards + ((self._discount_factor * y_) * falls)
        values =  self._modelTarget.getCriticNetwork().predict(states, batch_size=32)
        bellman_error = target - values
        return bellman_error
        # return self._bellman_errorTarget()
