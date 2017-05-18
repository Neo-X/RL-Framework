import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
# from ModelUtil import *

# For debugging
# theano.config.mode='FAST_COMPILE'
# from DeepCACLA import DeepCACLA

class AlgorithmInterface(object):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        self._batch_size=settings_['batch_size']
        self._state_length = n_in
        self._action_length = n_out
        self._settings = settings_
        
        self.setActionBounds(action_bounds) 
        self.setStateBounds(state_bounds) 
        self.setRewardBounds(reward_bound) 
        
        # data types for model
        
        """
        State = T.dmatrix("State")
        State.tag.test_value = np.random.rand(batch_size,self._state_length)
        ResultState = T.dmatrix("ResultState")
        ResultState.tag.test_value = np.random.rand(batch_size,self._state_length)
        Reward = T.col("Reward")
        Reward.tag.test_value = np.random.rand(batch_size,1)
        Action = T.dmatrix("Action")
        Action.tag.test_value = np.random.rand(batch_size, self._action_length)
        """
        # create a small convolutional neural network
        
        self._learning_rate = self.getSettings()['learning_rate']
        self._discount_factor= self.getSettings()['discount_factor']
        self._rho = self.getSettings()['rho']
        self._rms_epsilon = self.getSettings()['rms_epsilon']
        
        self._weight_update_steps= self.getSettings()['steps_until_target_network_update']
        self._regularization_weight= self.getSettings()['regularization_weight']
        self._updates=0
        
    def setActor(self, actor):
        self._actor = actor
    def setEnvironment(self, sim):
        self._sim = sim # The real simulator that is used for predictions
        
    def compile(self):
        """
            Compiles the functions for this algorithm
        """
        pass
        
    def updateTargetModel(self):
        pass

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
        
    def setTaskNetworkParameters(self, otherModel):
        all_paramsA = lasagne.layers.helper.get_all_param_values(otherModel.getModel().getActorNetworkTaskPart())
        all_paramsActA = lasagne.layers.helper.get_all_param_values(otherModel.getModel().getActorNetworkTaskPart())
        lasagne.layers.helper.set_all_param_values(self._model.getCriticNetworkTaskPart(), all_paramsA)
        lasagne.layers.helper.set_all_param_values(self._model.getActorNetworkTaskPart(), all_paramsActA)
        
    def getModel(self):
        return self._model
    
    def predict(self, state, deterministic_=True):
        pass
    
    def predictWithDropout(self, state, deterministic_=True):
        pass
    
    def q_value(self, state):
        """
            For returning a vector of q values, state should NOT be normalized
        """
        pass
    
    def q_values(self, state):
        """
            For returning a vector of q values, state should already be normalized
        """
        pass
    
    def bellman_error(self, state, action, reward, result_state):
        pass
    
    def train(self, states, actions, rewards, result_states):
        loss = self.trainCritic(states, actions, rewards, result_states)
        lossActor = self.trainActor(states, actions, rewards, result_states)
        return loss
    
    def getSettings(self):
        return self._settings
    
    def setStateBounds(self, bounds):
        self._state_bounds = bounds
    def setActionBounds(self, bounds):
        self._action_bounds = bounds
    def setRewardBounds(self, bounds):
        self._reward_bounds = bounds
    def getStateBounds(self):
        return self._state_bounds
    def getActionBounds(self):
        return self._action_bounds
    def getRewardBounds(self):
        return self._reward_bounds
        
        ### Setting network input values ###    
    def setStates(self, states):
        self._states_shared.set_value(states)
    def setActions(self, actions):
        self._actions_shared.set_value(actions)
    def setResultStates(self, resultStates):
        self._next_states_shared.set_value(resultStates)
    def setRewards(self, rewards):
        self._rewards_shared.set_value(rewards)
    
    def getStateSize(self):
        return self._state_length
    def getActionSize(self): 
        return self._action_length

    def init(self, state_length, action_length, state_bounds, action_bounds, actor, exp, settings):
        pass
    
    def initEpoch(self):
        pass