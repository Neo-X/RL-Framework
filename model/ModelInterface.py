import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *

# For debugging
# theano.config.mode='FAST_COMPILE'

class ModelInterface(object):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        # super(DeepCACLADropout,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        self._batch_size=settings_['batch_size']
        self._state_length = n_in
        self._action_length = n_out
        self._settings = settings_
        # data types for model
        self._dropout_p=settings_['dropout_p']
        
    def getNetworkParameters(self):
        pass
    
    def setNetworkParameters(self, params):
        pass
    
    def getActorNetwork(self):
        return self._actor
    
    def getActorNetworkTaskPart(self):
        return self._actor_task_part
    
    def getCriticNetwork(self):
        return self._critic
    
    def getCriticNetworkTaskPart(self):
        return self._critic_task_part
    
    def getForwardDynamicsNetwork(self):
        return self._forward_dynamics_net
    def getRewardNetwork(self):
        return self._reward_net
    
    ### Setting network input values ###    
    def setStates(self, states):
        self._states_shared.set_value(states)
    def setActions(self, actions):
        self._actions_shared.set_value(actions)
    def setResultStates(self, resultStates):
        self._next_states_shared.set_value(resultStates)
    def setRewards(self, rewards):
        self._rewards_shared.set_value(rewards)
    def setTargets(self, targets):
        self._targets_shared.set_value(targets)
        
    ### Setting network input values ###    
    def getStateValues(self):
        return self._states_shared.get_value()
    def getActionValues(self):
        return self._actions_shared.get_value()
    def getResultStateValues(self):
        return self._next_states_shared.get_value()
    def getRewardValues(self):
        return self._rewards_shared.get_value()
    def getTargetValues(self):
        return self._target_shared.get_value()
    
    ####### Getting the shared variables to set values. #######  
    def getStates(self):
        return self._states_shared
    def getActions(self):
        return self._actions_shared
    def getResultStates(self):
        return self._next_states_shared
    def getRewards(self):
        return self._rewards_shared
    def getTargets(self):
        return self._targets_shared
    
    def getSettings(self):
        return self._settings
    
    ######### Symbolic Variables ######
    def getStateSymbolicVariable(self):
        return self._State
    def getActionSymbolicVariable(self):
        return self._Action
    def getResultStateSymbolicVariable(self):
        return self._ResultState
    def getRewardSymbolicVariable(self):
        return self._Reward
    def getTargetsSymbolicVariable(self):
        return self._Target
