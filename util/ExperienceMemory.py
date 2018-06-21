
import numpy as np
import random
import h5py
from model.ModelUtil import validBounds, fixBounds, anneal_value, norm_state, norm_action, norm_reward, checkValidData
import copy
import sys


class ExperienceMemory(object):
    """
        Contains the recient history of experience tuples
         
        I have decided that the experience memory will contain real values from the simulation.
        Not values that have been normalize. I think this will make things easy down the road
        If I wanted to adjust the model scale now I won't have to update all the tuples in the memory.
        Also, a scale layer can be added to the model to compensate for having to scale every tuple
        when performing training updates.
    """
    
    def __init__(self, state_length, action_length, memory_length, continuous_actions=False, settings=None, result_state_length=None):
        
        if (settings == None):
            self._settings = {}
            self._settings['discount_factor'] = 0.0
            # self._settings['float_type'] = 'float32'
        else:
            self._settings = settings
        
        self._history_size=memory_length
        self._state_length = state_length
        self._action_length = action_length
        self._continuous_actions = continuous_actions
        if ( result_state_length == None ):
            self._result_state_length = state_length
        else:
            self._result_state_length = result_state_length
        # self._settings = settings
        self._history_update_index=0 # where the next experience should write
        self._inserts=0
        self.clear()
        # self._state_history = theano.shared(np.zeros((self._history_size, state_length)))
        # self._action_history = theano.shared(np.zeros((self._history_size, action_length)))
        # self._nextState_history = theano.shared(np.zeros((self._history_size, state_length)))
        # self._reward_history = theano.shared(np.zeros((self._history_size, 1)))
        
    def clear(self):
        self._history_update_index=0 # where the next experience should write
        self._inserts=0 ## How many samples are in the buffer
        
        if (self._settings['float_type'] == 'float32'):
            
            self._state_history = (np.zeros((self._history_size, self._state_length), dtype='float32'))
            if self._continuous_actions:
                self._action_history = (np.zeros((self._history_size, self._action_length), dtype='float32'))
            else:
                self._action_history = (np.zeros((self._history_size, self._action_length), dtype='int8'))
            self._nextState_history = (np.zeros((self._history_size, self._result_state_length), dtype='float32'))
            self._reward_history = (np.zeros((self._history_size, 1), dtype='float32'))
            self._fall_history = (np.zeros((self._history_size, 1), dtype='int8'))
            self._discounted_sum_history = (np.zeros((self._history_size, 1), dtype='float32'))
            self._advantage_history = (np.zeros((self._history_size, 1), dtype='float32'))
            self._exp_action_history = (np.zeros((self._history_size, 1), dtype='int8'))
        else:
            self._state_history = (np.zeros((self._history_size, self._state_length), dtype='float64'))
            if self._continuous_actions:
                self._action_history = (np.zeros((self._history_size, self._action_length), dtype='float64'))
            else:
                self._action_history = (np.zeros((self._history_size, self._action_length), dtype='int8'))
            self._nextState_history = (np.zeros((self._history_size, self._result_state_length), dtype='float64'))
            self._reward_history = (np.zeros((self._history_size, 1), dtype='float64'))
            self._fall_history = (np.zeros((self._history_size, 1), dtype='int8'))
            self._discounted_sum_history = (np.zeros((self._history_size, 1), dtype='float64'))
            self._advantage_history = (np.zeros((self._history_size, 1), dtype='float64'))
            self._exp_action_history = (np.zeros((self._history_size, 1), dtype='int8'))
        
    def insertTuple(self, tuple):
        
        (state, action, nextState, reward, fall, G_t, exp_action, advantage) = tuple
        self.insert(state, action, nextState, reward, fall, G_t, exp_action, advantage)
        
    def insert(self, state, action, nextState, reward, fall=[[0]], G_t=[[0]], exp_action=[[0]], advantage=[[0]]):
        # print "Instert State: " + str(state)
        # state = list(state)
        assert len(state[0]) == self._state_length, "len(state[0]) == self._state_length: " + str(state) 
        assert len(action[0]) == self._action_length, "len(action[0]) == self._action_length: " + str(action)
        assert len(nextState[0]) == self._result_state_length, "len(nextState[0]) == self._result_state_length: "+ str(nextState)
        assert len(reward[0]) == 1
        assert len(fall[0]) == 1
        assert len(G_t[0]) == 1
        assert len(exp_action[0]) == 1
        """
        state = list(state)
        action = list(action)
        nextState = list(nextState)
        reward = list(reward)
        nums = state+action+nextState+reward
        """
        
        if ( checkValidData(state, action, nextState, reward) == False ):
            print ("Failed inserting bad tuple: ")
            return
        
        if ( (self._history_update_index % (self._history_size-1) ) == 0):
            self._history_update_index=0
            # print("Reset history index in exp buffer:")
        
        # print ("Tuple: " + str(state) + ", " + str(action) + ", " + str(nextState) + ", " + str(reward))
        # print ("action type: ", self._action_history.dtype)
        self._state_history[self._history_update_index] = copy.deepcopy(np.array(state))
        self._action_history[self._history_update_index] = copy.deepcopy(np.array(action))
        # print("inserted action: ", self._action_history[self._history_update_index])
        self._nextState_history[self._history_update_index] = copy.deepcopy(np.array(nextState))
        self._reward_history[self._history_update_index] = copy.deepcopy(np.array(reward))
        self._fall_history[self._history_update_index] = copy.deepcopy(np.array(fall))
        self._discounted_sum_history[self._history_update_index] = copy.deepcopy(np.array(G_t))
        self._advantage_history[self._history_update_index] = copy.deepcopy(np.array(advantage))
        self._exp_action_history[self._history_update_index] = copy.deepcopy(np.array(exp_action))
        # print ("fall: ", fall)
        # print ("self._fall_history: ", self._fall_history[self._history_update_index])
        
        self._inserts+=1
        self._history_update_index+=1
        self.updateScalling(state, action, nextState, reward)
        
        
    def samples(self):
        return self._inserts
    
    def history_size(self):
        return self._history_size
    
    def updateScalling(self, state, action, nextState, reward):
        
        if (self.samples() == 1):
            self._state_mean =  self._state_history[0]
            self._state_var = np.zeros_like(state)
            
            self._reward_mean =  self._reward_history[0]
            self._reward_var = np.zeros_like(reward)
            
            self._action_mean =  self._action_history[0]
            self._action_var = np.zeros_like(action)
        else:
            x_mean_old = self._state_mean
            self._state_mean = self._state_mean + ((state - self._state_mean)/self.samples())
            
            reward_mean_old = self._reward_mean
            self._reward_mean = self._reward_mean + ((reward - self._reward_mean)/self.samples())
            
            action_mean_old = self._action_mean
            self._action_mean = self._action_mean + ((action - self._action_mean)/self.samples())
        
        if ( self.samples() == 2):
            self._state_var = (self._state_history[1] - ((self._state_history[0]+self._state_history[1])/2.0)**2)/2.0
            self._reward_var = (self._reward_history[1] - ((self._reward_history[0]+self._reward_history[1])/2.0)**2)/2.0
            self._action_var = (self._action_history[1] - ((self._action_history[0]+self._action_history[1])/2.0)**2)/2.0
            
        elif (self.samples() > 2):
            self._state_var = (((self.samples()-2)*self._state_var) + ((self.samples()-1)*(x_mean_old - self._state_mean)**2) + ((state - self._state_mean)**2))
            self._state_var = (self._state_var/float(self.samples()-1))
            
            self._reward_var = (((self.samples()-2)*self._reward_var) + ((self.samples()-1)*(reward_mean_old - self._reward_mean)**2) + ((reward - self._reward_mean)**2))
            self._reward_var = (self._reward_var/float(self.samples()-1))
            
            self._action_var = (((self.samples()-2)*self._action_var) + ((self.samples()-1)*(action_mean_old - self._action_mean)**2) + ((action - self._action_mean)**2))
            self._action_var = (self._action_var/float(self.samples()-1))
            
        if ( 'state_normalization' in self._settings and self._settings["state_normalization"] == "adaptive"):
            self._updateScaling()
            
    def _updateScaling(self):
        
        scale_factor = 1.0
        # state_std = np.maximum(np.sqrt(self._state_var[0]), 0.05)
        state_std = np.sqrt(self._state_var[0])
        # print("Running mean: ", self._state_mean)
        # print("Running std: ", state_std)
        low = self._state_mean[0] - (state_std*scale_factor)
        high = self._state_mean[0] + (state_std*scale_factor)
        # self.setStateBounds(np.array([low,high]))
        self.setStateBounds(fixBounds(np.array([low,high])))
        
        # print("New scaling parameters: ", self.getStateBounds())
        
        # print("Running reward mean: ", self._reward_mean)
        # print("Running reward std: ", np.sqrt(self._reward_var))
        low = self._reward_mean[0] - (np.sqrt(self._reward_var[0])*scale_factor)
        high = self._reward_mean[0] + (np.sqrt(self._reward_var[0])*scale_factor)
        self.setRewardBounds(np.array([low,high]))
        # print("New scaling parameters: ", self.getStateBounds())
        """
        low = self._action_mean[0] - np.sqrt(self._action_var[0])
        high = self._action_mean[0] + np.sqrt(self._action_var[0])
        self.setActionBounds(np.array([low,high]))
        """
        
    def get_exporation_action_batch(self, batch_size=32):
        return self.get_batch(batch_size=batch_size, excludeActionTypes=[0])
    
    def getNonMBAEBatch(self, batch_size=32):
        """
            Avoids training critic on MBAE actions.
        """ 
        return self.get_batch(batch_size=batch_size, excludeActionTypes=[2])
            
    def get_batch(self, batch_size=32, excludeActionTypes=[]):
        """
        len(experience > batch_size
        """
        # assert batch_size <= self._history_size, "batch_size <= self._history_size: " + str(batch_size) +" <=  " + str(self._history_size)
        assert batch_size <= self.samples(), "batch_size <= self.samples(): " + str(batch_size) +" <=  " + str(self.samples())
        # indices = list(nprnd.randint(low=0, high=len(experience), size=batch_size))
        max_size = min(self._history_size, self.samples())
        # print ("Indicies: " , indices)
        # print("Exp buff state bounds: ", self.getStateBounds())

        state = []
        action = []
        resultState = []
        reward = []
        fall = []
        G_ts = []
        exp_actions = []
        advantage = []
        indices = set([])
        trys = 0
        ### collect batch and try at most 3 times the batch size for valid tuples
        while len(indices) <  batch_size and (trys < batch_size*3):
        # for i in indices:
            trys = trys + 1
            i = (random.sample(set(range(0, max_size))-indices, 1))[0]
            ## skip tuples that were not exploration actions
            if ( self._exp_action_history[i] in excludeActionTypes):
                continue
            indices.add(i)
            
            if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
                # state.append(self._state_history[i])
                state.append(norm_state(self._state_history[i], self.getStateBounds()))
                # print("Action pulled out: ", self._action_history[i])
                action.append(self._action_history[i]) # won't work for discrete actions...
                # action.append(norm_action(self._action_history[i], self.getActionBounds())) # won't work for discrete actions...
                resultState.append(norm_state(self._nextState_history[i], self.getResultStateBounds()))
                # resultState.append(self._nextState_history[i])
                reward.append(norm_state(self._reward_history[i] , self.getRewardBounds()) * ((1.0-self._settings['discount_factor']))) # scale rewards
            else:
                                
                state.append(norm_state(self._state_history[i], self.getStateBounds()))
                # print("Action pulled out: ", self._action_history[i])
                action.append(norm_action(self._action_history[i], self.getActionBounds())) # won't work for discrete actions...
                resultState.append(norm_state(self._nextState_history[i], self.getResultStateBounds()))
                reward.append(norm_state(self._reward_history[i] , self.getRewardBounds() ) * ((1.0-self._settings['discount_factor']))) # scale rewards
            fall.append(self._fall_history[i])
            G_ts.append(self._discounted_sum_history[i])
            advantage.append(self._advantage_history[i])
            exp_actions.append(self._exp_action_history[i])
            
        # print c
        # print experience[indices]
        if (self._settings['float_type'] == 'float32'):
            state = np.array(state, dtype='float32')
            if (self._continuous_actions):
                action = np.array(action, dtype='float32')
            else:
                action = np.array(action, dtype='int8')
            resultState = np.array(resultState, dtype='float32')
            reward = np.array(reward, dtype='float32')
            # fall = np.array(fall, dtype='int8')
            G_ts = np.array(G_ts, dtype='float32')
            advantage = np.array(advantage, dtype='float32')
        else:
            state = np.array(state, dtype='float64')
            if (self._continuous_actions):
                action = np.array(action, dtype='float64')
            else:
                action = np.array(action, dtype='int8')
            resultState = np.array(resultState, dtype='float64')
            reward = np.array(reward, dtype='float64')
            G_ts = np.array(G_ts, dtype='float64')
            advantage = np.array(advantage, dtype='float32')
        
        fall = np.array(fall, dtype='int8')
        exp_actions = np.array(exp_actions, dtype='int8')
        
        assert state.shape == (len(indices), self._state_length), "state.shape == (len(indices), self._state_length): " + str(state.shape) + " == " + str((len(indices), self._state_length))
        assert action.shape == (len(indices), self._action_length), "action.shape == (len(indices), self._action_length): " + str(action.shape) + " == " + str((len(indices), self._action_length))
        assert resultState.shape == (len(indices), self._result_state_length), "resultState.shape == (len(indices), self._result_state_length): " + str(resultState.shape) + " == " + str((len(indices), self._result_state_length))
        assert reward.shape == (len(indices), 1), "reward.shape == (len(indices), 1): " + str(reward.shape) + " == " + str((len(indices), 1))
        assert G_ts.shape == (len(indices), 1), "G_ts.shape == (len(indices), 1): " + str(G_ts.shape) + " == " + str((len(indices), 1))
        assert fall.shape == (len(indices), 1), "fall.shape == (len(indices), 1): " + str(fall.shape) + " == " + str((len(indices), 1))
        assert exp_actions.shape == (len(indices), 1), "exp_actions.shape == (len(indices), 1): " + str(exp_actions.shape) + " == " + str((len(indices), 1))
        assert advantage.shape == (len(indices), 1), "G_ts.shape == (len(indices), 1): " + str(advantage.shape) + " == " + str((len(indices), 1))
        # assert len(np.unique(indices)[0]) == batch_size, "np.unique(indices).shape[0] == batch_size: " + str(np.unique(indices).shape[0]) + " == " + str(batch_size)
        
        return (state, action, resultState, reward, fall, G_ts, exp_actions, advantage)
    
    def setStateBounds(self, _state_bounds):
        self._state_bounds = _state_bounds
        self.setResultStateBounds(_state_bounds)
        
    def setRewardBounds(self, _reward_bounds):
        self._reward_bounds = _reward_bounds
    def setActionBounds(self, _action_bounds):
        self._action_bounds = _action_bounds
    def setResultStateBounds(self, _result_state_bounds):
        self._result_state_bounds = _result_state_bounds
        
    def getStateBounds(self):
        return self._state_bounds
    def getRewardBounds(self):
        return self._reward_bounds
    def getActionBounds(self):
        return self._action_bounds
    def getResultStateBounds(self):
        return self._result_state_bounds
    
    def setSettings(self, settings):
        self._settings = settings
    def getSettings(self):
        return self._settings
    
    def saveToFile(self, filename):
        hf = h5py.File(filename, "w")
        hf.create_dataset('_state_history', data=self._state_history)
        hf.create_dataset('_action_history', data=self._action_history)
        hf.create_dataset('_next_state_history', data=self._nextState_history)
        hf.create_dataset('_reward_history', data=self._reward_history)
        hf.create_dataset('_fall_history', data=self._fall_history)
        hf.create_dataset('_discounted_sum_history', data=self._discounted_sum_history)
        hf.create_dataset('_advantage_history', data=self._advantage_history)
        hf.create_dataset('_exp_action_history', data=self._exp_action_history)
        
        hf.create_dataset('_history_size', data=[self._history_size])
        hf.create_dataset('_history_update_index', data=[self._history_update_index])
        hf.create_dataset('_inserts', data=[self._inserts])
        hf.create_dataset('_state_length', data=[self._state_length])
        hf.create_dataset('_action_length', data=[self._action_length])
        hf.create_dataset('_state_bounds', data=self._state_bounds)
        hf.create_dataset('_reward_bounds', data=self._reward_bounds)
        hf.create_dataset('_action_bounds', data=self._action_bounds)
        hf.create_dataset('_result_state_bounds', data=self._result_state_bounds)
        
        hf.flush()
        hf.close()
        
    def loadFromFile(self, filename):
        hf = h5py.File(filename,'r')
        self._state_history = np.array(hf.get('_state_history'))
        self._action_history= np.array(hf.get('_action_history'))
        self._nextState_history = np.array(hf.get('_next_state_history'))
        self._reward_history = np.array(hf.get('_reward_history'))
        self._fall_history = np.array(hf.get('_fall_history'))
        self._discounted_sum_history = np.array(hf.get('_discounted_sum_history'))
        self._advantage_history = np.array(hf.get('_advantage_history'))
        self._exp_action_history = np.array(hf.get('_exp_action_history'))
        
        self._history_size = int(hf.get('_history_size')[()])
        self._history_update_index = int(hf.get('_history_update_index')[()])
        self._inserts = int(hf.get('_inserts')[()])
        self._state_length = int(hf.get('_state_length')[()])
        self._action_length = int(hf.get('_action_length')[()])
        self._state_bounds = np.array(hf.get('_state_bounds'))
        self._reward_bounds = np.array(hf.get('_reward_bounds'))
        self._action_bounds = np.array(hf.get('_action_bounds'))
        self._result_state_bounds = np.array(hf.get('_result_state_bounds'))
        
        hf.close()
        