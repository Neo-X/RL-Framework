
import theano
# from theano import tensor as T
import numpy as np
import random
import h5py
from model.ModelUtil import *


class ExperienceMemory(object):
    """
        Contains the recient history of experience tuples
         
        I have decided that the experience memory will contain real values from the simulation.
        Not values that have been normalize. I think this will make things easy down the road
        If I wanted to adjust the model scale now I won't have to update all the tuples in the memory.
        Also, a scale layer can be added to the model to compensate for having to scale every tuple
        when performing training updates.
    """
    
    def __init__(self, state_length, action_length, memory_length, continuous_actions=False, settings=None):
        
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
        # self._settings = settings
        self.clear()
        # self._state_history = theano.shared(np.zeros((self._history_size, state_length)))
        # self._action_history = theano.shared(np.zeros((self._history_size, action_length)))
        # self._nextState_history = theano.shared(np.zeros((self._history_size, state_length)))
        # self._reward_history = theano.shared(np.zeros((self._history_size, 1)))
        
    def clear(self):
        self._history_update_index=0 # where the next experience should write
        self._inserts=0
        
        if (self._settings['float_type'] == 'float32'):
            
            self._state_history = (np.zeros((self._history_size, self._state_length), dtype='float32'))
            if self._continuous_actions:
                self._action_history = (np.zeros((self._history_size, self._action_length), dtype='float32'))
            else:
                self._action_history = (np.zeros((self._history_size, self._action_length), dtype='int8'))
            self._nextState_history = (np.zeros((self._history_size, self._state_length), dtype='float32'))
            self._reward_history = (np.zeros((self._history_size, 1), dtype='float32'))
            self._fall_history = (np.zeros((self._history_size, 1), dtype='int8'))
            self._discounted_sum_history = (np.zeros((self._history_size, 1), dtype='float32'))
        else:
            self._state_history = (np.zeros((self._history_size, self._state_length), dtype='float64'))
            if self._continuous_actions:
                self._action_history = (np.zeros((self._history_size, self._action_length), dtype='float64'))
            else:
                self._action_history = (np.zeros((self._history_size, self._action_length), dtype='int8'))
            self._nextState_history = (np.zeros((self._history_size, self._state_length), dtype='float64'))
            self._reward_history = (np.zeros((self._history_size, 1), dtype='float64'))
            self._fall_history = (np.zeros((self._history_size, 1), dtype='int8'))
            self._discounted_sum_history = (np.zeros((self._history_size, 1), dtype='float64'))
        
    def insertTuple(self, tuple):
        
        (state, action, nextState, reward, fall, G_t) = tuple
        self.insert(state, action, nextState, reward, fall, G_t)
        
    def checkValidData(self, state, action, nextState, reward):
        """
            Checks to make sure the data going into the exp buffer is not garbage...
        
        """
        if (not np.all(np.isfinite(state))):
            less_ = np.isfinite(state)
            bad_indecies = np.where(less_ == False)
            bad_values_ = state[bad_indecies]
            print ("State not finite: ", np.isfinite(state) )
            print ("Bad Value indx: ", bad_indecies)
            print ("Bad Values: ", bad_values_)
            return False
        
        if (not np.all(np.isfinite(action))):
            less_ = np.isfinite(action)
            bad_indecies = np.where(less_ == False)
            bad_values_ = action[bad_indecies]
            print ("Action not finite: ", np.isfinite(state) )
            print ("Bad Value indx: ", bad_indecies)
            print ("Bad Values: ", bad_values_)
            return False
        
        if (not np.all(np.isfinite(nextState))):
            less_ = np.isfinite(state)
            bad_indecies = np.where(less_ == False)
            bad_values_ = nextState[bad_indecies]
            print ("NextState not finite: ", np.isfinite(state) )
            print ("Bad Value indx: ", bad_indecies)
            print ("Bad Values: ", bad_values_)
            return False
        
        if (not np.all(np.isfinite(reward))):
            less_ = np.isfinite(reward)
            bad_indecies = np.where(less_ == False)
            bad_values_ = reward[bad_indecies]
            print ("Reward not finite: ", np.isfinite(state) )
            print ("Bad Value indx: ", bad_indecies)
            print ("Bad Values: ", bad_values_)
            return False
        
        if (np.any(np.less(state, -1000.0))):
            less_ = np.less(state, -1000.0)
            bad_indecies = np.where(less_ == True)
            bad_values_ = state[bad_indecies]
            print ("State too negative: ", np.less(state, -1000.0) )
            print ("Bad Value indx: ", bad_indecies)
            print ("Bad Values: ", bad_values_)
            return False
        
        if (np.any(np.less(nextState, -1000.0))):
            less_ = np.less(nextState, -1000.0)
            bad_indecies = np.where(less_ == True)
            bad_values_ = nextState[bad_indecies]
            print ("nextState too negative: ", np.less(nextState, -1000.0) )
            print ("Bad Value indx: ", bad_indecies)
            print ("Bad Values: ", bad_values_)
            return False
        
        if (np.any(np.greater(state, 1000.0))):
            less_ = np.greater(state, 1000.0)
            bad_indecies = np.where(less_ == True)
            bad_values_ = state[bad_indecies]
            print ("State too positive: ", np.greater(state, 1000.0) )
            print ("Bad Value indx: ", bad_indecies)
            print ("Bad Values: ", bad_values_)
            return False
        
        if (np.any(np.greater(nextState, 1000.0))):
            less_ = np.greater(nextState, 1000.0)
            bad_indecies = np.where(less_ == True)
            bad_values_ = nextState[bad_indecies]
            print ("nextState too positive: ", np.greater(nextState, 1000.0) )
            print ("Bad Value indx: ", bad_indecies)
            print ("Bad Values: ", bad_values_)
            return False
        
        return True
        
    def insert(self, state, action, nextState, reward, fall=[[0]], G_t=[[0]]):
        # print "Instert State: " + str(state)
        # state = list(state)
        
        """
        state = list(state)
        action = list(action)
        nextState = list(nextState)
        reward = list(reward)
        nums = state+action+nextState+reward
        """
        
        if ( self.checkValidData(state, action, nextState, reward) == False ):
            print ("Failed inserting: ")
            return
        
        if ( (self._history_update_index % (self._history_size-1) ) == 0):
            self._history_update_index=0
        
        # print "Tuple: " + str(state) + ", " + str(action) + ", " + str(nextState) + ", " + str(reward)
        self._state_history[self._history_update_index] = np.array(state)
        self._action_history[self._history_update_index] = np.array(action)
        self._nextState_history[self._history_update_index] = np.array(nextState)
        self._reward_history[self._history_update_index] = np.array(reward)
        self._fall_history[self._history_update_index] = np.array(fall)
        self._discounted_sum_history[self._history_update_index] = np.array(G_t)
        # print ("fall: ", fall)
        # print ("self._fall_history: ", self._fall_history[self._history_update_index])
        
        self._inserts+=1
        self._history_update_index+=1
        
        
    def samples(self):
        return self._inserts
    
    def history_size(self):
        return self._history_size
        
    def get_batch(self, batch_size=32):
        """
        len(experience > batch_size
        """
        # indices = list(nprnd.randint(low=0, high=len(experience), size=batch_size))
        indices = (random.sample(range(0, min(self._history_size, self.samples())), batch_size))
        # print ("Indicies: " , indices)

        state = []
        action = []
        resultState = []
        reward = []
        fall = []
        G_ts = []
        # scale_state(self._state_history[i], self._state_bounds)
        for i in indices:
            state.append(norm_state(self._state_history[i], self._state_bounds))
            action.append(norm_action(self._action_history[i], self._action_bounds)) # won't work for discrete actions...
            resultState.append(norm_state(self._nextState_history[i], self._state_bounds))
            reward.append(self._reward_history[i] * ((1.0-self._settings['discount_factor']))) # scale rewards
            fall.append(self._fall_history[i])
            G_ts.append(self._discounted_sum_history[i])
            
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
        else:
            state = np.array(state, dtype='float64')
            if (self._continuous_actions):
                action = np.array(action, dtype='float64')
            else:
                action = np.array(action, dtype='int8')
            resultState = np.array(resultState, dtype='float64')
            reward = np.array(reward, dtype='float64')
            G_ts = np.array(G_ts, dtype='float64')
        
        fall = np.array(fall, dtype='int8')
         
        return (state, action, resultState, reward, fall, G_ts)
    
    def setStateBounds(self, _state_bounds):
        self._state_bounds = _state_bounds
    def setRewardBounds(self, _reward_bounds):
        self._reward_bounds = _reward_bounds
    def setActionBounds(self, _action_bounds):
        self._action_bounds = _action_bounds
    def setSettings(self, settings):
        self._settings = settings
    
    def saveToFile(self, filename):
        hf = h5py.File(filename, "w")
        hf.create_dataset('_state_history', data=self._state_history)
        hf.create_dataset('_action_history', data=self._action_history)
        hf.create_dataset('_next_state_history', data=self._nextState_history)
        hf.create_dataset('_reward_history', data=self._reward_history)
        hf.create_dataset('_fall_history', data=self._fall_history)
        hf.create_dataset('_discounted_sum_history', data=self._discounted_sum_history)
        
        hf.create_dataset('_history_size', data=[self._history_size])
        hf.create_dataset('_history_update_index', data=[self._history_update_index])
        hf.create_dataset('_inserts', data=[self._inserts])
        hf.create_dataset('_state_length', data=[self._state_length])
        hf.create_dataset('_action_length', data=[self._action_length])
        hf.create_dataset('_state_bounds', data=self._state_bounds)
        hf.create_dataset('_reward_bounds', data=self._reward_bounds)
        hf.create_dataset('_action_bounds', data=self._action_bounds)
        
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
        
        self._history_size = int(hf.get('_history_size')[()])
        self._history_update_index = int(hf.get('_history_update_index')[()])
        self._inserts = int(hf.get('_inserts')[()])
        self._state_length = int(hf.get('_state_length')[()])
        self._action_length = int(hf.get('_action_length')[()])
        self._state_bounds = np.array(hf.get('_state_bounds'))
        self._reward_bounds = np.array(hf.get('_reward_bounds'))
        self._action_bounds = np.array(hf.get('_action_bounds'))
        
        hf.close()
        