
import numpy as np
import random
import h5py
from model.ModelUtil import validBounds, fixBounds, anneal_value, norm_state, norm_action, norm_reward, checkValidData, action_bound_std
import copy
import sys


class ExperienceMemory(object):
    """
        Contains the recient history of experience tuples
         
        I have decided that the experience memory will contain real values from the simulation.
        Not values that have been normalize. I think this will make things easier down the road
        If I wanted to adjust the model scale now I won't have to update all the tuples in the memory.
        Also, a scale layer can be added to the model to compensate for having to scale every tuple
        when performing training updates.
    """
    
    def __init__(self, state_length, action_length, memory_length, continuous_actions=False, settings=None, result_state_length=None,
                 use_dense_results_state=None):
        
        if (settings == None):
            self._settings = {}
            self._settings['discount_factor'] = 0.0
            # self._settings['float_type'] = 'float32'
        else:
            self._settings = settings
        
        self._use_dense_results_state=use_dense_results_state
        self._history_size=memory_length
        self._trajectory_size=int(np.mean(memory_length)/100)
        if ("fd_experience_length" in self._settings):
            self._trajectory_size=int(self._settings["fd_experience_length"])
        self._state_length = state_length
        self._action_length = action_length
        self._continuous_actions = continuous_actions
        
        if ( result_state_length == None ):
            self._result_state_length = state_length
        else:
            self._result_state_length = result_state_length
        self._history_update_index=0 # where the next experience should write
        self._samples=0 # Number of inserts since last clear()
        self._inserts=0 # total number of inserts
        if self._continuous_actions:
            self._action_bounds = np.array(settings["action_bounds"])
        self.clear()
        
    def clear(self):
        self._history_update_index=0 # where the next experience should write
        self._samples=0 ## How many samples are in the buffer
        
        if (self._settings['float_type'] == 'float32'):
            self._state_history = (np.zeros((self._history_size, self._state_length), dtype='float32'))
            if self._continuous_actions:
                self._action_history = (np.zeros((self._history_size, self._action_length), dtype='float32'))
            else:
                self._action_history = (np.zeros((self._history_size, self._action_length), dtype='int8'))
            self._nextState_history = (np.zeros((self._history_size, self._result_state_length), dtype='float32'))
            self._reward_history = (np.zeros((self._history_size, 1), dtype='float32'))
            self._discounted_sum_history = (np.zeros((self._history_size, 1), dtype='float32'))
            self._advantage_history = (np.zeros((self._history_size, 1), dtype='float32'))
        else:
            self._state_history = (np.zeros((self._history_size, self._state_length), dtype='float64'))
            if self._continuous_actions:
                self._action_history = (np.zeros((self._history_size, self._action_length), dtype='float64'))
            else:
                self._action_history = (np.zeros((self._history_size, self._action_length), dtype='int8'))
            self._nextState_history = (np.zeros((self._history_size, self._result_state_length), dtype='float64'))
            self._reward_history = (np.zeros((self._history_size, 1), dtype='float64'))
            self._discounted_sum_history = (np.zeros((self._history_size, 1), dtype='float64'))
            self._advantage_history = (np.zeros((self._history_size, 1), dtype='float64'))
            
        self._fall_history = (np.zeros((self._history_size, 1), dtype='int8'))
        """
        if ("perform_multiagent_training" in self._settings):
            self._exp_action_history = (np.zeros((self._history_size, self._settings["perform_multiagent_training"]), dtype='int8'))
        else:
        """
        self._exp_action_history = (np.zeros((self._history_size, 1), dtype='int8'))
        
        self._data = {
            "agent_id": [[0]] * self._history_size, 
            "task_id": [[0]] * self._history_size}
        
        
            
        self._trajectory_history = [None] * self._trajectory_size
        self._samplesTrajectory = 0
        self._insertsTrajectory = 0
        self._trajectory_update_index = 0
        
    def insertsTrajectory(self):
        return self._insertsTrajectory
    def samplesTrajectory(self):
        return self._samplesTrajectory
    
    def history_size_Trajectory(self):
        return self._trajectory_size
            
    def _insertTrajectory(self, trajectory):
        
        if ( (self._trajectory_update_index >= (self.history_size_Trajectory()) ) ):
            self._trajectory_update_index=0
            # print("Reset history index in exp buffer:")
            
        self._trajectory_history[self._trajectory_update_index] = trajectory
        
        self._insertsTrajectory+=1
        self._trajectory_update_index+=1
        self._samplesTrajectory+=1
        
    def getTrajectory(self, i):
        """
            Insert a trajectory as a collection of sequences
        """
        return self._trajectory_history[i]
        
    def insertTrajectory(self, states, actions, result_states, rewards, falls, G_ts, advantage, exp_actions, data):
        """
            Insert a trajectory as a collection of sequences
        """
        assert len(states[0]) == self._state_length
        assert len(actions[0]) == self._action_length
        
        self._insertTrajectory([states, actions, result_states, rewards, falls, G_ts, advantage, exp_actions, data])
        
    def get_multitask_trajectory_batch(self, batch_size=4, excludeActionTypes=[], randomLength=False, randomStart=False,
                                       max_length=32):
        
        state_, action_, resultState_, reward_, fall_, G_ts_, exp_actions_, advantage_, data_ = self.get_trajectory_batch(batch_size=batch_size, cast=False)
        
        ### Find length of shortest trajectory...
        min_seq_length = 1
        if ("min_sequece_length" in self._settings):
            min_seq_length = self._settings["min_sequece_length"] + 1
        shortest_traj = 100000000
        traj_start = 0
        for t in range(len(state_)):
            if len(state_[t]) < shortest_traj:
                shortest_traj = max(len(state_[t]), min_seq_length)
                
        ### Choose a random time to start
        if (randomStart == True):
            inds = range(0, max(1, shortest_traj- min_seq_length))
            ### plus one so because of index count mismatch.
            if (np.random.random() > 0.5):
                traj_start = np.random.choice(inds, p=np.array(list(reversed(inds)), dtype='float64')/np.sum(inds))
            else:
                traj_start = random.sample(set(inds), 1)[0]
             
        ### Choose a random time for trajectory to end
        inds = range(traj_start + min_seq_length, shortest_traj)
        if ( ( randomLength == True )
            and (shortest_traj > traj_start + min_seq_length)):  
#                 ### shortest_traj Must be at least 2 for this to return 1
#                 ### Make shorter sequence more probable
            shortest_traj = np.random.choice(inds, p=np.array(list(reversed(inds)), dtype='float64')/np.sum(inds))
        
        if ((shortest_traj - traj_start) > max_length):
            ### Things tend to run out of memory beyond this.
            shortest_traj = traj_start + max_length
        ### Make all trajectories as long as the shortest one...
        for t in range(len(state_)):
            if (len(state_[t]) < min_seq_length):
                continue
            state_[t] = state_[t][traj_start:shortest_traj]
            action_[t] = action_[t][traj_start:shortest_traj]
            # print ("resultState_[t]: ", np.array(resultState_[t]).shape)
            resultState_[t] = resultState_[t][traj_start:shortest_traj]
            # print ("resultState_[t] after: ", np.array(resultState_[t]).shape)
            reward_[t] = reward_[t][traj_start:shortest_traj]
            fall_[t] = fall_[t][traj_start:shortest_traj]
            G_ts_[t] = G_ts_[t][traj_start:shortest_traj]
            exp_actions_[t] = exp_actions_[t][traj_start:shortest_traj]
            advantage_[t] = advantage_[t][traj_start:shortest_traj]
            for key in data_[t]:
                data_[t][key] = data_[t][key][traj_start:shortest_traj]
            
        state_ = np.array(state_, dtype=self._settings['float_type'])
        if (self._continuous_actions):
            action_ = np.array(action_, dtype=self._settings['float_type'])
        else:
            action_ = np.array(action_, dtype='int8')
        resultState_ = np.array(resultState_, dtype=self._settings['float_type'])
        reward_ = np.array(reward_, dtype=self._settings['float_type'])
        G_ts_ = np.array(G_ts_, dtype=self._settings['float_type'])
        advantage_ = np.array(advantage_, dtype=self._settings['float_type'])
        
        fall_ = np.array(fall_, dtype='int8')
        exp_actions_ = np.array(exp_actions_, dtype='int8')
            
        return (state_, action_, resultState_, reward_, fall_, G_ts_, exp_actions_, advantage_, data_)
        
    def get_trajectory_batch(self, batch_size=4, excludeActionTypes=[], cast=True):
        """
        len(experience > batch_size
        """
        # assert batch_size <= self._history_size, "batch_size <= self._history_size: " + str(batch_size) +" <=  " + str(self._history_size)
        assert batch_size <= self.samplesTrajectory(), "batch_size <= self.samplesTrajectory(): " + str(batch_size) +" <=  " + str(self.samplesTrajectory())
        # indices = list(nprnd.randint(low=0, high=len(experience), size=batch_size))
        max_size = min(self.history_size_Trajectory(), self.samplesTrajectory())
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
        datas = []
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
            assert self._trajectory_history[i] != None, "self._trajectory_history["+str(i)+"] != None: " + str(self._trajectory_history[i]) + " state shape: " + str(np.asarray(state).shape)
            # print ("states shape: ", np.array(self._trajectory_history[i][0]))
            # print ("states bounds shape: ", np.array(self.getStateBounds()))
            state.append(norm_state(self._trajectory_history[i][0], self.getStateBounds()))
            # print("Action pulled out: ", self._action_history[i])
            action.append(norm_action(self._trajectory_history[i][1], self.getActionBounds())) # won't work for discrete actions...
            resultState.append(norm_state(self._trajectory_history[i][2], self.getResultStateBounds()))
            # reward.append(norm_state(self._trajectory_history[i][3] , self.getRewardBounds() ) * ((1.0-self._settings['discount_factor']))) # scale rewards
            reward.append(self._trajectory_history[i][3] / action_bound_std(self.getRewardBounds()) * ((1.0-self._settings['discount_factor']))) # scale rewards
            fall.append(self._trajectory_history[i][4])
            G_ts.append(norm_state(self._trajectory_history[i][5], self.getRewardBounds()) * ((1.0-self._settings['discount_factor'])))
            advantage.append(self._trajectory_history[i][6])
            exp_actions.append(self._trajectory_history[i][7])
            datas.append(copy.deepcopy(self._trajectory_history[i][8]))
            
        # print c
        # print experience[indices]
        ### All sequences must be the same length for this to work
        if (cast):
            state = np.array(state, dtype=self._settings['float_type'])
            if (self._continuous_actions):
                action = np.array(action, dtype=self._settings['float_type'])
            else:
                action = np.array(action, dtype='int8')
            resultState = np.array(resultState, dtype=self._settings['float_type'])
            reward = np.array(reward, dtype=self._settings['float_type'])
            G_ts = np.array(G_ts, dtype=self._settings['float_type'])
            advantage = np.array(advantage, dtype=self._settings['float_type'])
            
            fall = np.array(fall, dtype='int8')
            exp_actions = np.array(exp_actions, dtype='int8')
        
        # assert state.shape == (len(indices), self._state_length), "state.shape == (len(indices), self._state_length): " + str(state.shape) + " == " + str((len(indices), self._state_length))
        # assert action.shape == (len(indices), self._action_length), "action.shape == (len(indices), self._action_length): " + str(action.shape) + " == " + str((len(indices), self._action_length))
        # assert resultState.shape == (len(indices), self._result_state_length), "resultState.shape == (len(indices), self._result_state_length): " + str(resultState.shape) + " == " + str((len(indices), self._result_state_length))
        # assert reward.shape == (len(indices), 1), "reward.shape == (len(indices), 1): " + str(reward.shape) + " == " + str((len(indices), 1))
        # assert G_ts.shape == (len(indices), 1), "G_ts.shape == (len(indices), 1): " + str(G_ts.shape) + " == " + str((len(indices), 1))
        # assert fall.shape == (len(indices), 1), "fall.shape == (len(indices), 1): " + str(fall.shape) + " == " + str((len(indices), 1))
        # assert exp_actions.shape == (len(indices), 1), "exp_actions.shape == (len(indices), 1): " + str(exp_actions.shape) + " == " + str((len(indices), 1))
        # assert advantage.shape == (len(indices), 1), "G_ts.shape == (len(indices), 1): " + str(advantage.shape) + " == " + str((len(indices), 1))
        # assert len(np.unique(indices)[0]) == batch_size, "np.unique(indices).shape[0] == batch_size: " + str(np.unique(indices).shape[0]) + " == " + str(batch_size)
        
        return (state, action, resultState, reward, fall, G_ts, exp_actions, advantage, datas)
        
        
    def insertTuple(self, tuple):
        
        (state, action, nextState, reward, fall, G_t, exp_action, advantage, data) = tuple
        self.insert(state, action, nextState, reward, fall, G_t, exp_action, advantage, data)
        
    def insert(self, state, action, nextState, reward, fall=[[0]], G_t=[[0]], exp_action=[[0]], advantage=[[0]], data={}):
#         print ("Instert ", self.inserts(), " State: ", str(state))
        # state = list(state)
        # print ("state shape: ", np.array(state).shape)
        if ("use_hack_state_trans" in self.getSettings()
            and (self.getSettings()["use_hack_state_trans"] == True)):
            state = np.array(state)
            state = state[:,:len(self.getStateBounds()[0])]
            nextState = np.array(nextState)
            nextState = nextState[:,:len(self.getResultStateBounds()[0])]
        assert self._state_length == len(state[0]), "self._state_length == len(state[0]): " + str(self._state_length) + " state shape: " + str(np.asarray(state).shape)
        if self._continuous_actions: 
            assert len(action[0]) == self._action_length, "len(action[0]) == self._action_length: " + str(len(action[0])) + " == " + str(self._action_length)
        assert len(nextState[0]) == self._result_state_length, "len(nextState[0]) == self._result_state shape: " + str(np.asarray(nextState).shape) + " == " + str(self._result_state_length)
        assert len(reward[0]) == 1
        assert len(fall[0]) == 1
        assert len(G_t[0]) == 1
        assert len(exp_action[0]) == 1, "len(exp_action[0]) == 1: " + str(np.asarray(exp_action[0]).shape) + " == " + str(1)

        if ( checkValidData(state, action, nextState, reward, verbose=True) == False ):
            print ("Skip inserting bad tuple: ")
            return
        
        if ( (self._history_update_index >= (self._history_size) )):
            self._history_update_index=0
            # print("Reset history index in exp buffer:")
        
        # print ("Tuple: " + str(state) + ", " + str(action) + ", " + str(nextState) + ", " + str(reward))
#         print ("state: ", str(state) )
#         print ("nextState: ", str(nextState) )
        # print ("action type: ", self._action_history.dtype)
        self._state_history[self._history_update_index] = copy.deepcopy(np.array(state))
        self._action_history[self._history_update_index] = copy.deepcopy(np.array(action))
        # print("inserted action: ", self._action_history[self._history_update_index])
        self._nextState_history[self._history_update_index] = copy.deepcopy(np.array(nextState))
#         print ("nextState2: ", self._nextState_history[self._history_update_index]) 
        self._reward_history[self._history_update_index] = copy.deepcopy(np.array(reward))
        self._fall_history[self._history_update_index] = copy.deepcopy(np.array(fall))
        self._discounted_sum_history[self._history_update_index] = copy.deepcopy(np.array(G_t))
        self._advantage_history[self._history_update_index] = copy.deepcopy(np.array(advantage))
        self._exp_action_history[self._history_update_index] = copy.deepcopy(np.array(exp_action))

        skip = ["rendering"]
        for key in data:
            if key in skip: ## Skip some things...
                continue
            if key not in self._data:
                 self._data[key] =  [[0]] * self._history_size
            self._data[key][self._history_update_index] = data[key]
        
        self._inserts+=1
        self._history_update_index+=1
        self._samples+=1
        self.updateScalling(state, action, nextState, reward)
        
    def inserts(self):
        return self._inserts
    def samples(self):
        return self._samples
    
    def history_size(self):
        return self._history_size
    
    def updateScalling(self, state, action, nextState, reward):
        assert self.inserts() > 0
        if (self.inserts() == 1):
            self._state_mean =  self._state_history[0]
            self._state_var = np.ones_like(state)
            
            self._reward_mean =  self._reward_history[0]
            self._reward_var = np.ones_like(reward)
            
            self._action_mean =  self._action_history[0]
            self._action_var = np.ones_like(action)
        else:
            x_mean_old = self._state_mean
            self._state_mean = self._state_mean + ((state - self._state_mean)/self.inserts())
            
            reward_mean_old = self._reward_mean
            self._reward_mean = self._reward_mean + ((reward - self._reward_mean)/self.inserts())
            
            action_mean_old = self._action_mean
            self._action_mean = self._action_mean + ((action - self._action_mean)/self.inserts())
        
        if ( self.inserts() == 2):
            self._state_var = (self._state_history[1] - ((self._state_history[0]+self._state_history[1])/2.0)**2)/2.0
            self._reward_var = (self._reward_history[1] - ((self._reward_history[0]+self._reward_history[1])/2.0)**2)/2.0
            self._action_var = (self._action_history[1] - ((self._action_history[0]+self._action_history[1])/2.0)**2)/2.0
            
        elif (self.inserts() > 2):
            self._state_var = (((self.inserts()-2)*self._state_var) + ((self.inserts()-1)*(x_mean_old - self._state_mean)**2) + ((state - self._state_mean)**2))
            self._state_var = (self._state_var/float(self.inserts()-1))
            
            self._reward_var = (((self.inserts()-2)*self._reward_var) + ((self.inserts()-1)*(reward_mean_old - self._reward_mean)**2) + ((reward - self._reward_mean)**2))
            self._reward_var = (self._reward_var/float(self.inserts()-1))
            
            self._action_var = (((self.inserts()-2)*self._action_var) + ((self.inserts()-1)*(action_mean_old - self._action_mean)**2) + ((action - self._action_mean)**2))
            self._action_var = (self._action_var/float(self.inserts()-1))
            
        self._state_var = np.fabs(self._state_var)
        self._reward_var = np.fabs(self._reward_var)
        self._action_var = np.fabs(self._action_var)
        # if ( 'state_normalization' in self._settings and self._settings["state_normalization"] == "adaptive"):
        #     self._updateScaling()
            
    def _updateScaling(self):
        
#         print ("_updateScaling self.inserts(): ", self.inserts())
        if self.inserts() < 5:
            return
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
        if ("update_adaptive_reward_normalization" in self.getSettings()
            and (self.getSettings()["update_adaptive_reward_normalization"] == False)):
            pass
        else:
            self.setRewardBounds(np.array([low,high]))
        # print("New scaling parameters: ", self.getStateBounds())
        """
        low = self._action_mean[0] - np.sqrt(self._action_var[0])
        high = self._action_mean[0] + np.sqrt(self._action_var[0])
        self.setActionBounds(np.array([low,high]))
        """
        
    def get_exporation_action_batch(self, batch_size=32):
        if ("Use_fast_batch_computation" in self.getSettings()
            and (self.getSettings()["Use_fast_batch_computation"] == True)):
            return self.get_batch_fast(batch_size=batch_size, excludeActionTypes=[0])
        else:
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
        assert max_size >= batch_size, "max_size >= batch_size " + str(max_size) + " >= " + str(batch_size) + " self._history_size, self.samples()" + str(self._history_size) + "," + str(self.samples())
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
        data = {}
        for key in self._data:
            data[key] = []
        trys = 0
        ### collect batch and try at most 5 times the batch size for valid tuples
        while len(indices) <  batch_size and (trys < batch_size*5):
        # for i in indices:
            trys = trys + 1
            i = (random.sample(set(range(0, max_size))-indices, 1))[0]
            ## skip tuples that were not exploration actions
            if ( self._exp_action_history[i] in excludeActionTypes):
                continue
            ### Or if multitasking and only want to train policy on single task
            # print ("self._fall_history[i]: ", self._fall_history[i])
            if ( (type(self._settings["sim_config_file"]) is list)):
                 
                if  (not ("multitask_learning" in self._settings
                          and (self._settings["multitask_learning"] == True))
                     ):
                     
                    if ("ask_env_for_multitask_id" in self._settings 
                        and (self._settings["ask_env_for_multitask_id"] == True)
                        and (self._fall_history[i][0] != 0)): 
                        # print ("Skipping: ", self._fall_history[i][0])
                        continue
                    if ("ask_env_for_multitask_id" in self._settings 
                        and (self._settings["ask_env_for_multitask_id"] == "multi_task")
                        and (self._fall_history[i][0] != 0)): 
                        ## Don't skip.
                        pass
                    elif ("worker_to_task_mapping" in self._settings
                        and (self._settings["worker_to_task_mapping"][self._fall_history[i][0]] != 0)): ### Only use training data for the task of interest
                    # print ("skipping non desired task tuple")
                        continue
            indices.add(i)
            
                             
#             print ("self.getStateBounds(): ", np.array(self.getStateBounds()).shape)
#             print ("self._state_history[i]: ", self._state_history[i])   
            state.append(norm_state(self._state_history[i], self.getStateBounds()))
#             print("State pulled out: ", self._state_history[i])
            # print ("self.getActionBounds(): ", self.getActionBounds())
            if (self._continuous_actions):
                action.append(norm_action(self._action_history[i], self.getActionBounds())) # won't work for discrete actions...
            else:
                action.append(self._action_history[i])
#             print ("self.getResultStateBounds(): ", np.array(self.getResultStateBounds()).shape)
#             print ("self._nextState_history[i]: ", self._nextState_history[i])
            resultState.append(norm_state(self._nextState_history[i], self.getResultStateBounds()))
            # reward.append(norm_state(self._reward_history[i] , self.getRewardBounds() ) * ((1.0-self._settings['discount_factor']))) # scale rewards
            reward.append(self._reward_history[i] / action_bound_std(self.getRewardBounds()) * ((1.0-self._settings['discount_factor']))) # scale rewards
            # action_bound_std(self.getRewardBounds())
            fall.append(self._fall_history[i])
            G_ts.append(self._discounted_sum_history[i]/ action_bound_std(self.getRewardBounds()) * ((1.0-self._settings['discount_factor'])))
            # print ("G_ts Before: ", self._discounted_sum_history[i], " reward bounds: ", self.getRewardBounds(), " normalized: ", norm_state(self._discounted_sum_history[i], self.getRewardBounds()))
            # print ("after: ", norm_state(self._discounted_sum_history[i], self.getRewardBounds()) * (1.0-self._settings['discount_factor']) )
            # print ("G_ts before, after: ", np.concatenate((self._discounted_sum_history[i], norm_state(self._discounted_sum_history[i], self.getRewardBounds()) * (1.0-self._settings['discount_factor'])), axis=1))
            advantage.append(self._advantage_history[i])
            exp_actions.append(self._exp_action_history[i])
            
            for key in self._data:
                data[key].append(self._data[key][i])
            
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
        
        assert len(indices) > 0, "empty batch"
        assert state.shape == (len(indices), self._state_length), "state.shape == (len(indices), self._state_length): " + str(state.shape) + " == " + str((len(indices), self._state_length))
        assert action.shape == (len(indices), self._action_length), "action.shape == (len(indices), self._action_length): " + str(action.shape) + " == " + str((len(indices), self._action_length))
        assert resultState.shape == (len(indices), self._result_state_length), "resultState.shape == (len(indices), self._result_state_length): " + str(resultState.shape) + " == " + str((len(indices), self._result_state_length))
        assert reward.shape == (len(indices), 1), "reward.shape == (len(indices), 1): " + str(reward.shape) + " == " + str((len(indices), 1))
        assert G_ts.shape == (len(indices), 1), "G_ts.shape == (len(indices), 1): " + str(G_ts.shape) + " == " + str((len(indices), 1))
        assert fall.shape == (len(indices), 1), "fall.shape == (len(indices), 1): " + str(fall.shape) + " == " + str((len(indices), 1))
        assert exp_actions.shape == (len(indices), 1), "exp_actions.shape == (len(indices), 1): " + str(exp_actions.shape) + " == " + str((len(indices), 1))
        assert advantage.shape == (len(indices), 1), "G_ts.shape == (len(indices), 1): " + str(advantage.shape) + " == " + str((len(indices), 1))
        # assert len(np.unique(indices)[0]) == batch_size, "np.unique(indices).shape[0] == batch_size: " + str(np.unique(indices).shape[0]) + " == " + str(batch_size)
        
        return (state, action, resultState, reward, fall, G_ts, exp_actions, advantage, data)
    
    def get_batch_fast(self, batch_size=32, excludeActionTypes=[]):
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
        # while len(indices) <  batch_size and (trys < batch_size*5):
        indices = (random.sample(range(0, max_size), batch_size))
        for i in indices:
            ## skip tuples that were not exploration actions
            if ( self._exp_action_history[i] in excludeActionTypes):
                continue
            ### Or if multitasking and only want to train policy on single task
            # print ("self._fall_history[i]: ", self._fall_history[i])
            if ( (type(self._settings["sim_config_file"]) is list)):
                 
                if (
                    (not ("multitask_learning" in self._settings
                          and (self._settings["multitask_learning"] == True))
                     )
                    and
                    ("worker_to_task_mapping" in self._settings
                     and (self._settings["worker_to_task_mapping"][self._fall_history[i][0]] is not 0))
                    ): ### Only use training data for the task of interest
                    # print ("skipping non desired task tuple")
                    continue
            # indices.add(i)
            
            if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
                state.append(norm_state(self._state_history[i], self.getStateBounds()))
                action.append(self._action_history[i]) # won't work for discrete actions...
                resultState.append(norm_state(self._nextState_history[i], self.getResultStateBounds()))
                reward.append(self._reward_history[i] / action_bound_std(self.getRewardBounds()) * ((1.0-self._settings['discount_factor']))) # scale rewards
            else:
                                
                state.append(norm_state(self._state_history[i], self.getStateBounds()))
                action.append(norm_action(self._action_history[i], self.getActionBounds())) # won't work for discrete actions...
                resultState.append(norm_state(self._nextState_history[i], self.getResultStateBounds()))
                reward.append(self._reward_history[i] / action_bound_std(self.getRewardBounds()) * ((1.0-self._settings['discount_factor']))) # scale rewards
            fall.append(self._fall_history[i])
            G_ts.append(self._discounted_sum_history[i]/ action_bound_std(self.getRewardBounds()) * ((1.0-self._settings['discount_factor'])))
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
        
        """
        assert state.shape == (len(indices), self._state_length), "state.shape == (len(indices), self._state_length): " + str(state.shape) + " == " + str((len(indices), self._state_length))
        assert action.shape == (len(indices), self._action_length), "action.shape == (len(indices), self._action_length): " + str(action.shape) + " == " + str((len(indices), self._action_length))
        assert resultState.shape == (len(indices), self._result_state_length), "resultState.shape == (len(indices), self._result_state_length): " + str(resultState.shape) + " == " + str((len(indices), self._result_state_length))
        assert reward.shape == (len(indices), 1), "reward.shape == (len(indices), 1): " + str(reward.shape) + " == " + str((len(indices), 1))
        assert G_ts.shape == (len(indices), 1), "G_ts.shape == (len(indices), 1): " + str(G_ts.shape) + " == " + str((len(indices), 1))
        assert fall.shape == (len(indices), 1), "fall.shape == (len(indices), 1): " + str(fall.shape) + " == " + str((len(indices), 1))
        assert exp_actions.shape == (len(indices), 1), "exp_actions.shape == (len(indices), 1): " + str(exp_actions.shape) + " == " + str((len(indices), 1))
        assert advantage.shape == (len(indices), 1), "G_ts.shape == (len(indices), 1): " + str(advantage.shape) + " == " + str((len(indices), 1))
        """
        # assert len(np.unique(indices)[0]) == batch_size, "np.unique(indices).shape[0] == batch_size: " + str(np.unique(indices).shape[0]) + " == " + str(batch_size)
        
        return (state, action, resultState, reward, fall, G_ts, exp_actions, advantage)
    
    def setStateBounds(self, _state_bounds):
        assert len(_state_bounds[0]) == self._state_length, "len(_state_bounds[0]) == self._state_length: " + str(len(_state_bounds[0])) + " == " + str(self._state_length)
        self._state_bounds = np.array(_state_bounds)
        if (self._use_dense_results_state is None):
            self.setResultStateBounds(_state_bounds)
        # self._state_length = len(self.getStateBounds()[0])
        
    def setRewardBounds(self, _reward_bounds):
        self._reward_bounds = np.array(_reward_bounds)
        
    def setActionBounds(self, _action_bounds):
        if ("policy_connections" in self._settings
            and (any([self._settings["agent_id"] == m[1] for m in self._settings["policy_connections"]])) ):
            pass
        else:
            assert len(_action_bounds[0]) == self._action_length
            self._action_bounds = np.array(_action_bounds)
    def setResultStateBounds(self, _result_state_bounds):
#         assert len(_result_state_bounds[0]) == self._result_state_length, "len(_result_state_bounds[0]) == self._result_state_length: " + str(len(_result_state_bounds[0])) + " == " + str(self._result_state_length)
        self._result_state_bounds = np.array(_result_state_bounds)
        # self._result_state_length = len(self.getResultStateBounds()[0])
        
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
        hf.create_dataset('_samples', data=[self._samples])
        hf.create_dataset('_state_length', data=[self._state_length])
        hf.create_dataset('_action_length', data=[self._action_length])
        hf.create_dataset('_result_state_length', data=[self._result_state_length])
        hf.create_dataset('_state_bounds', data=self._state_bounds)
        hf.create_dataset('_reward_bounds', data=self._reward_bounds)
        hf.create_dataset('_action_bounds', data=np.array(self._action_bounds))
        hf.create_dataset('_result_state_bounds', data=self._result_state_bounds)
        
        ### Adaptive scaling values
        hf.create_dataset('_state_mean', data=self._state_mean)
        hf.create_dataset('_state_var', data=self._state_var)
        hf.create_dataset('_reward_mean', data=self._reward_mean)
        hf.create_dataset('_reward_var', data=self._reward_var)
        hf.create_dataset('_action_mean', data=self._action_mean)
        hf.create_dataset('_action_var', data=self._action_var)
        
        grp_d = hf.create_group('datas')
        for key in self._data:
            # print ("key: ", key, " value: ", self._data[key])
            try: 
                grp_d.create_dataset(str(key),data=np.array(self._data[key]))
            except:
                print ("Can not convert string object to numpy array, skipping storing data for key ", str(key))
                pass
            
        
        ### Save a variable length list of data
        # data = np.array(self._trajectory_history, dtype=object)
        if (((("train_LSTM_FD" in self._settings)
            and (self._settings["train_LSTM_FD"] == True))
            or
            (("train_LSTM_Reward" in self._settings)
            and (self._settings["train_LSTM_Reward"] == True))
            )
            and 
            self._settings["save_experience_memory"] != "continual"
            ):
            grp = hf.create_group('trajectories')
            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                print ("Saving trajectory data")
            for i in range(min(self.history_size_Trajectory(), self.samplesTrajectory())):
                list = self._trajectory_history[i]
                # print (i,list)
                if (list is not None):
                    grp_ = grp.create_group('traj'+str(i))
                    for it in range(len(list)):
                        ### This might be storing renderings, taking up a lot of space.
                        grp_.create_dataset(str(it),data=np.array(list[it]))
                else:
                    break
                
            hf.create_dataset('_trajectory_size', data=[self._trajectory_size])
            hf.create_dataset('_trajectory_update_index', data=[self._trajectory_update_index])
            hf.create_dataset('_insertsTrajectory', data=[self._insertsTrajectory])
            hf.create_dataset('_samplesTrajectory', data=[self._samplesTrajectory])
        
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
        self._samples = int(hf.get('_samples')[()])
        self._state_length = int(hf.get('_state_length')[()])
        self._action_length = int(hf.get('_action_length')[()])
        self._result_state_length = int(hf.get('_result_state_length')[()])
        self._state_bounds = np.array(hf.get('_state_bounds'))
        self._reward_bounds = np.array(hf.get('_reward_bounds'))
        self._action_bounds = np.array(hf.get('_action_bounds'))
        self._result_state_bounds = np.array(hf.get('_result_state_bounds'))
        
        ### Adaptive scaling values
        self._state_mean = np.array(hf.get('_state_mean'))
        self._state_var = np.array(hf.get('_state_var'))
        self._reward_mean = np.array(hf.get('_reward_mean'))
        self._reward_var = np.array(hf.get('_reward_var'))
        self._action_mean = np.array(hf.get('_action_mean'))
        self._action_var = np.array(hf.get('_action_var'))
        
        # print ("self._state_mean: ", self._state_mean)
        grp_d = hf.get('datas')
        for key in grp_d.keys():
            self._data[str(key)] = np.array(grp_d.get(str(key)))
            print ("loading key: ", key, " value: ", self._data[str(key)])
        
        if (((("train_LSTM_FD" in self._settings)
                and (self._settings["train_LSTM_FD"] == True))
            or
            (("train_LSTM_Reward" in self._settings)
                and (self._settings["train_LSTM_Reward"] == True))
            )
            and
            self._settings["save_experience_memory"] != "continual"
            ):
            self._trajectory_size = int(hf.get('_trajectory_size')[()])
            self._trajectory_update_index = int(hf.get('_trajectory_update_index')[()])
            self._insertsTrajectory = int(hf.get('_insertsTrajectory')[()])
            self._samplesTrajectory = int(hf.get('_samplesTrajectory')[()])
            
            
            grp = hf.get('trajectories')
            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                print ("Loading trajectory data")
            for i in range(min(self.history_size_Trajectory(), self.samplesTrajectory())):
                # print (i)
                traj = []
                grp_ = grp.get('traj'+str(i))
                for it in range(8):
                    traj.append(np.array(grp_.get(str(it))))
            
                self._trajectory_history[i] = traj
            print ("num trajectories: ", min(self.history_size_Trajectory(), self.samplesTrajectory()))
            
        hf.close()
        