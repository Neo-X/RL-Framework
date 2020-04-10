"""
    An interface class for Agents to be used in the system.

"""
from multiprocessing import Process, Queue
# from pathos.multiprocessing import Pool
import threading
import time
from model.AgentInterface import AgentInterface
from model.ModelUtil import *
from util.utils import rlPrint
import os
import copy
import time
import datetime
# np.set_printoptions(threshold=np.nan)

class LearningAgent(AgentInterface):
    
    def __init__(self, settings_):
        super(LearningAgent,self).__init__(n_in=None, n_out=None, state_bounds=None, 
                                           action_bounds=None, reward_bound=None, settings_=settings_)
        self._useLock = False
        if self._useLock:
            self._accesLock = threading.Lock()
        self._pol = None
        self._fd = None
        self._sampler = None
        self._expBuff = None
        self._expBuff_FD = None
        
    def reset(self):
        self.getPolicy().reset()
        if (self.getSettings()['train_forward_dynamics']):
            self.getForwardDynamics().reset()
        
    def getPolicy(self):
        pol = self._pol
        return pol
        
    def setPolicy(self, pol):
        self._pol = pol
        if (not self._sampler == None ):
            self._sampler.setPolicy(pol)
        
    def getForwardDynamics(self):
        fd = self._fd
        return fd
                
    def setForwardDynamics(self, fd):
        self._fd = fd
            
    def getRewardModel(self):
        rm = self._rm
        return rm
                
    def setRewardModel(self, rm):
        self._rm = rm
        
    def getPolicyNetworkParameters(self):
        return self.getPolicy().getNetworkParameters()
    def getFDNetworkParameters(self):
        return self.getForwardDynamics().getNetworkParameters()
    def getRewardNetworkParameters(self):
        return self.getRewardModel().getNetworkParameters()
    
    def setPolicyNetworkParameters(self, params):
        return self.getPolicy().setNetworkParameters(params)
    def setFDNetworkParameters(self, params):
        return self.getForwardDynamics().setNetworkParameters(params)
    def setRewardNetworkParameters(self, params):
        return self.getRewardModel().setNetworkParameters(params)
        
    def setExperience(self, experienceBuffer):
        self._expBuff = experienceBuffer 
    def getExperience(self):
        return self._expBuff
    
    def setFDExperience(self, experienceBuffer):
        self._expBuff_FD = experienceBuffer 
    def getFDExperience(self):
        return self._expBuff_FD  
    
    def putDataInExpMem(self, _states, _actions, _rewards, _result_states, _falls, _advantage=None, 
              _exp_actions=None, _G_t=None, datas=None, recomputeRewards=False):
        import numpy as np
        num_samples_ = 0
        tmp_states = []
        tmp_actions = []
        tmp_result_states = [] 
        tmp_rewards = []
        tmp_falls = []
        tmp_G_t = []
        tmp_advantage = []
        tmp_exp_action = []
        tmp_datas = []
        ### Causes the new scaling values to be computed but not applied. They are applied later after the updates
        self.getExperience()._settings["state_normalization"] = "variance"
        for (state__, action__, next_state__, reward__, fall__, G_t__, exp_action__, advantage__,
              datas__) in zip(_states, _actions, _result_states, _rewards, _falls, _G_t,
                               _exp_actions, _advantage, datas):

            ### Because the valid state checks only like numpy arrays, not lists
            state___ = state__
            next_state___ = next_state__
            
            if ("use_dual_state_representations" in self._settings
                        and (self._settings["use_dual_state_representations"] == True)):
                state___ = [s[1] for s in state__]
                next_state___ = [ns[1] for ns in next_state__]
            ### Validate data
            if (checkValidData(state___, action__, next_state___, reward__, verbose=True) and 
                checkDataIsValid(advantage__, verbose=True), checkDataIsValid(G_t__, verbose=True)):
                
                if (recomputeRewards==True):
                    path = {}
                    ### timestep, agent, state
                    path["terminated"] = False
                
                    # print ("imitation_traj shape: ", imitation_traj.shape)
                    # print ("previous reward__: ", reward__)
                    if ("use_learned_reward_function" in self._settings
                        and (self._settings["use_learned_reward_function"] == "ic2")):
                        reward__ = self.getForwardDynamics().predict_reward(state___, action__)
                    else:
                        agent_traj = np.array([np.array([np.array(np.array(tmp_states__[1]), dtype=self._settings['float_type']) for tmp_states__ in state__])])
                        # print ("agent_traj shape: ", agent_traj.shape)
                        imitation_traj = np.array([np.array([np.array(np.array(tmp_states__[1]), dtype=self._settings['float_type']) for tmp_states__ in next_state__])])
                        reward__ = -self.getForwardDynamics().predict_reward_(agent_traj, imitation_traj)
                    w_d = -2.0
                    if ("learned_reward_function_norm_weight" in self._settings):
                        w_d = self._settings["learned_reward_function_norm_weight"]
                    # reward__ = np.exp((reward__*reward__)*w_d)
                    # print ("reward__", reward__)
                    path['states'] = state__ # np.array([np.array(np.array(tmp_states__[0]), dtype=self._settings['float_type']) for tmp_states__ in state__])
                    path['reward'] = reward__
                    path['falls'] = fall__
                    path['agent_id'] = datas__['agent_id']
                    # print ("state__ shape: ", path['states'].shape)
                    paths = compute_advantage_(self, [path], self._settings["discount_factor"], self._settings['GAE_lambda'])
                    advantage__ = paths["advantage"]
                    # baselines_.append(np.array(paths["baseline"]))
                    # advantage.append(np.array(adv__))
                
                tmp_states.append(state__)
                tmp_actions.append(action__)
                tmp_result_states.append(next_state__)
                tmp_rewards.append(reward__)
                tmp_falls.append(fall__)
                tmp_advantage.append(advantage__)
                tmp_exp_action.append(exp_action__)
                tmp_G_t.append(G_t__)
                tmp_datas.append(datas__)
                
                ### Data is a trajectory
                for j in range(len(state__)):
                    data___ = {}
                    for key in datas__:
                        # print ("key: ", key, " datas__", datas__[key])
                        data___[key] = datas__[key][j] 
                    
                    tup = ([state__[j]], [action__[j]], [next_state__[j]], [reward__[j]], [fall__[j]], [G_t__[j]], [exp_action__[j]], [advantage__[j]], data___)
                    if ("use_dual_state_representations" in self._settings
                        and (self._settings["use_dual_state_representations"] == True)):
                        if ("use_viz_for_policy" in self._settings 
                                and self._settings["use_viz_for_policy"] == True):
                            tup = ([state__[j][1]], [action__[j]], [next_state__[j][1]], [reward__[j]], [fall__[j]], [G_t__[j]], [exp_action__[j]], [advantage__[j]], data___)
                        elif ("use_dual_viz_state_representations" in self._settings
                            and (self._settings["use_dual_viz_state_representations"] == True)):
                            tup = ([state__[j][0]], [action__[j]], [next_state__[j][0]], [reward__[j]], [fall__[j]], [G_t__[j]], [exp_action__[j]], [advantage__[j]], data___)
                        else:
                            tup = ([state__[j][0]], [action__[j]], [next_state__[j][0]], [reward__[j]], [fall__[j]], [G_t__[j]], [exp_action__[j]], [advantage__[j]], data___)
                    self.getExperience().insertTuple(tup)
                    if ( 'keep_seperate_fd_exp_buffer' in self._settings and (self._settings['keep_seperate_fd_exp_buffer'])):
                        if ("use_dual_state_representations" in self._settings
                        and (self._settings["use_dual_state_representations"] == True)):
                            if ("use_viz_for_policy" in self._settings 
                                and self._settings["use_viz_for_policy"] == True):
                                ### Want viz for input and dense for output to condition the preception part of the network
                                tup = ([state__[j][1]], [action__[j]], [next_state__[j][0]], [reward__[j]], [fall__[j]], [G_t__[j]], [exp_action__[j]], [advantage__[j]], data___)
                            else:
                                # print ("self.getFDExperience().getStateBounds() shape : ", self.getFDExperience().getStateBounds())
                                # print ("fd exp state shape: ", state__[j][1].shape)
                                tup = ([state__[j][1]], [action__[j]], [next_state__[j][1]], [reward__[j]], [fall__[j]], [G_t__[j]], [exp_action__[j]], [advantage__[j]], data___)
                        ### This is always done and works well for computing the adaptive state bounds.
                        self.getFDExperience().insertTuple(tup)
                    num_samples_ = num_samples_ + 1
                    
        return ( num_samples_, (tmp_states, tmp_actions, tmp_result_states, tmp_rewards, tmp_falls, tmp_G_t, tmp_advantage, tmp_exp_action, tmp_datas))
        
    def recomputeRewards(self, _states, _actions, _rewards, _result_states, _falls, _advantage, 
              _exp_actions, _G_t, datas):
        """
            While learning a reward function re compute the rewards after performing critic and fd updates before policy update
        """
        self.putDataInExpMem(_states, _actions, _rewards, _result_states, _falls, _advantage, _exp_actions, _G_t, datas, recomputeRewards=True)
        
    def applyHER(self, _states, _actions, _rewards, _result_states, _falls, _advantage, 
              _exp_actions, _G_t, datas):
        import numpy as np
        ### Hindsight Experience Replay
        ## Add Trajectories that achieved some goal
        ### For each trajectory add another trajectory with a modified goal

        hindsight_relabel_probability = 1.0 if not "hindsight_relabel_probability" in self.getSettings() else self.getSettings()["hindsight_relabel_probability"]

        new_states = []
        new_result_states = []
        new_rewards = []
        new_actions = []
        new_falls = []
        new_advantage = []
        new_exp_actions = []
        new_G_t = []

        trajectories = len(_states)
        for traj in range(trajectories):
            ### Get the new goal(s) for the trajectory
            # new_goals = [x for x in copy.deepcopy(_result_states[traj])]
            states = np.array(copy.deepcopy(_states[traj]))
            new_goals = []
            for g in range(len(_result_states[traj])):
                g_index = ((int(g / self._settings["hlc_timestep"])+1) * self._settings["hlc_timestep"]) -1
                g_index = min(g_index, len(_result_states[traj])-1)
                original_goal = states[g, -self._settings["goal_slice_index"]:]
                if np.random.uniform() < hindsight_relabel_probability:
                    new_goals.append(_result_states[traj][g_index][:self._settings["goal_slice_index"]])
                else:
                    new_goals.append(original_goal)

            ### Copy in the new goals
            new_goals = np.array(new_goals)
            result_states = np.array(copy.deepcopy(_result_states[traj]))
            states[:,-self._settings["goal_slice_index"]:] = new_goals
            result_states[:,-self._settings["goal_slice_index"]:] = new_goals
            rewards = copy.deepcopy(_rewards[traj])
            diff = result_states[:,:self._settings["goal_slice_index"]] - new_goals
            rewards = -np.sum(diff * diff, axis=-1, keepdims=True)
            
            new_states.append(states)
            new_result_states.append(result_states)
            new_rewards.append(rewards)
            new_actions.append(_actions[traj])
            new_falls.append(_falls[traj])
            new_advantage.append(_advantage[traj])
            new_exp_actions.append(_exp_actions[traj])
            new_G_t.append(_G_t[traj])
            
            """
                achieved_goal = result_states___[-1][0, :self._settings["goal_slice_index"]]
                states[jj][0, ...] = np.concatenate([
                    states[jj][0, :self._settings["goal_slice_index"]],
                    achieved_goal], 0)
                result_states___[jj][0, ...] = np.concatenate([
                    result_states___[jj][0, :self._settings["goal_slice_index"]],
                    achieved_goal], 0)
                ### Basic version of reward function is indicator of reached goal threshold
                rewards = (rewards * 0) + -1
                rewards[-1] = [1]
            """
        return (new_states, new_actions, new_rewards, new_result_states, new_falls, new_advantage,
              new_exp_actions, new_G_t, datas)
                
    def getAgents(self):
        return [self]
    # @profile(precision=5)
    def train(self, _states, _actions, _rewards, _result_states, _falls, _advantage=None, 
              _exp_actions=None, _G_t=None, p=1.0, datas=None, trainInfo=None):
        if self._useLock:
            self._accesLock.acquire()
        loss = 0
        critic_loss = 0
        loss_actor = 0
        import numpy as np
        from util.SimulationUtil import logExperimentData
        
        if ("value_function_batch_size" in self._settings):
            value_function_batch_size = self._settings['value_function_batch_size']
        else:
            value_function_batch_size = self._settings["batch_size"]
        if self._settings['on_policy']:
            if ( ('clear_exp_mem_on_poli' in self._settings) 
                 and (self._settings['clear_exp_mem_on_poli'] == True)):
                self.getExperience().clear()
            
            ### Update target networks
            self.updateTargetModel()
            num_samples_=1
            t0 = time.time()
            
            if ("use_hindsight_relabeling" in self._settings and
                self._settings["use_hindsight_relabeling"] and
                "goal_slice_index" in self._settings):
                (_states, _actions, _rewards, _result_states, _falls, _advantage, 
                  _exp_actions, _G_t) = self.applyHER(_states, _actions, _rewards, _result_states, _falls, _advantage, 
                  _exp_actions, _G_t, datas)
            
            ( num_samples_, (tmp_states, tmp_actions, tmp_result_states, tmp_rewards, tmp_falls, tmp_G_t, tmp_advantage, tmp_exp_action, tmp_datas)) = self.putDataInExpMem(_states, _actions, _rewards, _result_states, _falls, _advantage, 
              _exp_actions, _G_t, datas)
            
            ## Update scaling values
            ### Updating the scaling values after the update(s) will help make things more accurate
            if ('state_normalization' in self._settings and (self._settings["state_normalization"] == "adaptive")):
                self.getExperience()._updateScaling()
                self.setStateBounds(self.getExperience().getStateBounds())
                if (self._settings["action_space_continuous"]):
                    self.setActionBounds(self.getExperience().getActionBounds())
                self.setRewardBounds(self.getExperience().getRewardBounds())
                if ( ('keep_seperate_fd_exp_buffer' in self._settings and (self._settings['keep_seperate_fd_exp_buffer'] == True))
                     ):
                    self.getFDExperience()._updateScaling()
                    if ( ('train_forward_dynamics' in self._settings and (self._settings['train_forward_dynamics'] == True))):
                        self.getForwardDynamics().setStateBounds(self.getFDExperience().getStateBounds())
                        if (self._settings["action_space_continuous"]):
                            self.getForwardDynamics().setActionBounds(self.getFDExperience().getActionBounds())
                        self.getForwardDynamics().setRewardBounds(self.getFDExperience().getRewardBounds())
                if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                    print("Learner, Scaling State params: ", self.getStateBounds())
                    print("Learner, Scaling Action params: ", self.getActionBounds())
                    print("Learner, Scaling Reward params: ", self.getRewardBounds())
            
            batch_size_ = self._settings["batch_size"]        
            if (self._settings["batch_size"] == "all"):
                batch_size_ = max(self._expBuff.samples(), 1)
                # print ("num_samples_: ", num_samples_)
            ### If for some reason the data was all garbage, skip this training update.
            if ((self._expBuff.samples() < value_function_batch_size 
                or (self._expBuff.samples() < batch_size_))
                and
                (not ("skip_rollouts" in self._settings and 
                        (self._settings["skip_rollouts"] == True)))):
                print("Data was mostly/all garbage or your batch size is larger than the data collected.", self._expBuff.samples(),
                      " batch size ", batch_size_, 
                      " value func batch size ", value_function_batch_size)
                return 0
            t1 = time.time()
            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                sim_time_ = datetime.timedelta(seconds=(t1-t0))
                print ("Pushing data into exp buffer complete in " + str(sim_time_) + " seconds")
                        
            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):        
                print ("self._expBuff.samples(): ", self.getExperience().samples(), " states.shape: ", np.array(_states).shape)
                if ( 'keep_seperate_fd_exp_buffer' in self._settings and (self._settings['keep_seperate_fd_exp_buffer'])):
                    print ("self.getFDExperience().samples(): ", self.getFDExperience().samples())
                # print ("exp_actions sum: ", np.sum(tmp_exp_action))
            
            # if (((len(tmp_states) > 0 ) or (self._settings))
            #     and (
            if ((         
                     (("train_LSTM" in self._settings)
                        and (self._settings["train_LSTM"] == True))
                     or
                     (("train_LSTM_Critic" in self._settings)
                        and (self._settings["train_LSTM_Critic"] == True))
                     or 
                     (("train_LSTM_FD" in self._settings)
                        and (self._settings["train_LSTM_FD"] == True))
                     or
                     (("train_LSTM_Reward" in self._settings)
                        and (self._settings["train_LSTM_Reward"] == True))
                     )
                ):
                
                ### Need to normalize data
                __states = _states 
                __actions = _actions
                __rewards = _rewards 
                __result_states = _result_states
                __falls = _falls 
                __advantage = _advantage
                __exp_actions = _exp_actions
                __G_t = _G_t
                __datas = datas
                
                _states = []
                _result_states = []
                _states_fd = []
                _result_states_fd = []
                
                for i in range(len(tmp_states)):
                    if ("use_dual_state_representations" in self.getSettings()
                        and (self.getSettings()["use_dual_state_representations"] == True)):
                        _states.append([np.array(np.array(tmp_states__[0]), dtype=self._settings['float_type']) for tmp_states__ in tmp_states[i]])
                        _result_states.append([np.array(np.array(tmp_result_states__[0]), dtype=self._settings['float_type']) for tmp_result_states__ in tmp_result_states[i]])
                        
                        # print ("use_dual_state_representations")
                        _states_fd.append([np.array(np.array(tmp_states__[1]), dtype=self._settings['float_type']) for tmp_states__ in tmp_states[i]])
                        _result_states_fd.append([np.array(np.array(tmp_result_states__[1]), dtype=self._settings['float_type']) for tmp_result_states__ in tmp_result_states[i]])
                    else:
                        _states = tmp_states
                        _result_states = tmp_result_states
                        
                        _states_fd = _states
                        _result_states_fd = _result_states
                    
                if ((("train_LSTM" in self._settings)
                    and (self._settings["train_LSTM"] == True))
                    or
                     (("train_LSTM_Critic" in self._settings)
                        and (self._settings["train_LSTM_Critic"] == True))
                    ):
                    for e in range(len(_states)):
                        
                        self.getExperience().insertTrajectory(_states[e], tmp_actions[e], _result_states[e], tmp_rewards[e], 
                                                                    tmp_falls[e], tmp_G_t[e], tmp_advantage[e], tmp_exp_action[e], tmp_datas[e])
                    batch_size_lstm = 4
                    if ("lstm_batch_size" in self._settings):
                        batch_size_lstm = self._settings["lstm_batch_size"][1]
                    updates___ = max(1, int((len(_states)/batch_size_lstm) * self._settings["additional_on_policy_training_updates"]))
                    print ("Performing lstm policy training")
                    if (("train_LSTM_Critic" in self._settings)
                        and (self._settings["train_LSTM_Critic"] == True)
                        and self._settings['train_critic']):
                        for e in range(updates___):  
                            for cu in range(self._settings["critic_updates_per_actor_update"]): 
                                states_, actions_, result_states_, rewards_, falls_, G_ts_, exp_actions_, advantages_, datas_ = self.getExperience().get_multitask_trajectory_batch(batch_size=min(batch_size_lstm, self.getExperience().samplesTrajectory()))
                                trainInfo["iteration"] = e
                                loss = self.getPolicy().trainCritic(states=states_, actions=actions_, rewards=rewards_, 
                                                             result_states=result_states_, falls=falls_, G_t=G_ts_,
                                                             p=p, datas=datas_, trainInfo=trainInfo)
                                if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                    print("Critic loss: ", loss)
                    if (("train_LSTM" in self._settings)
                        and (self._settings["train_LSTM"] == True)):
                        states_, actions_, result_states_, rewards_, falls_, G_ts_, exp_actions_, advantages_, datas_ = self.getExperience().get_multitask_trajectory_batch(batch_size=min(batch_size_lstm, self.getExperience().samplesTrajectory()))
                        if (self._settings["train_actor"] == True):
                            loss_ = self.getPolicy().trainActor(states=states_, actions=actions_, rewards=rewards_, result_states=result_states_, 
                                                         falls=falls_, advantage=advantages_, exp_actions=exp_actions_, 
                                                         G_t=G_ts_, forwardDynamicsModel=self._fd, p=p,
                                                         datas=datas_, trainInfo=trainInfo)
                            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                print("Policy Loss: ", loss_)
                    print ("Done lstm policy training")
                
                if ((("train_LSTM_FD" in self._settings)
                     and (self._settings["train_LSTM_FD"] == True))
                    or
                    (("train_LSTM_Reward" in self._settings)
                    and (self._settings["train_LSTM_Reward"] == True))
                    ):
                    use_random_sequence_length_for_lstm = False
                    if ("use_random_sequence_length_for_lstm" in self._settings
                        and (self._settings["use_random_sequence_length_for_lstm"] == True)):
                        use_random_sequence_length_for_lstm = True
                    if ("lstm_batch_size" in self._settings):
                        batch_size_lstm_fd = self._settings["lstm_batch_size"][0]
                    for e in range(len(_states_fd)):
                        if ( 'keep_seperate_fd_exp_buffer' in self._settings 
                             and (self._settings['keep_seperate_fd_exp_buffer'])):
                            self.getFDExperience().insertTrajectory(_states_fd[e], tmp_actions[e], _result_states_fd[e], tmp_rewards[e], 
                                                                    tmp_falls[e], tmp_G_t[e], tmp_advantage[e], tmp_exp_action[e], tmp_datas[e])
                            
                    updates___ = max(1, int((len(_states)/batch_size_lstm_fd) * self._settings["fd_updates_per_actor_update"]))
                    if ( "rnn_updates" in self._settings ):
                        updates___ = int(self._settings["rnn_updates"])
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print ("num trajectories: ", len(_states), " num updates: ", updates___)
                
                    batch_size_lstm_fd = 4
                    if ("lstm_batch_size" in self._settings):
                        batch_size_lstm_fd = self._settings["lstm_batch_size"][0]
                        
                    ### Hack so that I can hyper tune without changing the batch_size as well.
                    if ("include_agent_imitator_pairs" in self._settings
                        and (self._settings["include_agent_imitator_pairs"] == True)):
                        batch_size_lstm_fd = batch_size_lstm_fd * 2
                    if (("train_LSTM_FD" in self._settings)
                        and (self._settings["train_LSTM_FD"] == True)):
                        for e in range(updates___):   
                            state_, action_, resultState_, reward_, fall_, G_ts_, exp_actions, advantage_, datas_ = self.getFDExperience().get_multitask_trajectory_batch(batch_size=min(batch_size_lstm_fd, self.getFDExperience().samplesTrajectory()), 
                                                                                                                                                                  randomLength=use_random_sequence_length_for_lstm,
                                                                                                                                                                  randomStart=use_random_sequence_length_for_lstm)
                            trainInfo["iteration"] = e
                            dynamicsLoss = self._fd.train(states=state_, actions=action_, result_states=resultState_, rewards=reward_, datas=datas_, trainInfo=trainInfo)
                            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                print ("Forward Dynamics Loss: ", dynamicsLoss)
                                
                        ### Updates over Multi-task data
                        if (type(self._settings["sim_config_file"]) == list
                            # and False
                            ):
                            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                    print ("Additional Multi-task training fd: ")
                            for e in range(updates___):   
                                state_, action_, resultState_, reward_, fall_, G_ts_, exp_actions, advantage_, datas_ = self.getFDExperience().get_multitask_trajectory_batch(batch_size=min(batch_size_lstm_fd*2, self.getFDExperience().samplesTrajectory()), 
                                                                                                                                                                  randomLength=use_random_sequence_length_for_lstm,
                                                                                                                                                                  randomStart=use_random_sequence_length_for_lstm)
                                trainInfo["iteration"] = e
                                dynamicsLoss = self._fd.train(states=state_, actions=action_, result_states=resultState_, rewards=reward_, falls=fall_, datas=datas_, trainInfo=trainInfo)
                                if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                    print ("Forward Dynamics Loss: ", dynamicsLoss)
                    
                    if (self._settings['train_forward_dynamics']
                        and ("train_LSTM_Reward" in self._settings)
                        and (self._settings["train_LSTM_Reward"] == True)):
                        
                        for e in range(updates___):   
                            state_, action_, resultState_, reward_, fall_, G_ts_, exp_actions, advantage_, datas_ = self.getFDExperience().get_multitask_trajectory_batch(batch_size=min(batch_size_lstm_fd, self.getFDExperience().samplesTrajectory()), 
                                                                                                                                                                  randomLength=use_random_sequence_length_for_lstm,
                                                                                                                                                                  randomStart=use_random_sequence_length_for_lstm)
                            trainInfo["iteration"] = e
                            dynamicsLoss = self._fd.train(states=state_, actions=action_, result_states=resultState_, rewards=reward_, datas=datas_, trainInfo=trainInfo)
                            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                print ("Forward Dynamics Loss: ", dynamicsLoss)
                                
                        ### Updates over Multi-task data
                        if (type(self._settings["sim_config_file"]) == list
                            # and False
                            ):
                            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                print ("Additional Multi-task reward training : ")
                            for e in range(updates___):   
                                state_, action_, resultState_, reward_, fall_, G_ts_, exp_actions, advantage_, datas_ = self.getFDExperience().get_multitask_trajectory_batch(batch_size=min(batch_size_lstm_fd*2, self.getFDExperience().samplesTrajectory()), 
                                                                                                                                                                  randomLength=use_random_sequence_length_for_lstm,
                                                                                                                                                                  randomStart=use_random_sequence_length_for_lstm)
                                trainInfo["iteration"] = e
                                dynamicsLoss = self._fd.train(states=state_, actions=action_, result_states=resultState_, rewards=reward_, falls=fall_, datas=datas_, trainInfo=trainInfo)
                                if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                    print ("Forward Dynamics Loss: ", dynamicsLoss)
                    
                # pass
                # print("Not Falls: ", _falls)
                # print("Rewards: ", _rewards)
                # print ("Actions after: ", _actions)
            if ( "refresh_rewards" in self._settings
                 and (self._settings["refresh_rewards"] == "lstm_fd")):
                rlPrint(self._settings, "train", "Refreshing rewards.")
                self.recomputeRewards(__states, __actions, __rewards, __result_states, __falls, __advantage, 
                                      __exp_actions, __G_t, __datas)
            loss = 0
            additional_on_poli_trianing_updates = 1
            if ( "additional_on_policy_training_updates" in self._settings
                 and (self._settings["additional_on_policy_training_updates"] != False)):
                additional_on_poli_trianing_updates = self._settings["additional_on_policy_training_updates"]
            # if ("perform_multiagent_training" in self._settings): ### Reduce number of updates by agent count
            #     additional_on_poli_trianing_updates = additional_on_poli_trianing_updates / self._settings["perform_multiagent_training"]
            ### The data should be seen ~ 4 times
            additional_on_poli_trianing_updates = int(np.ceil(((self._settings["num_on_policy_rollouts"] * self._settings["max_epoch_length"] * 1) / batch_size_) * additional_on_poli_trianing_updates))
            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                print ("additional_on_poli_trianing_updates: ", additional_on_poli_trianing_updates)
            
            if ( additional_on_poli_trianing_updates < 1 ): ## should have at least one training update
                additional_on_poli_trianing_updates = 1  
                    
            ## This lets the model do most of the training and batching. more efficient
            if ("model_perform_batch_training" in self._settings 
                and (self._settings["model_perform_batch_training"] == True )):
                ### How many more times should the value function see the data
                batch_ratio = value_function_batch_size / batch_size_
                ### Compensate for the ratio collected each run and the size of the replay buffer
                min_samples = self._settings["num_on_policy_rollouts"] * self._settings["max_epoch_length"]
                data_ratio = min_samples / self.getExperience().samples()
                batch_ratio = batch_ratio * data_ratio
                
                
                additional_on_poli_trianing_updates_ = self._settings["additional_on_policy_training_updates"]
                if ( additional_on_poli_trianing_updates_ < 1 ): ## should have at least one training update
                    additional_on_poli_trianing_updates_ = 1
                if (self._settings['train_critic']):
                    states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self._expBuff.getNonMBAEBatch(min(self._expBuff.samples(), self._settings["experience_length"]))
                    vf_updates = int(additional_on_poli_trianing_updates_ * batch_ratio)
                    if ("critic_updates_per_actor_update" in self._settings 
                        and (self._settings['critic_updates_per_actor_update'] > 1)):
                        vf_updates = int(vf_updates * self._settings['critic_updates_per_actor_update'])
                    vf_updates = max(vf_updates, 1)
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print ("Performing ", vf_updates, " critic epochs")
                    loss = self.getPolicy().trainCritic(states=states__, actions=actions__, 
                                                 rewards=rewards__, result_states=result_states__, 
                                                 falls=falls__, G_t=G_ts__, p=p, updates=vf_updates, 
                                                 batch_size=value_function_batch_size)
                if (self._settings['train_forward_dynamics']):
                    if ( 'keep_seperate_fd_exp_buffer' in self._settings and (self._settings['keep_seperate_fd_exp_buffer'])):
                        # print ("Using seperate (off-policy) exp mem for FD model")
                        # states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self.getFDExperience().get_batch(min(self.getFDExperience().samples(), self._settings["experience_length"]))
                        ### The FD model data grows rather large...
                        states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self.getFDExperience().get_batch(min(self._expBuff.samples(), self._settings["experience_length"]))
                    else:
                        states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self.getExperience().get_batch(min(self._expBuff.samples(), self._settings["experience_length"]))
                    fd_updates = int(additional_on_poli_trianing_updates_ * batch_ratio)
                    if ("fd_updates_per_actor_update" in self._settings 
                        and (self._settings['fd_updates_per_actor_update'] > 1)):
                        fd_updates = int(max(fd_updates * self._settings['fd_updates_per_actor_update'], 1))
                    fd_updates = max(fd_updates, 1)
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print ("Performing ", fd_updates, " fd epochs")
                        
                    dynamicsLoss = self._fd.train(states=states__, actions=actions__, 
                                                  result_states=result_states__, rewards=rewards__, 
                                                  updates=fd_updates, batch_size=value_function_batch_size,
                                                  datas=datas_, trainInfo=trainInfo)
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print ("Forward Dynamics Loss: ", dynamicsLoss)
                
                if (self._settings['train_actor']):
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print ("Performing ", int(additional_on_poli_trianing_updates_), " policy epoch(s)")
                        
                    states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self._expBuff.get_batch(min(self._expBuff.samples(), self._settings["experience_length"]))
                    loss_ = self.getPolicy().trainActor(states=states__, actions=actions__, rewards=rewards__, result_states=result_states__, falls=falls__, 
                                                     advantage=advantage__, exp_actions=exp_actions__, G_t=G_ts__, forwardDynamicsModel=self._fd,
                                                     p=p, updates=int(additional_on_poli_trianing_updates_), batch_size=batch_size_)
                dynamicsLoss = 0
                
                if ('state_normalization' in self._settings and 
                    (self._settings["state_normalization"] == "adaptive")):
                    self.getExperience()._updateScaling()
                    self.setStateBounds(self.getExperience().getStateBounds())
                    self.setActionBounds(self.getExperience().getActionBounds())
                    self.setRewardBounds(self.getExperience().getRewardBounds())
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        print("Learner, Scaling State params: ", self.getStateBounds())
                        print("Learner, Scaling Action params: ", self.getActionBounds())
                        print("Learner, Scaling Reward params: ", self.getRewardBounds())
                    
                return (loss, dynamicsLoss)
                        
                                
            for ii__ in range(additional_on_poli_trianing_updates):
                trainInfo["iteration"] = ii__
                if (self._settings['train_forward_dynamics']
                    and not ((("train_LSTM_FD" in self._settings)
                    and (self._settings["train_LSTM_FD"] == True))
                             or 
                             (("train_LSTM_Reward" in self._settings)
                    and (self._settings["train_LSTM_Reward"] == True))
                             )):
                    t0 = time.time()
                    if ("fd_updates_per_actor_update" in self._settings 
                        and (self._settings['fd_updates_per_actor_update'] >= 1)):
                        for i in range(self._settings['fd_updates_per_actor_update']):
                            
                            if ("fd_algorithm" in self._settings
                                and (self._settings["fd_algorithm"] == "algorithm.DiscriminatorKeras.DiscriminatorKeras")): 
                                ### hack to train a batch from the policy state distribution
                                states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self.getExperience().get_batch(value_function_batch_size)
                                dynamicsLoss = self._fd.train(states=states__, actions=actions__, result_states=result_states__, rewards=rewards__*0, datas=datas__, trainInfo=trainInfo)
                                
                            if ( 'keep_seperate_fd_exp_buffer' in self._settings and (self._settings['keep_seperate_fd_exp_buffer'])):
                                # print ("Using seperate (off-policy) exp mem for FD model")
                                states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self.getFDExperience().get_batch(value_function_batch_size)
                                # print ("fd exp buff state bounds: ", self.getFDExperience()._state_bounds)
                                # print ("fd state bounds: ", self._fd._state_bounds)
                            else:
                                states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self.getExperience().get_batch(value_function_batch_size)
                            if ("fd_algorithm" in self._settings
                                and (self._settings["fd_algorithm"] == "algorithm.DiscriminatorKeras.DiscriminatorKeras")):
                                rewards__ = (rewards__ * 0) + 1
                            dynamicsLoss = self._fd.train(states=states__, actions=actions__, result_states=result_states__, rewards=rewards__, lstm=False, datas=datas__, trainInfo=trainInfo)
                            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                print ("Forward Dynamics Loss: ", dynamicsLoss)
                            
                                # loss = self.getPolicy().trainDyna(predicted_states=predicted_result_states__, actions=actions__, rewards=rewards__, result_states=result_states__, falls=falls__)
                            if (self._settings['train_critic_on_fd_output'] and 
                                (( self.getPolicy().numUpdates() % self._settings['dyna_update_lag_steps']) == 0) and 
                                ( ( self.getPolicy().numUpdates() %  self._settings['steps_until_target_network_update']) >= (self._settings['steps_until_target_network_update']/10)) and
                                ( ( self.getPolicy().numUpdates() %  self._settings['steps_until_target_network_update']) <= (self._settings['steps_until_target_network_update'] - (self._settings['steps_until_target_network_update']/10)))
                                ):
                                predicted_result_states__ = self._fd.predict_batch(states=states__, actions=actions__)
                                loss = self.getPolicy().trainDyna(predicted_states=predicted_result_states__, actions=actions__, rewards=rewards__, result_states=result_states__, falls=falls__)
                                
                                if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                    print("Performing Dyna Update, loss: ", loss)
                                # print("Updated params: ", self.getPolicy().getNetworkParameters()[0][0][0])
                    else:
                        if ( 'keep_seperate_fd_exp_buffer' in self._settings and (self._settings['keep_seperate_fd_exp_buffer'])):
                            # print ("Using seperate (off-policy) exp mem for FD model")
                            states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self.getFDExperience().get_batch(value_function_batch_size)
                            dynamicsLoss = self._fd.train(states=states__, actions=actions__, rewards=rewards__, result_states=result_states__, lstm=False, datas=datas__, trainInfo=trainInfo)
                        else:
                            print ("tmp_states:", np.array(tmp_states).shape)
                            states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self.getExperience().get_batch(value_function_batch_size)
                            dynamicsLoss = self._fd.train(states=states__, actions=actions__, rewards=rewards__, result_states=result_states__, lstm=False, datas=datas__, trainInfo=trainInfo)
                        if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                print ("Forward Dynamics Loss: ", dynamicsLoss)
                                
                        if (self._settings['train_critic_on_fd_output'] and 
                            (( self.getPolicy().numUpdates() % self._settings['dyna_update_lag_steps']) == 0) and 
                            ( ( self.getPolicy().numUpdates() %  self._settings['steps_until_target_network_update']) >= (self._settings['steps_until_target_network_update']/10)) and
                            ( ( self.getPolicy().numUpdates() %  self._settings['steps_until_target_network_update']) <= (self._settings['steps_until_target_network_update'] - (self._settings['steps_until_target_network_update']/10)))
                            ):
                            predicted_result_states__ = self._fd.predict_batch(states=_states, actions=_actions)
                            loss = self.getPolicy().trainDyna(predicted_states=predicted_result_states__, actions=_actions, rewards=_rewards, result_states=_result_states, falls=_falls)
                            
                            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                print("Performing Dyna Update, loss: ", loss)
                            # print("Updated params: ", self.getPolicy().getNetworkParameters()[0][0][0])
                    
                    t1 = time.time()
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        sim_time_ = datetime.timedelta(seconds=(t1-t0))
                        print ("FD training complete in " + str(sim_time_) + " seconds")
                        
                    logExperimentData({}, "fd_net_loss", dynamicsLoss, self._settings)
                        
                if (("train_reward_distance_metric" in self._settings
                    and (self._settings['train_reward_distance_metric']))
                    and not ((("train_LSTM_FD" in self._settings)
                    and (self._settings["train_LSTM_FD"] == True))
                             or 
                             (("train_LSTM_Reward" in self._settings)
                    and (self._settings["train_LSTM_Reward"] == True))
                             )):
                    t0 = time.time()
                    if ("fd_updates_per_actor_update" in self._settings 
                        and (self._settings['fd_updates_per_actor_update'] >= 1)):
                        for i in range(self._settings['fd_updates_per_actor_update']):
                            
                            if ("fd_algorithm" in self._settings
                                and (self._settings["fd_algorithm"] == "algorithm.DiscriminatorKeras.DiscriminatorKeras")): 
                                ### hack to train a batch from the policy state distribution
                                states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self.getExperience().get_batch(value_function_batch_size)
                                dynamicsLoss = self.getRewardModel().train(states=states__, actions=actions__, result_states=result_states__, rewards=rewards__*0, datas=datas__, trainInfo=trainInfo)
                                
                            states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self.getExperience().get_batch(value_function_batch_size)
                            if ("fd_algorithm" in self._settings
                                and (self._settings["fd_algorithm"] == "algorithm.DiscriminatorKeras.DiscriminatorKeras")):
                                rewards__ = (rewards__ * 0) + 1
                            dynamicsLoss = self.getRewardModel().train(states=states__, actions=actions__, result_states=result_states__, rewards=rewards__, lstm=False, datas=datas__, trainInfo=trainInfo)
                            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                print ("Reward Distance Model Loss: ", dynamicsLoss)
                            
                    else:
                        if ( 'keep_seperate_fd_exp_buffer' in self._settings and (self._settings['keep_seperate_fd_exp_buffer'])):
                            # print ("Using seperate (off-policy) exp mem for FD model")
                            states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self.getFDExperience().get_batch(value_function_batch_size)
                            dynamicsLoss = self.getRewardModel().train(states=states__, actions=actions__, rewards=rewards__, result_states=result_states__, lstm=False, datas=datas__, trainInfo=trainInfo)
                        else:
                            dynamicsLoss = self.getRewardModel().train(states=_states, actions=_actions, rewards=_rewards, result_states=_result_states, lstm=False, datas=datas__, trainInfo=trainInfo)
                        if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                print ("Reward Distance Model Loss: ", dynamicsLoss)
                                
                    
                    t1 = time.time()
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        sim_time_ = datetime.timedelta(seconds=(t1-t0))
                        print ("Reward Distance Model training complete in " + str(sim_time_) + " seconds")
                    
                    logExperimentData({}, "reward_net_loss", dynamicsLoss, self._settings)
                    
                if ( "refresh_rewards" in self._settings
                     and (self._settings["refresh_rewards"] == True)):
                    rlPrint(self._settings, "train", "Refreshing rewards.")
                    self.recomputeRewards(__states, __actions, __rewards, __result_states, __falls, __advantage, 
                                          __exp_actions, __G_t, __datas)
                
                if (self._settings['train_critic']
                    and (not (("train_LSTM_Critic" in self._settings)
                    and (self._settings["train_LSTM_Critic"] == True)))):
                    t0 = time.time()
                    if (self._settings['critic_updates_per_actor_update'] > 1):
                        
                        for i in range(self._settings['critic_updates_per_actor_update']):
                            if ( self._settings['agent_name'] == "algorithm.QProp.QProp"
                              or (self._settings['agent_name'] == 'algorithm.QPropKeras.QPropKeras')
                              ):
                                states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self._expBuff.getNonMBAEBatch(min(value_function_batch_size, self._expBuff.samples()))
                                critic_loss = self.getPolicy().trainOnPolicyCritic(states=states__, actions=actions__, rewards=rewards__, result_states=result_states__, falls=falls__)
                            # loss = self.getPolicy().trainOnPolicyCritic(states=tmp_states, actions=tmp_actions, rewards=tmp_rewards, result_states=tmp_result_states, falls=tmp_falls)
                            # print ("Number of samples:", self._expBuff.samples())
                            if ( 'give_mbae_actions_to_critic' in self._settings and 
                                 (self._settings['give_mbae_actions_to_critic'] == False)):
                                # if ( np.random.random() >= self._settings['model_based_action_omega']):
                                if ( np.random.random() >= -1.0):
                                    states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self._expBuff.getNonMBAEBatch(min(value_function_batch_size, self._expBuff.samples()))
                                    critic_loss = self.getPolicy().trainCritic(states=states__, actions=actions__, rewards=rewards__, 
                                                                 result_states=result_states__, falls=falls__, G_t=G_ts__,
                                                                 p=p)
                                else:
                                    # print('off-policy action update')
                                    states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self._expBuff.get_batch(min(value_function_batch_size, self._expBuff.samples()))
                                    actions____ = self.getPolicy().predict_batch(states=result_states__) 
                                    predicted_result_states__ = self._fd.predict_batch(states=result_states__, actions=actions____)
                                    rewards____ = self._fd.predict_reward_batch(states=result_states__, actions=actions____)
                                    critic_loss = self.getPolicy().trainCritic(states=result_states__, actions=actions____, rewards=rewards____, 
                                                                 result_states=predicted_result_states__, falls=falls__, G_t=G_ts__,
                                                                 p=p)
                            else:
                                if ( 'keep_seperate_fd_exp_buffer' in self._settings 
                                     and (self._settings['keep_seperate_fd_exp_buffer'] == True)
                                    and ('train_critic_with_fd_data' in self._settings) 
                                     and (self._settings['train_critic_with_fd_data'] == True)
                                     ):
                                    # print ("Using seperate (off-policy) exp mem for Q model")
                                    states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self.getFDExperience().get_batch(value_function_batch_size)
                                else:
                                    states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self._expBuff.get_batch(min(value_function_batch_size, self._expBuff.samples()))
                                critic_loss = self.getPolicy().trainCritic(states=states__, actions=actions__, rewards=rewards__, 
                                                             result_states=result_states__, falls=falls__, G_t=G_ts__,
                                                             p=p)
                            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                print("Critic loss: ", critic_loss)
                            if not np.isfinite(loss) or (critic_loss > 500) :
                                # np.set_printoptions(threshold=np.nan)
                                print ("Critic training loss is Odd: ", critic_loss)
                                print ("States: " + str(np.mean(states__)) + " ResultsStates: " + str(np.mean(result_states__)) + " Rewards: " + str(np.mean(rewards__)) + " Actions: " + str(np.mean(actions__)))
                            
                    else:
                        _states, _actions, _result_states, _rewards, _falls, G_ts__, exp_actions__, _advantage, _datas = self._expBuff.get_batch(value_function_batch_size)
                        critic_loss = self.getPolicy().trainCritic(states=_states, actions=_actions, rewards=_rewards, 
                                                     result_states=_result_states, falls=_falls, G_t=G_ts__,
                                                     p=p)
                        if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                print("Critic loss: ", loss)
                        if not np.isfinite(loss) or (loss > 500) :
                            np.set_printoptions(threshold=np.nan)
                            print ("Critic training loss is Odd: ", loss)
                            print ("States: " + str(np.mean(_states)) + " ResultsStates: " + str(np.mean(_result_states)) + " Rewards: " + str(np.mean(_rewards)) + " Actions: " + str(np.mean(_actions)))
                    
                    t1 = time.time()
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        sim_time_ = datetime.timedelta(seconds=(t1-t0))
                        print ("Critic training complete in " + str(sim_time_) + " seconds")

                if (self._settings['train_actor']
                    and (not (("train_LSTM" in self._settings)
                    and (self._settings["train_LSTM"] == True)))):
                    t1 = time.time()
                    if ( 'use_multiple_policy_updates' in self._settings and 
                         ( self._settings['use_multiple_policy_updates'] == True) ):
                        for i in range(self._settings['critic_updates_per_actor_update']):
                        
                            _states, _actions, _result_states, _rewards, _falls, G_ts__, exp_actions__, _advantage, _datas = self._expBuff.get_exporation_action_batch(batch_size_)
                            
                            loss_actor = self.getPolicy().trainActor(states=_states, actions=_actions, rewards=_rewards, result_states=_result_states, 
                                                         falls=_falls, advantage=_advantage, exp_actions=exp_actions__, G_t=G_ts__, 
                                                         forwardDynamicsModel=self._fd, p=p)
                    else:
                        _states, _actions, _result_states, _rewards, _falls, G_ts__, exp_actions__, _advantage, _datas = self._expBuff.get_exporation_action_batch(batch_size_)
                        loss_actor = self.getPolicy().trainActor(states=_states, actions=_actions, rewards=_rewards, result_states=_result_states, falls=_falls, 
                                                     advantage=_advantage, exp_actions=exp_actions__, G_t=G_ts__, forwardDynamicsModel=self._fd,
                                                     p=p)
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print("Policy Loss: ", loss_actor)
                    t1 = time.time()
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        sim_time_ = datetime.timedelta(seconds=(t1-t0))
                        print ("Policy training complete in " + str(sim_time_) + " seconds")
                dynamicsLoss = 0 
                
                logExperimentData({}, "critic_loss", critic_loss, self._settings)
                logExperimentData({}, "loss_actor", loss_actor, self._settings)
                               
        else: ## Off-policy
            
            for update in range(self._settings['training_updates_per_sim_action']): ## Even more training options...
                t0 = time.time()
                for i in range(self._settings['critic_updates_per_actor_update']):
                    
                    if ( 'give_mbae_actions_to_critic' in self._settings and 
                         (self._settings['give_mbae_actions_to_critic'] == False)):
                        # if ( np.random.random() >= self._settings['model_based_action_omega']):
                        _states, _actions, _result_states, _rewards, _falls, _G_ts, _exp_actions, _advantage, _datas = self._expBuff.getNonMBAEBatch(value_function_batch_size)
                        loss = self.getPolicy().trainCritic(states=_states, actions=_actions, rewards=_rewards, 
                                                     result_states=_result_states, falls=_falls, G_t=G_ts__,
                                                     p=p)
                        _states, _actions, _result_states, _rewards, _falls, _G_ts, _exp_actions, _advantage, _datas = self._expBuff.get_batch(value_function_batch_size)
                    else:
                        _states, _actions, _result_states, _rewards, _falls, _G_ts, _exp_actions, _advantage, _datas = self._expBuff.get_batch(value_function_batch_size)
                        loss = self.getPolicy().trainCritic(states=_states, actions=_actions, rewards=_rewards, 
                                                     result_states=_result_states, falls=_falls, G_t=G_ts__,
                                                     p=p)
                        if ('rebatch_data' in self._settings 
                            and (self._settings['rebatch_data'] == True)
                            ):
                            _states, _actions, _result_states, _rewards, _falls, _G_ts, _exp_actions, _advantage, _datas = self._expBuff.get_batch(value_function_batch_size)

                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print("Critic loss: ", loss)
                    if not np.isfinite(loss) or (loss > 500) :
                        np.set_printoptions(threshold=np.nan)
                        print ("Critic training loss is Odd: ", loss)
                        print ("States: " + str(np.mean(_states)) + " ResultsStates: " + str(np.mean(_result_states)) + " Rewards: " + str(np.mean(_rewards)) + " Actions: " + str(np.mean(_actions)))
                        
                t1 = time.time()
                if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                    sim_time_ = datetime.timedelta(seconds=(t1-t0))
                    print ("Crtic training complete in " + str(sim_time_) + " seconds")
                    
                if (self._settings['train_actor']):
                    t1 = time.time()
                    loss_ = self.getPolicy().trainActor(states=_states, actions=_actions, rewards=_rewards, 
                                                 result_states=_result_states, falls=_falls, advantage=_advantage, 
                                                 exp_actions=_exp_actions, G_t=G_ts__, forwardDynamicsModel=self._fd, p=p)
                    t1 = time.time()
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        sim_time_ = datetime.timedelta(seconds=(t1-t0))
                        print ("Policy training complete in " + str(sim_time_) + " seconds")
                dynamicsLoss = 0 
                if (self._settings['train_forward_dynamics']):
                    t1 = time.time()
                    if ( 'keep_seperate_fd_exp_buffer' in self._settings and (self._settings['keep_seperate_fd_exp_buffer'])):
                        # print ("Using seperate (off-policy) exp mem for FD model")
                        _states, _actions, _result_states, _rewards, _falls, _G_ts, _exp_actions, _advantage, _datas = self.getFDExperience().get_batch(value_function_batch_size)
                        
                    dynamicsLoss = self._fd.train(states=_states, actions=_actions, result_states=_result_states, rewards=_rewards, datas=_datas, trainInfo=trainInfo)
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print ("Forward Dynamics Loss: ", dynamicsLoss)
                    
                    if ( 'give_mbae_actions_to_critic' in self._settings and 
                         (self._settings['give_mbae_actions_to_critic'] == False)):
                        ### perform a Q like update
                        actions____ = self.getPolicy().predict_batch(states=_result_states) ### I think these could have noise added to them.
                        predicted_result_states__ = self._fd.predict_batch(states=_result_states, actions=actions____)
                        rewards____ = self._fd.predict_reward_batch(states=_result_states, actions=actions____)
                        loss = self.getPolicy().trainCritic(states=_result_states, actions=actions____, rewards=rewards____, 
                                                     result_states=predicted_result_states__, falls=_falls,
                                                     p=p)
                        
                    if (self._settings['train_critic_on_fd_output'] and 
                        (( self.getPolicy().numUpdates() % self._settings['dyna_update_lag_steps']) == 0) and 
                        ( ( self.getPolicy().numUpdates() %  self._settings['steps_until_target_network_update']) >= (self._settings['steps_until_target_network_update']/10)) and
                        ( ( self.getPolicy().numUpdates() %  self._settings['steps_until_target_network_update']) <= (self._settings['steps_until_target_network_update'] - (self._settings['steps_until_target_network_update']/10)))
                        ):
                        
                        predicted_result_states__ = self._fd.predict_batch(states=_states, actions=_actions)
                        loss = self.getPolicy().trainDyna(predicted_states=predicted_result_states__, actions=_actions, rewards=_rewards, result_states=_result_states, falls=_falls)
                        if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                            print ( "Dyna training loss: ", loss)
                        if not np.isfinite(loss) or (loss > 500) :
                            np.set_printoptions(threshold=np.nan)
                            print ("Critic training loss is Odd: ", loss)
                            print ("States: " + str(np.mean(_states)) + " ResultsStates: " + str(np.mean(_result_states)) + " Rewards: " + str(np.mean(_rewards)) + " Actions: " + str(np.mean(_actions)))
                    
                    t1 = time.time()
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        sim_time_ = datetime.timedelta(seconds=(t1-t0))
                        print ("FD training complete in " + str(sim_time_) + " seconds")
                            
            # import lasagne
            # val_params = lasagne.layers.helper.get_all_param_values(self.getPolicy().getModel().getCriticNetwork())
            # pol_params = lasagne.layers.helper.get_all_param_values(self.getPolicy().getModel().getActorNetwork())
            # fd_params = lasagne.layers.helper.get_all_param_values(self._fd.getModel().getForwardDynamicsNetwork())
            # print ("Learning Agent: Model pointers: val, ", self.getPolicy().getModel(), " poli, ", self.getPolicy().getModel(),  " fd, ", self._fd.getModel())
            # print("pol first layer params: ", pol_params[1])
            # print("val first layer params: ", val_params[1])
            # print("fd first layer params: ", fd_params[1])               
                            
        if self._useLock:
            self._accesLock.release()
        return (loss, dynamicsLoss) 
    
    def predict(self, state, evaluation_=False, p=None, sim_index=None, bootstrapping=False, use_mbrl=False):
        if self._useLock:
            self._accesLock.acquire()
        # import numpy as np
        # print ("state: ", np.array(state).shape, state)
        state = self.processState(state)
        # print ("state after: ", np.array(state).shape, state)
        if (use_mbrl):
            action = self.getSampler().predict(state, p=p, sim_index=sim_index, bootstrapping=bootstrapping)
            act = [action]
        else:
            act = self.getPolicy().predict(state, evaluation_=evaluation_, p=p, sim_index=sim_index, bootstrapping=bootstrapping)
        if self._useLock:
            self._accesLock.release()
        return act
    
    def setNoise(self, noise):
        """
            Normalized states for learning
        """
        self._noise_ = noise
        
    def getNoise(self):
        """
            Normalized states for learning
        """
        return self._noise_
    
    def sample(self, state_, deterministic_=True, evaluation_=False, p=None, sim_index=None, bootstrapping=False,
               epsilon=1.0, sampling=False, time_step=0):
        """
            The logic for sampling for different types of distributions
        """
        import numpy as np
        from model.LearningUtil import entropy
        omega = self.getSettings()["omega"]
        entropy_ = 0
        stds=[]
        """
            epsilon greedy action select
            pa1 is best action from policy
            ra1 is the noisy policy action action
            ra2 is the random action
            e is proabilty to select random action
            0 <= e < omega < 1.0
        """
        r = np.random.rand(1)[0]
        # print ("float(settings['anneal_exploration']), epsilon * p: ", float(settings['anneal_exploration']), epsilon * p)
        if ((not evaluation_) ### This logic has gotten far to complicated.... 
            and 
            (
                ( ### Explore if r < annealing value
                    (self.getSettings()['on_policy']) 
                    and 
                    (
                        ("anneal_exploration" in self.getSettings()) 
                        and (self.getSettings()['anneal_exploration'] != False)
                        and (r < (max(float(self.getSettings()['anneal_exploration']), epsilon * p))) 
                    )
                ) 
                    or ### Always explore 
                    ( 
                        (self.getSettings()['on_policy'])
                        and ("anneal_exploration" in self.getSettings()) 
                        and (self.getSettings()['anneal_exploration'] == False)
                    )
                    or ### Always explore 
                    (
                        (self.getSettings()['on_policy'])
                        and (not "anneal_exploration" in self.getSettings()) 
                    )
                    or  
                    ( ### Explore sometimes
                        (self.getSettings()['on_policy'])
                        and (r < (epsilon * p)) 
                    )
                )
            ): # explore random actions  
            exp_action = int(1)
            r2 = np.random.rand(1)[0]
            if ((r2 < (self.getSettings()["omega"] * p))) and (not sampling) :
                ### explore hand crafted actions
                # return ra2
                # randomAction = randomUniformExporation(self.getActionBounds()) # Completely random action
                # action = randomAction
                if ((self.getSettings()['exploration_method'] == 'sampling') or
                    (self.getSettings()['exploration_method'] == 'gaussian_network')): 
                    action = [randomUniformExporation(self.getActionBounds())] # Completely random action
                else:
                    action = np.random.choice(action_selection)
                    action__ = actor.getActionParams(action)
                    action = [action__]
                ### off policy
                exp_action = int(0)
                # print ("Discrete action choice: ", action, " epsilon * p: ", omega * p)
            else : 
                ### add noise to current policy
                pa_ = self.predict(state_, p=p, sim_index=sim_index, bootstrapping=bootstrapping)
                if ( ((self.getSettings()['exploration_method'] == 'OrnsteinUhlenbeck') 
                      # or (bootstrapping)
                      ) 
                     and (not sampling)):
                    if ( 'anneal_policy_std' in self.getSettings() and (self.getSettings()['anneal_policy_std'])):
                        self._noise_ = OUNoise(theta=0.15, sigma=self.getSettings()["exploration_rate"] * p, previousNoise=self._noise_)
                        action = pa_ + (self._noise_ * action_bound_std(self.getActionBounds())) 
                    else:
                        self._noise_ = OUNoise(theta=0.15, sigma=self.getSettings()["exploration_rate"], previousNoise=self._noise_)
                        action = pa_ + (self._noise_ * action_bound_std(self.getActionBounds()))
                elif ( (self.getSettings()['exploration_method'] == 'gaussian_network' or 
                      (self.getSettings()['use_stochastic_policy'] == True))
                      or (self.getSettings()['exploration_method'] == 'gaussian_random')
                       ):
    
                    # action = randomExporation(self.getSettings()["exploration_rate"], pa)
                    if ( 'anneal_policy_std' in self.getSettings() and (self.getSettings()['anneal_policy_std'])):
                        std_ = self.predict_std(state_, p=p)
                    else:
                        std_ = self.predict_std(state_, p=1.0)
                    stds.append(std_)
                    entropy_ = entropy(std_)
                    action = randomExporationSTD(pa_, std_, np.array(self.getActionBounds(), dtype=float))
                    # print("Action2: ", action)
                        
                elif ((self.getSettings()['exploration_method'] == 'thompson')):
                    # print ('Using Thompson sampling')
                    action = thompsonExploration(self, self.getSettings()["exploration_rate"], state_)
                elif ((self.getSettings()['exploration_method'] == 'deterministic')):
                    ### try random action
                    pa_ = np.random.choice(range(2), 1)
                    # print ("pa_:", pa_)
                    action = pa_
                elif ((self.getSettings()['exploration_method'] == 'sampling')):
                    ## Use a sampling method to find a good action
                    if (self.getSettings()["forward_dynamics_predictor"] == "simulator"
                        or (self.getSettings()["forward_dynamics_predictor"] == "simulator_parallel")):
                        sim_state_ = exp.getSimState()
                    else:
                        sim_state_ = state_
                    # print ("explore on state: ", sim_state_)
                    action = self.getSampler().predict(sim_state_, p=p, sim_index=sim_index, bootstrapping=bootstrapping)
                    action = [action]
                    # print("samples action: ", action)
                else:
                    print ("Exploration method unknown: " + str(self.getSettings()['exploration_method']))
                    sys.exit(1)
                # randomAction = randomUniformExporation(np.array(self.getActionBounds(), dtype=float)) # Completely random action
                # randomAction = random.choice(action_selection)
                if ("use_model_based_action_optimization" in self.getSettings() and self.getSettings()["use_model_based_action_optimization"] and self.getSettings()["train_forward_dynamics"] ):
                    """
                    if ( ('anneal_mbae' in self.getSettings()) and self.getSettings()['anneal_mbae'] ):
                        mbae_omega = p * self.getSettings()["model_based_action_omega"]
                    else:
                    """
                    mbae_omega = self.getSettings()["model_based_action_omega"]
                    # print ("model_based_action_omega", self.getSettings()["model_based_action_omega"])
                    if (np.random.rand(1)[0] < mbae_omega):
                        ## Need to be learning a forward dynamics deep network for this
                        mbae_lr = self.getSettings()["action_learning_rate"]
                        std_p = 1.0
                        use_rand_act = False
                        if ( ('use_std_avg_as_mbae_learning_rate' in self.getSettings()) 
                             and (self.getSettings()['use_std_avg_as_mbae_learning_rate'] == True )
                             ):
                            ### Need to normalize this learning space
                            avg_policy_std = np.mean(self.predict_std(state_)/action_bound_std(np.array(self.getActionBounds(), dtype=float)))
                            # print ("avg_policy_std: ", avg_policy_std)
                            mbae_lr = avg_policy_std
                        if ( ('anneal_mbae' in self.getSettings()) and self.getSettings()['anneal_mbae'] ):
                            mbae_lr = p * mbae_lr
                            # print("MBAE p: ", p)
                        if ( 'MBAE_anneal_policy_std' in self.getSettings() and (self.getSettings()['MBAE_anneal_policy_std'])):
                            std_p = p
                        if ( 'use_random_actions_for_MBAE' in self.getSettings()):
                            use_rand_act = self.getSettings()['use_random_actions_for_MBAE']
                            
                        # print ("old action:", action)
                        (action, value_diff) = getOptimalAction(self.getForwardDynamics(), self.getPolicy(), state_, action_lr=mbae_lr, use_random_action=use_rand_act, p=std_p)
                        # print ("new action:", action)
                        # if ( 'give_mbae_actions_to_critic' in self.getSettings() and 
                        #      (self.getSettings()['give_mbae_actions_to_critic'] == False)):
                        exp_action = int(2)
                        # print ("Using MBAE: ", state_)
                        # if ( ('print_level' in self.getSettings()) and (self.getSettings()["print_level"]== 'debug') ):
                            # print("MBAE action:")
                # print ("Exploration: Before action: ", pa, " after action: ", action, " epsilon: ", epsilon * p )
            exp_action = [[exp_action]] * len(state_)
        else: 
            ### exploit policy
            exp_action = [[0]] *  len(state_)
            ## For sampling method to skip sampling during evaluation.
            use_MBRL = False
            if ("evalaute_with_MBRL" in self.getSettings() and
                (self.getSettings()["evalaute_with_MBRL"] == True) ):
                use_MBRL = True

            pa = self.predict(state_, evaluation_=evaluation_, p=p, sim_index=sim_index, 
                               bootstrapping=bootstrapping, use_mbrl=use_MBRL)
            
            action = pa
            # print ("Exploitation: ", action , " epsilon: ", epsilon * p)
        
        """
        elif not action_space_continuous:
            pa = model.predict(state_)
            action = random.choice(action_selection)
            action = eGreedy(pa, action, epsilon * p)
            # print("Action selection:", action_selection, " action: ", action)
            action__ = actor.getActionParams(action)
            action = [action]
            reward_ = actor.actContinuous(exp, action__, bootstrapping=True)
            agent_not_fell = actor.hasNotFallen(exp)
        """
        return (action, exp_action, entropy_, state_)
    
    def predict_std(self, state, evaluation_=False, p=1.0):
        if self._useLock:
            self._accesLock.acquire()
        state = self.processState(state)
        std = self.getPolicy().predict_std(state, p=p)
        if self._useLock:
            self._accesLock.release()
        return std
    
    def predict_target(self, state, evaluation_=False, p=None, sim_index=None, bootstrapping=False, use_mbrl=False):
        if self._useLock:
            self._accesLock.acquire()
        # import numpy as np
        # print ("state: ", np.array(state).shape, state)
        state = self.processState(state)
        # print ("state after: ", np.array(state).shape, state)
        if (use_mbrl):
            action = self.getSampler().predict_target(state, p=p, sim_index=sim_index, bootstrapping=bootstrapping)
            act = [action]
        else:
            act = self.getPolicy().predict_target(state, evaluation_=evaluation_, p=p, sim_index=sim_index, bootstrapping=bootstrapping)
        if self._useLock:
            self._accesLock.release()
        return act
    
    def predictWithDropout(self, state):
        if self._useLock:
            self._accesLock.acquire()
        state = self.processState(state)
        act = self.getPolicy().predictWithDropout(state)
        if self._useLock:
            self._accesLock.release()
        return act
    
    def predictNextState(self, state, action):
        return self._fd.predict(state, action)
    
    def q_value(self, state):
        if self._useLock:
            self._accesLock.acquire()
        state = self.processState(state)
        q = self.getPolicy().q_value(state)
        if self._useLock:
            self._accesLock.release()
        return q
    
    def q_values(self, state):
        if self._useLock:
            self._accesLock.acquire()
        state = self.processState(state)
        q = self.getPolicy().q_values(state)
        if self._useLock:
            self._accesLock.release()
        return q
    
    def q_values2(self, state, falls):
        state = self.processState(state)
        q = self.getPolicy().q_values2(state)
        if self._useLock:
            self._accesLock.release()
        return q
    
    def updateFrontPolicy(self, lowerPolicy):
        self.getPolicy().updateFrontPolicy(lowerPolicy.getPolicy())
    
    def setFrontPolicy(self, lowerPolicy):
        self.getPolicy().setFrontPolicy(lowerPolicy.getPolicy())
    
    def processState(self, state):
        """
            Input: state, non-normalized states from environment
        """
        s_length = len(state)
        if self._useLock:
            self._accesLock.acquire()
        if ("use_dual_state_representations" in self.getSettings()
            and (self.getSettings()["use_dual_state_representations"] == True)):
            if ("use_viz_for_policy" in self.getSettings() 
                and self.getSettings()["use_viz_for_policy"] == True):
            # print ("State: ", state)
                state = [x[1] for x in state]
            elif ("use_dual_viz_state_representations" in self.getSettings()
                  and (self.getSettings()["use_dual_viz_state_representations"] == True)):
                # state = state[:,0,0]
                state = [x[0] for x in state]
            else:
                state = [x[0] for x in state]
        if ("use_hack_state_trans" in self.getSettings()
            and (self.getSettings()["use_hack_state_trans"] == True)):
            import numpy as np
            state = np.array(state)
            state = state[:,:len(self.getStateBounds()[0])]
        elif ("use_hack_state_trans" in self.getSettings()
            and (("policy_connections" in self.getSettings()
            and (any([self.getSettings()["agent_id"] == m[1] for m in self.getSettings()["policy_connections"]])) ))):
            import numpy as np
            state = np.array(state)
            # print ("state: ", state)
            state = np.concatenate((state, np.zeros((state.shape[0],
                        self.getSettings()["use_hack_state_trans"]-state.shape[1]))), axis=-1)
            return state
        
            
        assert s_length == len(state), "before state length: " + str(s_length) + " == " + str(len(state))
        return state
        
    def bellman_error(self):
        if self._useLock:
            self._accesLock.acquire()
        """
        if ("use_dual_state_representations" in self.getSettings()
            and (self.getSettings()["use_dual_state_representations"] == True)):
            print ("State: ", state)
            state = state[0]
            print ("State: ", state)
        """
        if ( 'value_function_batch_size' in self.getSettings()):
            batch_size=self.getSettings()["value_function_batch_size"]
        else:
            batch_size=self.getSettings()["batch_size"]
        if (("train_LSTM_Critic" in self.getSettings())
        and (self.getSettings()["train_LSTM_Critic"] == True)):
            # _states, _actions, _result_states, _rewards, _falls, _G_ts, _exp_actions, _advantage, _datas = model.getExperience().get_batch(batch_size)
            batch_size_lstm = 4
            if ("lstm_batch_size" in self.getSettings()):
                batch_size_lstm = self.getSettings()["lstm_batch_size"][1]
            state, action, result_state, reward, fall, G_ts_, exp_actions_, advantages_, datas = self.get_multitask_trajectory_batch(batch_size=min(batch_size_lstm, self.getExperience().samplesTrajectory()))
        else:
            state, action, result_state, reward, fall, _G_ts, exp_actions, advantage, datas = self.get_batch(batch_size)
        if ("use_hack_state_trans" in self.getSettings()
            and (self.getSettings()["use_hack_state_trans"] == True)):
            import numpy as np
            state = np.array(state)
            state = state[:,:len(self.getStateBounds()[0])]
        err = self.getPolicy().bellman_error(state, action, reward, result_state, fall)
        if self._useLock:
            self._accesLock.release()
        return err
        
    def initEpoch(self, exp_):
        if (not (self._sampler == None ) ):
            self._sampler.initEpoch(exp_)
    
    def setSampler(self, sampler):
        self._sampler = sampler
    def getSampler(self):
        return self._sampler
    
    def setEnvironment(self, exp):
        if (not self._sampler == None ):
            self._sampler.setEnvironment(exp)
            
    def getPolicyParameters(self):
        return self.getPolicy().getNetworkParameters()
    def setPolicyParameters(self, params):
        self.getPolicy().setNetworkParameters(params)
        
    def setFDStateBounds(self, bounds):
        self.getForwardDynamics().setStateBounds(bounds)
    def setFDActionBounds(self, bounds):
        self.getForwardDynamics().setActionBounds(bounds)
    def setFDRewardBounds(self, bounds):
        self.getForwardDynamics().setRewardBounds(bounds)
        
    def getFDParameters(self):
        return self.getForwardDynamics().getNetworkParameters()
    def setFDParameters(self, params):
        self.getPolicy().setNetworkParameters(params)
    
            
    def getStateBounds(self):
        return self.getPolicy().getStateBounds()
    def getActionBounds(self):
        return self.getPolicy().getActionBounds()
    def getRewardBounds(self):
        return self.getPolicy().getRewardBounds()
    
    def getFDStateBounds(self):
        return self.getForwardDynamics().getStateBounds()
    def getFDActionBounds(self):
        return self.getForwardDynamics().getActionBounds()
    def getFDRewardBounds(self):
        return self.getForwardDynamics().getRewardBounds()
    
    def setStateBounds(self, bounds):
        import numpy as np
        bounds = np.array(bounds)
        self.getPolicy().setStateBounds(bounds)
        if (self.getExperience() is not None):
            self.getExperience().setStateBounds(bounds)
        if (self.getSettings()['train_forward_dynamics']):
            if ("use_dual_state_representations" in self.getSettings()
                and (self.getSettings()["use_dual_state_representations"] == True)):
                pass
            else:
                # self.getForwardDynamics().setStateBounds(bounds)
                if ( 'keep_seperate_fd_exp_buffer' in self.getSettings() 
                     and (self.getSettings()['keep_seperate_fd_exp_buffer'] == True)):
                     
                    if (self.getFDExperience() is not None):
                        self.getForwardDynamics().setStateBounds(self.getFDExperience().getStateBounds())
                    # self.getFDExperience().setStateBounds(bounds)
                else:
                    ### Using the same state bounds across models 
                    self.getForwardDynamics().setStateBounds(bounds)
                    
    def setActionBounds(self, bounds):
        import numpy as np
        if not self.getSettings()["action_space_continuous"]:
            return
        bounds = np.array(bounds[:,:len(self.getPolicy().getActionBounds()[0])])
        self.getPolicy().setActionBounds(bounds)
        if (self.getExperience() is not None):
            self.getExperience().setActionBounds(bounds)
        if (self.getSettings()['train_forward_dynamics']):
            
            if ( 'keep_seperate_fd_exp_buffer' in self.getSettings() 
                 and (self.getSettings()['keep_seperate_fd_exp_buffer'])
                 and (self.getFDExperience() is not None)):
                # self.getFDExperience().setActionBounds(bounds)
                if ( "fd_action_bounds" not in self.getSettings()):
                    self.getForwardDynamics().setActionBounds(self.getFDExperience().getActionBounds())
            else:
                if ( "fd_action_bounds" not in self.getSettings()):
                    self.getForwardDynamics().setActionBounds(bounds)   
                
    def setRewardBounds(self, bounds):
        import numpy as np
        bounds = np.array(bounds)
        self.getPolicy().setRewardBounds(bounds)
        if (self.getExperience() is not None):
            self.getExperience().setRewardBounds(bounds)
        if (self.getSettings()['train_forward_dynamics']):
            self.getForwardDynamics().setRewardBounds(bounds)
            if ( 'keep_seperate_fd_exp_buffer' in self.getSettings() 
                 and (self.getSettings()['keep_seperate_fd_exp_buffer'])
                 and (self.getFDExperience() is not None)):
                # self.getFDExperience().setRewardBounds(bounds)
                self.getForwardDynamics().setRewardBounds(self.getFDExperience().getRewardBounds())
            else:
                self.getForwardDynamics().setRewardBounds(bounds)

    def updateTargetModel(self):
        self.getPolicy().updateTargetModel()
        if (self.getSettings()['train_forward_dynamics']):
            self.getForwardDynamics().updateTargetModel()
            
#     def setSettings(self, settings):
# #         super().setSettings(self, settings)
#         self._settings = settings
#         self.getPolicy().setSettings(settings)
#         if (self.getSettings()['train_forward_dynamics']):
#             self.getForwardDynamics().setSettings(settings)
        
    
    def insertTuple(self, tuple):
        self.getExperience().insertTuple(tuple)
        
    def insertTrajectory(self, states, actions, result_states, rewards, falls, G_ts, advantage, exp_actions, data):
        self.getExperience().insertTrajectory(states, actions, result_states, rewards, falls, G_ts, advantage, exp_actions, data)
        
    def insertFDTuple(self, tuple):
        self.getFDExperience().insertTuple(tuple)
        
    def insertFDTrajectory(self, states, actions, result_states, rewards, falls, G_ts, advantage, exp_actions, data):
        self.getFDExperience().insertTrajectory(states, actions, result_states, rewards, falls, G_ts, advantage, exp_actions, data)
        
    def samples(self):
        if self.getExperience() is None:
            return 0
        else:
            return self.getExperience().samples()
    
    def get_batch(self, batch_size):
        return self.getExperience().get_batch(batch_size)
    
    def get_multitask_trajectory_batch(self, batch_size):
        return self.getExperience().get_multitask_trajectory_batch(batch_size)
    
    def getFDBatch(self, batch_size):
        return self.getFDExperience().get_batch(batch_size)
    def getFDmultitask_trajectory_batch(self, batch_size):
        return self.getFDExperience().get_multitask_trajectory_batch(batch_size)
    
    def loadExperience(self, filename):
        self.getExperience().loadFromFile(filename+"_expBufferInit.hdf5")
        
    def loadFDExperience(self, filename):
        self.getFDExperience().loadFromFile(filename+"_FD_expBufferInit.hdf5")
    
    def _updateScaling(self):
        self.getExperience()._updateScaling()
        if ( "keep_seperate_fd_exp_buffer" in self.getSettings() 
                     and ( self.getSettings()["keep_seperate_fd_exp_buffer"] == True )):
            self.getFDExperience()._updateScaling()
                    
    def saveTo(self, directory, bestPolicy=False, bestFD=False, suffix=""):
        from util.SimulationUtil import getAgentName
        suffix_ = suffix
        if ( bestPolicy == True):
            suffix_ = suffix_ + "_Best"
        self.getPolicy().saveTo(directory+getAgentName()+suffix_ )
        if (self.getSettings()["save_experience_memory"] == "continual"
            or(self.getSettings()["save_experience_memory"] == "all")):
            self.getExperience().saveToFile(directory+getAgentName()+suffix_+"_expBufferInit.hdf5")
            if (self.getSettings()['train_forward_dynamics']):
                if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                    print ("Saving Experience FD memory")
                if ("keep_seperate_fd_exp_buffer" in self.getSettings()
                    and (self.getSettings()["keep_seperate_fd_exp_buffer"] == True)):
                    self.getFDExperience().saveToFile(directory+getAgentName()+suffix_+"_FD_expBufferInit.hdf5")
        
        suffix_ = suffix
        if ( bestFD == True):
            suffix_ = "_Best"
        if (self.getSettings()['train_forward_dynamics']):
            ### Grab a batch of data and save video poc for now
            if (("train_LSTM_Reward" in self.getSettings())
                and (self.getSettings()["train_LSTM_Reward"] == True)
                and (self.getSettings()["fd_algorithm"] == "algorithm.SLACModel.SLACModel") ):
                batch_size_lstm_fd = 4
                if ("lstm_batch_size" in self.getSettings()):
                    batch_size_lstm_fd = self.getSettings()["lstm_batch_size"][0]
                ### This can consume a lot of memory if trajectories are long...
                state_, action_, resultState_, reward_, fall_, G_ts_, exp_actions, advantage_, datas = self.getFDmultitask_trajectory_batch(batch_size=batch_size_lstm_fd)
                self.getForwardDynamics().saveTo(directory+"forward_dynamics"+suffix_, states=state_, actions=action_)
            else:
                self.getForwardDynamics().saveTo(directory+"forward_dynamics"+suffix_)
        
    def loadFrom(self, directory, best=False, suffix_=".plk"):
        import dill
        from util.SimulationUtil import getAgentName
        if ( bestPolicy == True):
            suffix = "_Best.pkl"
        file_name=directory+getAgentName()+"_Best.pkl"
        f = open(file_name, 'rb')
        self.setPolicy(dill.load(f))
        f.close()
        
        if (self.getSettings()['train_forward_dynamics']):
            file_name_dynamics=directory+"forward_dynamics_"+suffix
            f = open(file_name_dynamics, 'rb')
            self.setForwardDynamics(dill.load(f))
            f.close()
            
            
    def finish(self):
        if (self.getSampler() is not None):
            self.getSampler().finish()
        

import copy
# class LearningWorker(threading.Thread):
class LearningWorker(Process):
    def __init__(self, input_exp_queue, agent, random_seed_=23):
        super(LearningWorker, self).__init__()
        self._input_queue= input_exp_queue
        # self._namespace = namespace
        self._agent = agent
        self._process_random_seed = random_seed_
        
    def setLearningNamespace(self, learningNamespace):    
        self._learningNamespace = learningNamespace
    
    # @profile(precision=5)    
    def run(self):
        _profile=False
        print ('Worker started')
        if ( _profile ):
            import cProfile, pstats, io
            pr = cProfile.Profile()
            pr.enable()
        np.random.seed(self._process_random_seed)
        if (self._agent._settings['on_policy']):
            self._agent._expBuff.clear()
        # do some initialization here
        step_ = 0
        iterations_=0
        # self._agent._expBuff = self.__learningNamespace.experience
        while True:
            tmp = self._input_queue.get()
            if tmp == None:
                break
            #if len(tmp) == 6:
                # self._input_queue.put(tmp)
            if tmp == "clear":
                if (self._agent._settings["print_levels"][self._agent._settings["print_level"]] >= self._agent._settings["print_levels"]['train']):
                    print ("Clearing exp memory")
                self._agent._expBuff.clear()
                continue
            #    continue # don't learn from eval tuples
            # (state_, action, reward, resultState, fall, G_t) = tmp
            # print (tmp)
            # (state__, act__, res__, rew__,  fall__, G_t, exp__) = tmp
            # print ("learning state: ", state__)
            # print ("Learner Size of state input Queue: " + str(self._input_queue.qsize()))
            # self._agent._expBuff = self.__learningNamespace.experience
            if self._agent._settings['action_space_continuous']:
                self._agent._expBuff.insertTuple(tmp)
                if ( 'keep_seperate_fd_exp_buffer' in self._agent._settings and (self._agent._settings['keep_seperate_fd_exp_buffer'])):
                    self._agent._expBuff_FD.insertTuple(tmp)
                # print ("Experience buffer size: " + str(self.__learningNamespace.experience.samples()))
                # print ("Reward Scale: ", self._agent._reward_bounds)
                # print ("Reward Scale Model: ", self._agent._pol.getRewardBounds())
            else:
                self._agent._expBuff.insertTuple(tmp)
                if ( 'keep_seperate_fd_exp_buffer' in self._agent._settings and (self._agent._settings['keep_seperate_fd_exp_buffer'])):
                    self._agent._expBuff_FD.insertTuple(tmp)
            # print ("Learning agent experience size: " + str(self._agent._expBuff.samples()))
            step_ += 1
            if self._agent._expBuff.samples() > self._agent._settings["batch_size"] and ((step_ >= self._agent._settings['sim_action_per_training_update']) ):
                __states, __actions, __result_states, __rewards, __falls, __G_ts, __exp_actions, __advantage = self._agent._expBuff.get_batch(self._agent._settings["batch_size"])
                # print ("States: " + str(__states) + " ResultsStates: " + str(__result_states) + " Rewards: " + str(__rewards) + " Actions: " + str(__actions))
                (loss, dynamicsLoss) = self._agent.train(_states=__states, _actions=__actions, _rewards=__rewards, _result_states=__result_states, _falls=__falls, trainInfo=trainInfo)
                # print ("Master Agent Running training step, loss: " + str(loss) + " PID " + str(os.getpid()))
                # print ("Updated parameters: " + str(self._agent._pol.getNetworkParameters()[3]))
                if not np.isfinite(loss):
                    print ("States: " + str(__states) + " ResultsStates: " + str(__result_states) + " Rewards: " + str(__rewards) + " Actions: " + str(__actions))
                    print ("Training loss is Nan: ", loss)
                    sys.exit()
                # if (step_ % 10) == 0: # to help speed things up
                # self._learningNamespace.agentPoly = self._agent.getPolicy().getNetworkParameters()
                data = (self._agent._expBuff, self._agent.getPolicy().getNetworkParameters())
                if (self._agent._settings['train_forward_dynamics']):
                    # self._learningNamespace.forwardNN = self._agent.getForwardDynamics().getNetworkParameters()
                    data = (self._agent._expBuff, self._agent.getPolicy().getNetworkParameters(), self._agent.getForwardDynamics().getNetworkParameters())
                    if ( 'keep_seperate_fd_exp_buffer' in self._agent._settings and (self._agent._settings['keep_seperate_fd_exp_buffer'])):
                        data = (self._agent._expBuff, self._agent.getPolicy().getNetworkParameters(), self._agent.getForwardDynamics().getNetworkParameters(), self._agent._expBuff_FD)
                # self._learningNamespace.experience = self._agent._expBuff
                self._agent.setStateBounds(self._agent.getExperience().getStateBounds())
                self._agent.setActionBounds(self._agent.getExperience().getActionBounds())
                self._agent.setRewardBounds(self._agent.getExperience().getRewardBounds())
                # if (self._agent._settings["print_levels"][self._agent._settings["print_level"]] >= self._agent._settings["print_levels"]['train']):
                    # print("Learner, Scaling State params: ", self._agent.getStateBounds())
                    # print("Learner, Scaling Action params: ", self._agent.getActionBounds())
                    # print("Learner, Scaling Reward params: ", self._agent.getRewardBounds())
                ## put and do not block
                try:
                    # print ("Sending network params:")
                    if (not (self._output_message_queue.full())):
                        self._output_message_queue.put(data, False)
                    else:
                        ## Pull out (discard) an old one
                        self._output_message_queue.get(False)
                        self._output_message_queue.put(data, False)
                except Exception as inst:
                    if (self._agent._settings["print_levels"][self._agent._settings["print_level"]] >= self._agent._settings["print_levels"]['train']):
                        print ("LearningAgent: output model parameter message queue full: ", self._output_message_queue.qsize())
                step_=0
            iterations_+=1
            # print ("Done one update:")
        self._output_message_queue.close()
        self._output_message_queue.cancel_join_thread()
        print ("Learning Worker Complete:")
        
        if ( _profile ):
            pr.disable()
            f = open('x.prof', 'a')
            pstats.Stats(pr).sort_stats('time').print_stats()
            pstats.Stats(pr, stream=f).sort_stats('time').print_stats()
            f.close()
            
        return
        
    def updateExperience(self, experience):
        self._agent._expBuff = experience
        
    def setMasterAgentMessageQueue(self, queue):
        self._output_message_queue = queue
        
    # @profile(precision=5)  
    def updateModel(self):
        print ("Updating model to: ", self._learningNamespace.model)
        old_poli = self._agent.getPolicy()
        self._agent.setPolicy(self._learningNamespace.model)
        del old_poli
