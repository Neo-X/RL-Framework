"""
    An interface class for Agents to be used in the system.

"""
import datetime
from multiprocessing import Process
# from pathos.multiprocessing import Pool
import logging
from model.LearningAgent import LearningAgent
from model.ModelUtil import *
from util.SimulationUtil import logExperimentData
from util.utils import rlPrint
import os
import copy
import threading
import time

log = logging.getLogger(os.path.basename(__file__))
# np.set_printoptions(threshold=np.nan)

class LearningAgentLight(LearningAgent):
    
    def __init__(self, n_in=None, n_out=None, state_bounds=None, 
                       action_bounds=None, reward_bound=None, settings_=None):
        super(LearningAgent,self).__init__(n_in=n_in, n_out=n_out, state_bounds=state_bounds, 
                                           action_bounds=action_bounds, reward_bound=reward_bound, settings_=settings_)
        self._useLock = False
        if self._useLock:
            self._accesLock = threading.Lock()
        self._pol = None
        self._fd = None
        self._sampler = None
        self._expBuff = None
        self._expBuff_FD = None
        if ("use_learned_reward_function" in self._settings
            and (self._settings["use_learned_reward_function"] == "ic2_marginal")):
            from model.buffers import GaussianCircularBuffer
            self._marginal = GaussianCircularBuffer(8, self._settings["ic2_marginal_window_size"])
        
    
    def putDataInExpMem(self, _states, _actions, _rewards, _result_states, _falls, _advantage=None, 
              _exp_actions=None, _G_t=None, datas=None, recomputeRewards=False, p=1.0, mode="all"):
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
        # Causes the new scaling values to be computed but not applied. They are applied later after the updates
        self.getExperience()._settings["state_normalization"] = "variance"
#         print ("mode: ", mode)
        for (state__, action__, next_state__, reward__, fall__, G_t__, exp_action__, 
              datas__) in zip(_states, _actions, _result_states, _rewards, _falls, _G_t,
                               _exp_actions, datas):

            # Because the valid state checks only like numpy arrays, not lists
            path = {}
            path["states"] = np.array(state__)
            path['reward'] = np.array(reward__)
            path['agent_id'] = np.array(reward__) * 0
            path["terminated"] = False
            state___ = state__
            next_state___ = next_state__
            
            if ("use_dual_state_representations" in self._settings
                        and (self._settings["use_dual_state_representations"] == True)):
                state___ = [s[1] for s in state__]
                next_state___ = [ns[1] for ns in next_state__]
            # Validate data
            if (checkValidData(state___, action__, next_state___, reward__, verbose=True) and 
                checkDataIsValid(G_t__, verbose=True)):
                
                if (recomputeRewards 
#                     or
#                     ("force_use_mod_state_for_critic" in self._settings
#                     and (self._settings["force_use_mod_state_for_critic"] == True))
                    ):
                    # timestep, agent, state
                    path["terminated"] = False
#                     print("totally recomputing rewards")

                    # This is where we can use the forward dynamics to "relable" rewards, i.e. set the reward function. E.g. for SLAC
                    if self._settings.get("use_learned_reward_function", None) == "ic2":
                        reward__ = self.getForwardDynamics().predict_reward(state___, action__)
                    elif ("use_learned_reward_function" in self._settings
                        and (self._settings["use_learned_reward_function"] == "ic2_marginal")):
                        latent_sample = self.getForwardDynamics().predict_encoding(state___, action__)
                        reward__ = np.zeros((latent_sample.shape[0], 1))
                        for k in range(latent_sample.shape[0]):
                            reward__[k] = self._marginal.logprob(latent_sample[k])
                            self._marginal.add(latent_sample[k])
                    else:
                        agent_traj = np.array(
                            [np.array([np.array(np.array(tmp_states__), 
                                                dtype=self._settings['float_type']) for tmp_states__ in datas__["agent_obs"]])])
                        
                        imitation_traj = np.array([np.array([np.array(np.array(tmp_states__), 
                                                                      dtype=self._settings['float_type']) for tmp_states__ in datas__["expert_obs"]])])
#                         reward__0 = exp.computeImitationReward(rewmodel.predict)
                        ##Don't need - for BCE reward
#                         print ("imitation_traj shape: ", imitation_traj.shape)
#                         print ("agent_traj shape: ", agent_traj.shape)
                        
                        if ("refresh_rewards_rl_method" in self._settings
                            and (self._settings["refresh_rewards_rl_method"] == "fd")):
                            reward__fd = self.getForwardDynamics().predict_reward_fd(agent_traj, imitation_traj)
                            reward___ = reward__fd
                        else:
                            reward__r = self.getForwardDynamics().predict_reward_(agent_traj, imitation_traj)
                            reward__fd = self.getForwardDynamics().predict_reward_fd(agent_traj, imitation_traj)
                            reward___ = ((reward__r * p ) + (reward__fd * (1 + (1-p)))) / 2.0
#                         print ("Refreshed reward origin, reward refresh, r, fd, diff: ", np.concatenate((reward__, reward___, reward__r, reward__fd, reward__ - reward___), axis=1))
                        reward__ = reward___
#                         reward__1 = exp.computeImitationReward(rewmodel.predict_reward)
                    w_d = -2.0
                    if ("learned_reward_function_norm_weight" in self._settings):
                        w_d = self._settings["learned_reward_function_norm_weight"]
                    # reward__ = np.exp((reward__*reward__)*w_d)
                    # print ("reward__", reward__)
                    path['states'] = state__ # np.array([np.array(np.array(tmp_states__[0]), dtype=self._settings['float_type']) for tmp_states__ in state__])
                    path['reward'] = reward__
                    path['falls'] = fall__
                    
                    path['agent_id'] = datas__['agent_id']
#                     print ("state__ shape: ", np.array(path['states']).shape)
                    if ("force_use_mod_state_for_critic" in self._settings
                        and (self._settings["force_use_mod_state_for_critic"] == True)):
                        ### append recurrent state to state
                        agent_traj_before = np.array(
                            [np.array([np.array(np.array(tmp_states__), dtype=self._settings['float_type']) for tmp_states__ in datas__["agent_obs_before"]])])
                        imitation_traj_before = np.array([np.array([np.array(np.array(tmp_states__), dtype=self._settings['float_type']) for tmp_states__ in datas__["expert_obs_before"]])])
                        
                        agent_encode, imitation_encode = self.getForwardDynamics().predict_encodings(agent_traj_before, imitation_traj_before)
                       
                        agent_encode_next, imitation_encode_next = self.getForwardDynamics().predict_encodings(agent_traj, imitation_traj)
                        ## append to state
                        agent_states = np.array(
                            [np.array([np.array(np.array(tmp_states__[0]), dtype=self._settings['float_type']) for tmp_states__ in state__])])
#                         print ("path[\"states\"] shape2222: ", np.array(agent_states).shape, agent_encode.shape, imitation_encode.shape)
                        agent_states = np.concatenate((agent_states, agent_encode, imitation_encode), axis=-1)[0]
                        agent_states_next = np.array(
                            [np.array([np.array(np.array(tmp_states__[0]), dtype=self._settings['float_type']) for tmp_states__ in next_state__])])
                        agent_states_next = np.concatenate((agent_states_next, agent_encode_next, imitation_encode), axis=-1)[0]
#                         print ("path[\"states\"] shape2222: ", np.array(agent_states).shape)
                        
                        states___ = [[mod_state, state___[1]] for tmp_states__, mod_state in zip(state__, agent_states)]
                        path['states'] = states___
                        ### (state, encode_state_agent, encode_state_expert)
                        datas__["mod_state"] = agent_states
                        ### (state, encode_state_agent_next, encode_state_expert_next)
                        datas__["mod_state_next"] = agent_states_next
    #                         paths = compute_advantage_(self, [path], discount_factor, self._settings['GAE_lambda'])
    #                         adv__ = paths["advantage"]
    #                         baselines_.append(np.array(paths["baseline"]))
    #                         advantage.append(np.array(adv__))
                if (mode == "fd_only"):
                    advantage__ = path["reward"]
                else:
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
                
                # Data is a trajectory
                for j in range(len(state__)):
                    data___ = {}
                    for key in datas__:
                        # print ("key: ", key, " datas__", datas__[key])
                        try: ## sometimes the info dictionary has different data/keys...
                            data___[key] = datas__[key][j]
                        except:
                            print("Error putting data into dictionary for tupple")
                    
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
                    if (mode == "all" or (mode == "policy")):
                        self.getExperience().insertTuple(tup)
                    if ( 'keep_seperate_fd_exp_buffer' in self._settings and (self._settings['keep_seperate_fd_exp_buffer'])):
                        if ("use_dual_state_representations" in self._settings
                        and (self._settings["use_dual_state_representations"] == True)):
                            tup = ([data___["agent_obs_before"]], [action__[j]], [data___["expert_obs_before"]], [reward__[j]], [fall__[j]], [G_t__[j]], [exp_action__[j]], [advantage__[j]], data___)
                        # This is always done and works well for computing the adaptive state bounds.
                        if (mode == "all" or (mode == "fd_only")):
                            self.getFDExperience().insertTuple(tup)
                    num_samples_ = num_samples_ + 1
                    
        return ( num_samples_, (tmp_states, tmp_actions, tmp_result_states, tmp_rewards, tmp_falls, tmp_G_t, tmp_advantage, tmp_exp_action, tmp_datas))
        
    
    # @profile(precision=5)
    def train(self, _states, _actions, _rewards, _result_states, _falls, _advantage=None, 
              _exp_actions=None, _G_t=None, p=1.0, datas=None, trainInfo=None):
        import numpy as np
        if self._useLock:
            self._accesLock.acquire()
        loss = 0
        critic_loss = 0
        loss_actor = 0
        
        value_function_batch_size = self._settings.get('value_function_batch_size', self._settings["batch_size"])

        if self._settings['on_policy']:
            if self._settings.get('clear_exp_mem_on_poli', False):
                self.getExperience().clear()
                
            # Update target networks
            self.updateTargetModel()
            num_samples_= 1
            t0 = time.time()
            
            if self._settings.get("use_hindsight_relabeling", False) and "goal_slice_index" in self._settings:
                (_states,
                 _actions,
                 _rewards,
                 _result_states,
                 _falls,
                 _advantage,
                 _exp_actions,
                 _G_t) = self.applyHER(_states,
                                       _actions,
                                       _rewards,
                                       _result_states,
                                       _falls,
                                       _advantage,
                                       _exp_actions,
                                       _G_t,
                                       datas)
#             print ("_states: ", _states)
            (num_samples_,
             (tmp_states,
              tmp_actions,
              tmp_result_states,
              tmp_rewards,
              tmp_falls,
              tmp_G_t,
              tmp_advantage,
              tmp_exp_action,
              tmp_datas)) = self.putDataInExpMem(_states,
                                                 _actions,
                                                 _rewards,
                                                 _result_states,
                                                 _falls,
                                                 _advantage,
                                                 _exp_actions,
                                                 _G_t,
                                                 datas,
                                                 mode="fd_only")
            if self._settings.get("state_normalization", None) == "adaptive":
                if self._settings.get('keep_seperate_fd_exp_buffer', False):
                    self.getFDExperience()._updateScaling()
                    if self._settings.get('train_forward_dynamics', False):
                        self.getForwardDynamics().setStateBounds(self.getFDExperience().getStateBounds())
                        if (self._settings["action_space_continuous"]):
                            self.getForwardDynamics().setActionBounds(self.getFDExperience().getActionBounds())
                        self.getForwardDynamics().setRewardBounds(self.getFDExperience().getRewardBounds())
                log.debug("Learner, Scaling State params: ", self.getStateBounds())
                log.debug("Learner, Scaling Action params: ", self.getActionBounds())
                log.debug("Learner, Scaling Reward params: ", self.getRewardBounds())
            
            logExperimentData({}, "experience_mem_samples", self._expBuff.samples(), self._settings)
            
            batch_size_ = self._settings["batch_size"]        
                # print ("num_samples_: ", num_samples_)
            # If for some reason the data was all garbage, skip this training update.
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
            if (self._settings.get("train_LSTM", False) or
                self._settings.get("train_LSTM_Critic", False) or
                self._settings.get("train_LSTM_FD", False) or
                self._settings.get("train_LSTM_Reward", False)):
                
                # Need to normalize data
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
                
                                    
                # pass
                # print("Not Falls: ", _falls)
                # print("Rewards: ", _rewards)
                # print("Actions after: ", _actions)
#             if self._settings.get("refresh_rewards", None) == "lstm_fd":
#                 rlPrint(self._settings, "train", "Refreshing rewards.")
#                 self.recomputeRewards(__states, __actions, __rewards, __result_states, __falls, __advantage, 
#                                       __exp_actions, __G_t, __datas)

            if ( "refresh_rewards" in self._settings
                 and (self._settings["refresh_rewards"] == True)):
                rlPrint(self._settings, "train", "Refreshing rewards.")
                self.recomputeRewards(__states, __actions, __rewards, __result_states, __falls, __advantage, 
                                      __exp_actions, __G_t, __datas, p=p)
            elif (self._settings["train_actor"] == False and 
                  self._settings["train_critic"] == False):
                print("Not training actor or critic")
                return 0
            else:
                (num_samples_,
                 (tmp_states,
                  tmp_actions,
                  tmp_result_states,
                  tmp_rewards,
                  tmp_falls,
                  tmp_G_t,
                  tmp_advantage,
                  tmp_exp_action,
                  tmp_datas)) = self.putDataInExpMem(_states,
                                                     _actions,
                                                     _rewards,
                                                     _result_states,
                                                     _falls,
                                                     _advantage,
                                                     _exp_actions,
                                                     _G_t,
                                                     datas)
                
            if (self._settings["batch_size"] == "all"):
                batch_size_ = max(self._expBuff.samples(), 1)
            print ("batch_size_: ", batch_size_)
            if ((self._expBuff.samples() < value_function_batch_size 
                or (self._expBuff.samples() < batch_size_))
                and
                (not ("skip_rollouts" in self._settings and 
                        (self._settings["skip_rollouts"] == True)))):
                print("Data was mostly/all garbage or your batch size is larger than the data collected.", self._expBuff.samples(),
                      " batch size ", batch_size_, 
                      " value func batch size ", value_function_batch_size)
                return 0
                
            loss = 0
            additional_on_poli_training_updates = 1
            if self._settings.get("additional_on_policy_training_updates", False) != False:
                additional_on_poli_training_updates = self._settings["additional_on_policy_training_updates"]
            # if ("perform_multiagent_training" in self._settings): # Reduce number of updates by agent count
            #     additional_on_poli_training_updates = additional_on_poli_training_updates / self._settings["perform_multiagent_training"]
            # The data should be seen ~ 4 times
            additional_on_poli_training_updates = int(np.ceil(((self._settings["num_on_policy_rollouts"] * self._settings["max_epoch_length"] * 1) / batch_size_) * additional_on_poli_training_updates))
            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                print("additional_on_poli_training_updates: ", additional_on_poli_training_updates)
            
            if ( additional_on_poli_training_updates < 1 ): ## should have at least one training update
                additional_on_poli_training_updates = 1  
                    
            ## This lets the model do most of the training and batching. more efficient
            if ("model_perform_batch_training" in self._settings 
                and (self._settings["model_perform_batch_training"] == True )):
                # How many more times should the value function see the data
                batch_ratio = value_function_batch_size / batch_size_
                # Compensate for the ratio collected each run and the size of the replay buffer
                min_samples = self._settings["num_on_policy_rollouts"] * self._settings["max_epoch_length"]
                data_ratio = min_samples / self.getExperience().samples()
                batch_ratio = batch_ratio * data_ratio
                
                
                additional_on_poli_training_updates_ = self._settings["additional_on_policy_training_updates"]
                if ( additional_on_poli_training_updates_ < 1 ): ## should have at least one training update
                    additional_on_poli_training_updates_ = 1
                if (self._settings['train_critic']):
                    states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self._expBuff.getNonMBAEBatch(min(self._expBuff.samples(), self._settings["experience_length"]))
                    vf_updates = int(additional_on_poli_training_updates_ * batch_ratio)
                    if ("critic_updates_per_actor_update" in self._settings 
                        and (self._settings['critic_updates_per_actor_update'] > 1)):
                        vf_updates = int(vf_updates * self._settings['critic_updates_per_actor_update'])
                    vf_updates = max(vf_updates, 1)
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print("Performing ", vf_updates, " critic epochs")
                    loss = self.getPolicy().trainCritic(states=states__, actions=actions__, 
                                                 rewards=rewards__, result_states=result_states__, 
                                                 falls=falls__, G_t=G_ts__, p=p, updates=vf_updates, 
                                                 batch_size=value_function_batch_size)
                if (self._settings['train_forward_dynamics']):
                    if ( 'keep_seperate_fd_exp_buffer' in self._settings and (self._settings['keep_seperate_fd_exp_buffer'])):
                        # print("Using seperate (off-policy) exp mem for FD model")
                        # states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self.getFDExperience().get_batch(min(self.getFDExperience().samples(), self._settings["experience_length"]))
                        ### The FD model data grows rather large...
                        states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self.getFDExperience().get_batch(min(self._expBuff.samples(), self._settings["experience_length"]))
                    else:
                        states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self.getExperience().get_batch(min(self._expBuff.samples(), self._settings["experience_length"]))
                    fd_updates = int(additional_on_poli_training_updates_ * batch_ratio)
                    if ("fd_updates_per_actor_update" in self._settings 
                        and (self._settings['fd_updates_per_actor_update'] > 1)):
                        fd_updates = int(max(fd_updates * self._settings['fd_updates_per_actor_update'], 1))
                    fd_updates = max(fd_updates, 1)
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print("Performing ", fd_updates, " fd epochs")
                        
                    dynamicsLoss = self._fd.train(states=states__, actions=actions__, 
                                                  result_states=result_states__, rewards=rewards__, 
                                                  updates=fd_updates, batch_size=value_function_batch_size,
                                                  datas=datas_, trainInfo=trainInfo)
                    log.info("Forward Dynamics Loss: {}".format(dynamicsLoss))
                
                if (self._settings['train_actor']):
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print("Performing ", int(additional_on_poli_training_updates_), " policy epoch(s)")
                        
                    (states__,
                     actions__,
                     result_states__,
                     rewards__,
                     falls__,
                     G_ts__,
                     exp_actions__,
                     advantage__,
                     datas__) = self._expBuff.get_batch(min(self._expBuff.samples(), self._settings["experience_length"]))
                    
                    loss_ = self.getPolicy().trainActor(
                        states=states__,
                        actions=actions__,
                        rewards=rewards__,
                        result_states=result_states__,
                        falls=falls__,
                        advantage=advantage__,
                        exp_actions=exp_actions__,
                        G_t=G_ts__,
                        forwardDynamicsModel=self._fd,
                        p=p,
                        updates=int(additional_on_poli_training_updates_),
                        batch_size=batch_size_)
                dynamicsLoss = 0
                
                return (loss, dynamicsLoss)
                        
            for ii__ in range(additional_on_poli_training_updates):
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
                                # print("Using seperate (off-policy) exp mem for FD model")
                                states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self.getFDExperience().get_batch(value_function_batch_size)
                                # print("fd exp buff state bounds: ", self.getFDExperience()._state_bounds)
                                # print("fd state bounds: ", self._fd._state_bounds)
                            else:
                                states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self.getExperience().get_batch(value_function_batch_size)
                            if ("fd_algorithm" in self._settings
                                and (self._settings["fd_algorithm"] == "algorithm.DiscriminatorKeras.DiscriminatorKeras")):
                                rewards__ = (rewards__ * 0) + 1
#                             print ("self._fd:", self._fd)
                            dynamicsLoss = self._fd.train(states=states__, actions=actions__, result_states=result_states__, rewards=rewards__, lstm=False, datas=datas__, trainInfo=trainInfo)
                            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                log.info("Forward Dynamics Loss: {}".format(dynamicsLoss))
                            
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
                        if self._settings.get('keep_seperate_fd_exp_buffer', False):
                            # print("Using seperate (off-policy) exp mem for FD model")
                            (states__,
                             actions__,
                             result_states__,
                             rewards__,
                             falls__,
                             G_ts__,
                             exp_actions__,
                             advantage__,
                             datas__) = self.getFDExperience().get_batch(value_function_batch_size)
                            dynamicsLoss = self._fd.train(states=states__,
                                                          actions=actions__,
                                                          rewards=rewards__,
                                                          result_states=result_states__,
                                                          lstm=False,
                                                          datas=datas__,
                                                          trainInfo=trainInfo)
                        else:
                            print("tmp_states:", np.array(tmp_states).shape)
                            (states__,
                             actions__,
                             result_states__,
                             rewards__,
                             falls__,
                             G_ts__,
                             exp_actions__,
                             advantage__,
                             datas__) = self.getExperience().get_batch(value_function_batch_size)
                            dynamicsLoss = self._fd.train(states=states__,
                                                          actions=actions__,
                                                          rewards=rewards__,
                                                          result_states=result_states__,
                                                          lstm=False,
                                                          datas=datas__,
                                                          trainInfo=trainInfo)
                        if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                            log.info("Forward Dynamics Loss: {}".format(dynamicsLoss))
                                
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
                        print("FD training complete in " + str(sim_time_) + " seconds")
                        
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
                                print("Reward Distance Model Loss: ", dynamicsLoss)
                            
                    else:
                        if ( 'keep_seperate_fd_exp_buffer' in self._settings and (self._settings['keep_seperate_fd_exp_buffer'])):
                            # print("Using seperate (off-policy) exp mem for FD model")
                            states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self.getFDExperience().get_batch(value_function_batch_size)
                            dynamicsLoss = self.getRewardModel().train(states=states__, actions=actions__, rewards=rewards__, result_states=result_states__, lstm=False, datas=datas__, trainInfo=trainInfo)
                        else:
                            dynamicsLoss = self.getRewardModel().train(states=_states, actions=_actions, rewards=_rewards, result_states=_result_states, lstm=False, datas=datas__, trainInfo=trainInfo)
                        if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                print("Reward Distance Model Loss: ", dynamicsLoss)
                                
                    
                    t1 = time.time()
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        sim_time_ = datetime.timedelta(seconds=(t1-t0))
                        print("Reward Distance Model training complete in " + str(sim_time_) + " seconds")
                    
                    logExperimentData({}, "reward_net_loss", dynamicsLoss, self._settings)
                    
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
                            if ( 'keep_seperate_fd_exp_buffer' in self._settings 
                                 and (self._settings['keep_seperate_fd_exp_buffer'] == True)
                                and ('train_critic_with_fd_data' in self._settings) 
                                 and (self._settings['train_critic_with_fd_data'] == True)
                                 ):
                                # print("Using seperate (off-policy) exp mem for Q model")
                                states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self.getFDExperience().get_batch(value_function_batch_size)
                            else:
                                states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__, advantage__, datas__ = self._expBuff.get_batch(min(value_function_batch_size, self._expBuff.samples()))
                                if ("force_use_mod_state_for_critic" in self._settings
                                    and (self._settings["force_use_mod_state_for_critic"] == True)):
                                    states__ = np.array([x for x in datas__["mod_state"]])
                                    result_states__ = np.array([x for x in datas__["mod_state_next"]])
                                    ### TODO This needs to be normalized
                            critic_loss = self.getPolicy().trainCritic(states=states__, actions=actions__, rewards=rewards__, 
                                                         result_states=result_states__, falls=falls__, G_t=G_ts__,
                                                         p=p)
                            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                print("Critic loss: ", critic_loss)
                            if not np.isfinite(loss) or (critic_loss > 500) :
                                # np.set_printoptions(threshold=np.nan)
                                print("Critic training loss is Odd: ", critic_loss)
                                print("States: " + str(np.mean(states__)) + " ResultsStates: " + str(np.mean(result_states__)) + " Rewards: " + str(np.mean(rewards__)) + " Actions: " + str(np.mean(actions__)))
                            
                    else:
                        _states, _actions, _result_states, _rewards, _falls, G_ts__, exp_actions__, _advantage, _datas = self._expBuff.get_batch(value_function_batch_size)
                        critic_loss = self.getPolicy().trainCritic(states=_states, actions=_actions, rewards=_rewards, 
                                                     result_states=_result_states, falls=_falls, G_t=G_ts__,
                                                     p=p)
                        if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                print("Critic loss: ", loss)
                        if not np.isfinite(loss) or (loss > 500) :
                            np.set_printoptions(threshold=np.nan)
                            print("Critic training loss is Odd: ", loss)
                            print("States: " + str(np.mean(_states)) + " ResultsStates: " + str(np.mean(_result_states)) + " Rewards: " + str(np.mean(_rewards)) + " Actions: " + str(np.mean(_actions)))
                    
                    t1 = time.time()
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        sim_time_ = datetime.timedelta(seconds=(t1-t0))
                        print("Critic training complete in " + str(sim_time_) + " seconds")

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
                        print("Policy training complete in " + str(sim_time_) + " seconds")
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
                        print("Critic training loss is Odd: ", loss)
                        print("States: " + str(np.mean(_states)) + " ResultsStates: " + str(np.mean(_result_states)) + " Rewards: " + str(np.mean(_rewards)) + " Actions: " + str(np.mean(_actions)))
                        
                t1 = time.time()
                if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                    sim_time_ = datetime.timedelta(seconds=(t1-t0))
                    print("Crtic training complete in " + str(sim_time_) + " seconds")
                    
                if (self._settings['train_actor']):
                    t1 = time.time()
                    loss_ = self.getPolicy().trainActor(states=_states, actions=_actions, rewards=_rewards, 
                                                 result_states=_result_states, falls=_falls, advantage=_advantage, 
                                                 exp_actions=_exp_actions, G_t=G_ts__, forwardDynamicsModel=self._fd, p=p)
                    t1 = time.time()
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        sim_time_ = datetime.timedelta(seconds=(t1-t0))
                        print("Policy training complete in " + str(sim_time_) + " seconds")
                dynamicsLoss = 0 
                if (self._settings['train_forward_dynamics']):
                    t1 = time.time()
                    if ( 'keep_seperate_fd_exp_buffer' in self._settings and (self._settings['keep_seperate_fd_exp_buffer'])):
                        # print("Using seperate (off-policy) exp mem for FD model")
                        _states, _actions, _result_states, _rewards, _falls, _G_ts, _exp_actions, _advantage, _datas = self.getFDExperience().get_batch(value_function_batch_size)
                        
                    dynamicsLoss = self._fd.train(states=_states,
                                                  actions=_actions,
                                                  result_states=_result_states,
                                                  rewards=_rewards,
                                                  datas=_datas,
                                                  trainInfo=trainInfo)
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        log.info("Forward Dynamics Loss: {}".format(dynamicsLoss))
                    
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
                            print( "Dyna training loss: ", loss)
                        if not np.isfinite(loss) or (loss > 500) :
                            np.set_printoptions(threshold=np.nan)
                            print("Critic training loss is Odd: ", loss)
                            print("States: " + str(np.mean(_states)) + " ResultsStates: " + str(np.mean(_result_states)) + " Rewards: " + str(np.mean(_rewards)) + " Actions: " + str(np.mean(_actions)))
                    
                    t1 = time.time()
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        sim_time_ = datetime.timedelta(seconds=(t1-t0))
                        print("FD training complete in " + str(sim_time_) + " seconds")
                            
            # import lasagne
            # val_params = lasagne.layers.helper.get_all_param_values(self.getPolicy().getModel().getCriticNetwork())
            # pol_params = lasagne.layers.helper.get_all_param_values(self.getPolicy().getModel().getActorNetwork())
            # fd_params = lasagne.layers.helper.get_all_param_values(self._fd.getModel().getForwardDynamicsNetwork())
            # print("Learning Agent: Model pointers: val, ", self.getPolicy().getModel(), " poli, ", self.getPolicy().getModel(),  " fd, ", self._fd.getModel())
            # print("pol first layer params: ", pol_params[1])
            # print("val first layer params: ", val_params[1])
            # print("fd first layer params: ", fd_params[1])         
        ## Update scaling values
        # Updating the scaling values after the update(s) will help make things more accurate
        if self._settings.get("state_normalization", None) == "adaptive":
            self.getExperience()._updateScaling()
            self.setStateBounds(self.getExperience().getStateBounds())
            if (self._settings["action_space_continuous"]):
                self.setActionBounds(self.getExperience().getActionBounds())
            self.setRewardBounds(self.getExperience().getRewardBounds())
            if self._settings.get('keep_seperate_fd_exp_buffer', False):
                self.getFDExperience()._updateScaling()
                if self._settings.get('train_forward_dynamics', False):
                    self.getForwardDynamics().setStateBounds(self.getFDExperience().getStateBounds())
                    if (self._settings["action_space_continuous"]):
                        self.getForwardDynamics().setActionBounds(self.getFDExperience().getActionBounds())
                    self.getForwardDynamics().setRewardBounds(self.getFDExperience().getRewardBounds())
            log.debug("Learner, Scaling State params: ", self.getStateBounds())
            log.debug("Learner, Scaling Action params: ", self.getActionBounds())
            log.debug("Learner, Scaling Reward params: ", self.getRewardBounds())      
                            
        if self._useLock:
            self._accesLock.release()
        return (loss, dynamicsLoss) 
    
    