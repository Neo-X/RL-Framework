"""
    An interface class for Agents to be used in the system.

"""
from multiprocessing import Process, Queue
# from pathos.multiprocessing import Pool
import threading
import time
from model.LearningAgent import LearningAgent
from model.ModelUtil import *
from util.utils import rlPrint
import os
import copy
import time
import datetime
import numpy as np
# np.set_printoptions(threshold=np.nan)

class LearningMultiAgent(LearningAgent):
    
    def __init__(self, settings_):
        super(LearningMultiAgent,self).__init__(settings_=settings_)
        self._useLock = False
        if self._useLock:
            self._accesLock = threading.Lock()
        
        self._agents = []
        for m in range(self.getSettings()["perform_multiagent_training"]):
            ### I think these settings will get propogated to the experience memory objects
            settings__ = copy.deepcopy(self.getSettings())
            if ("additional_on_policy_training_updates" in  self.getSettings()
                and (type(self.getSettings()["additional_on_policy_training_updates"]) is list)):
                settings__["additional_on_policy_training_updates"] = self.getSettings()["additional_on_policy_training_updates"][m]
            if (type(self.getSettings()["exploration_method"]) is list):
                settings__["exploration_method"] = self.getSettings()["exploration_method"][m]
            if (type(self.getSettings()["state_normalization"]) is list):
                settings__["state_normalization"] = self.getSettings()["state_normalization"][m]
            # LearningAgent(self.getSettings())
            settings__["use_hindsight_relabeling"] = False
            settings__["agent_id"] = m
            agent = LearningAgent(settings__)
            self._agents.append(agent)
        # self._agents = [LearningAgent(self.getSettings()) for i in range(self.getSettings()["perform_multiagent_training"])]
        # list of periods over which each agent is active
        # TODO: make this not specific to 2 agents
        if "hlc_timestep" in self.getSettings():
            self.time_skips = [self.getSettings()["hlc_timestep"], 1]
        self.latest_actions = [None] * self.getSettings()["perform_multiagent_training"]
        self.latest_exp_act = [None] * self.getSettings()["perform_multiagent_training"]
        self.latest_entropy = [None] * self.getSettings()["perform_multiagent_training"]
        
    def getAgents(self):
        return self._agents
    
    def reset(self):
        self.latest_actions = [None] * self.getSettings()["perform_multiagent_training"]
        self.latest_exp_act = [None] * self.getSettings()["perform_multiagent_training"]
        self.latest_entropy = [None] * self.getSettings()["perform_multiagent_training"]
        [p.reset() for p in self.getAgents()]
        
    def getPolicy(self):
        if self._useLock:
            self._accesLock.acquire()
        pol = self.getAgents()[0].getPolicy()
        if self._useLock:
            self._accesLock.release()
        return pol
        
    def setPolicy(self, pol):
        if self._useLock:
            self._accesLock.acquire()
        [a.setPolicy(pol_) for a, pol_ in zip(self.getAgents(), pol)]
        if (not self._sampler == None ):
            self._sampler.setPolicy(pol)
        if self._useLock:
            self._accesLock.release()
        
    def getForwardDynamics(self):
        if self._useLock:
            self._accesLock.acquire()
        fd = self.getAgents()[0].getForwardDynamics()
        if self._useLock:
            self._accesLock.release()
        return fd
                
    def setForwardDynamics(self, pol):
        if self._useLock:
            self._accesLock.acquire()
        [a.setForwardDynamics(pol_) for a, pol_ in zip(self.getAgents(), pol)]
        if (not self._sampler == None ):
            self._sampler.setPolicy(pol)
        if self._useLock:
            self._accesLock.release()
            
    def getRewardModel(self):
        if self._useLock:
            self._accesLock.acquire()
        rm = self._rm
        if self._useLock:
            self._accesLock.release()
        return rm
                
    def setRewardModel(self, rm):
        if self._useLock:
            self._accesLock.acquire()
        self._rm = rm
        if self._useLock:
            self._accesLock.release()
        
    def setSettings(self, settings):
        self._settings = settings

        ### Only propogate the logger settings because settings are unqiue for each agent.
        if hasattr(self, '_agents'):
            for a in range(len(self.getAgents())):
                set = self.getAgents()[a].getSettings()
                if "logger_instance" in self._settings:
                    set["logger_instance"] = self._settings["logger_instance"]
                if ("logger_instance" in self._settings):
                    set["round"] = self._settings["round"]
                
                self.getAgents()[a].setSettings(set)
        
    def getSettings(self):
        return self._settings
    
    def getPolicyNetworkParameters(self):
        return [a.getPolicy().getNetworkParameters() for a in self.getAgents()]
    def getFDNetworkParameters(self):
        return [p.getForwardDynamics().getNetworkParameters() for p in self.getAgents()]
    def getRewardNetworkParameters(self):
        return [p.getRewardModel().getNetworkParameters() for p in self.getAgents()]
    
    def setPolicyNetworkParameters(self, params):
        return [p.getPolicy().setNetworkParameters(param) for p, param in zip(self.getAgents(), params)]
    def setFDNetworkParameters(self, params):
        return [p.getForwardDynamics().setNetworkParameters(param) for p, param in zip(self.getAgents(), params)]
    def setRewardNetworkParameters(self, params):
        return [p.getRewardModel().setNetworkParameters(param) for p, param in zip(self.getAgents(), params)]
    
    
        
    def setExperience(self, experienceBuffer):
        [p.setExperience(exp) for p, exp in zip(self.getAgents(), experienceBuffer)]
        self._expBuff = experienceBuffer 
    def getExperience(self):
        return [p.getExperience() for p in self.getAgents()]
    
    def setFDExperience(self, experienceBuffer):
        [p.setFDExperience(exp) for p, exp in zip(self.getAgents(), experienceBuffer)]
    def getFDExperience(self):
        return [p.getFDExperience() for p in self.getAgents()]
    
    def putDataInExpMem(self, _states, _actions, _rewards, _result_states, _falls, _advantage=None, 
              _exp_actions=None, _G_t=None, recomputeRewards=False):
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
        ### Causes the new scaling values to be computed but not applied. They are applied later after the updates
        self.getExperience()._settings["state_normalization"] = "variance"
        for (state__, action__, next_state__, reward__, fall__, G_t__, exp_action__, advantage__) in zip(_states, _actions, _result_states, _rewards, _falls, _G_t, _exp_actions, _advantage):

            ### Because the valid state checks only like numpy arrays, not lists
            state___ = state__
            next_state___ = next_state__
            
            if ("use_dual_state_representations" in self._settings
                        and (self._settings["use_dual_state_representations"] == True)):
                state___ = [s[0] for s in state__]
                next_state___ = [ns[1] for ns in next_state__]
            ### Validate data
            if (checkValidData(state___, action__, next_state___, reward__, verbose=True) and 
                checkDataIsValid(advantage__, verbose=True), checkDataIsValid(G_t__, verbose=True)):
                
                if (recomputeRewards==True):
                    path = {}
                    ### timestep, agent, state
                    path["terminated"] = False
                    agent_traj = np.array([np.array([np.array(np.array(tmp_states__[1]), dtype=self._settings['float_type']) for tmp_states__ in state__])])
                    print ("agent_traj shape: ", agent_traj.shape)
                    imitation_traj = np.array([np.array([np.array(np.array(tmp_states__[1]), dtype=self._settings['float_type']) for tmp_states__ in next_state__])])
                    print ("imitation_traj shape: ", imitation_traj.shape)
                    reward__ = self.getForwardDynamics().predict_reward_(agent_traj, imitation_traj)
                    # print ("reward__", reward__)
                    path['states'] = state__ # np.array([np.array(np.array(tmp_states__[0]), dtype=self._settings['float_type']) for tmp_states__ in state__])
                    path['reward'] = reward__
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
                
                ### Data is a trajectory
                for j in range(len(state__)):
                    
                    tup = ([state__[j]], [action__[j]], [next_state__[j]], [reward__[j]], [fall__[j]], [G_t__[j]], [exp_action__[j]], [advantage__[j]])
                    if ("use_dual_state_representations" in self._settings
                        and (self._settings["use_dual_state_representations"] == True)):
                        if ("use_viz_for_policy" in self._settings 
                                and self._settings["use_viz_for_policy"] == True):
                            tup = ([state__[j][1]], [action__[j]], [next_state__[j][1]], [reward__[j]], [fall__[j]], [G_t__[j]], [exp_action__[j]], [advantage__[j]])
                        elif ("use_dual_viz_state_representations" in self._settings
                            and (self._settings["use_dual_viz_state_representations"] == True)):
                            tup = ([state__[j][0]], [action__[j]], [next_state__[j][0]], [reward__[j]], [fall__[j]], [G_t__[j]], [exp_action__[j]], [advantage__[j]])
                        else:
                            tup = ([state__[j][0]], [action__[j]], [next_state__[j][0]], [reward__[j]], [fall__[j]], [G_t__[j]], [exp_action__[j]], [advantage__[j]])
                    self.getExperience().insertTuple(tup)
                    if ( 'keep_seperate_fd_exp_buffer' in self._settings and (self._settings['keep_seperate_fd_exp_buffer'])):
                        if ("use_dual_state_representations" in self._settings
                        and (self._settings["use_dual_state_representations"] == True)):
                            if ("use_viz_for_policy" in self._settings 
                                and self._settings["use_viz_for_policy"] == True):
                                ### Want viz for input and dense for output to condition the preception part of the network
                                tup = ([state__[j][1]], [action__[j]], [next_state__[j][0]], [reward__[j]], [fall__[j]], [G_t__[j]], [exp_action__[j]], [advantage__[j]])
                            else:
                                # print ("self.getFDExperience().getStateBounds() shape : ", self.getFDExperience().getStateBounds().shape)
                                # print ("fd exp state shape: ", state__[j][1].shape)
                                tup = ([state__[j][1]], [action__[j]], [next_state__[j][1]], [reward__[j]], [fall__[j]], [G_t__[j]], [exp_action__[j]], [advantage__[j]])
                        ### This is always done and works well for computing the adaptive state bounds.
                        self.getFDExperience().insertTuple(tup)
                    num_samples_ = num_samples_ + 1
                    
        return ( num_samples_, (tmp_states, tmp_actions, tmp_result_states, tmp_rewards, tmp_falls, tmp_G_t, tmp_advantage, tmp_exp_action))
    
    
    def processRewards(self, model, states__, actions__, rewards__, result_states__, falls__, _advantage, 
                  _exp_actions, _G_t, ep_len):
        """
            For things like HRL I am using a cheap trick to throw out extra data samples
            This will most likely mess up the advantage estimation and G_t
        """
        import numpy as np
        for tar in range(len(states__)):
            tmp_len = len(states__[tar])
            
            ### split the data according to episode length
            split_indices = [i for i  in range(ep_len, len(rewards__[tar]), ep_len) ]# math.floor(a.shape[axis] / chunk_shape[axis]))]
            states__split = np.array_split(states__[tar], split_indices, axis=0)
            reward_split = np.array_split(rewards__[tar], split_indices, axis=0)
            falls__split =  np.array_split(falls__[tar], split_indices, axis=0)
            advantage__split =  np.array_split(_advantage[tar], split_indices, axis=0)
            assert len(rewards__[tar][0::ep_len]) == len(reward_split), "len(rewards__[tar][0::ep_len]) == len(first_split): " + str(len(rewards__[tar][0::ep_len])) + " == " + str(len(reward_split))
            
            _advantage_ = []
            for ep in range(len(states__split)):
                path = {"states": np.array(states__split[ep]),
                        "reward": np.array(reward_split[ep]),
                        "falls": np.array(falls__split[ep]), 
                        "terminated": False}
                # print ("path: ", path)
                paths = compute_advantage_(model, [path], model._settings["discount_factor"], model._settings['GAE_lambda'])
                adv__ = paths["advantage"]
                # print ("advantage diff: ", _advantage[tar] - adv__)
                advantage__split[ep] = adv__
                # baselines_.append(np.array(paths["baseline"]))
                _advantage_.extend(adv__)
            
            # assert np.ceil(tmp_len/ep_len) == len(states__[tar]), "np.ceil(tmp_len/skip_num) == len(states__[tar])" + str(np.ceil(tmp_len/ep_len)) + " == " + str(len(states__[tar]))
            assert np.array(_advantage[tar]). shape == np.array(_advantage_).shape
            _advantage[tar] = _advantage_
        return _advantage
    
    
    def dataSkip(self, model, states__, actions__, rewards__, result_states__, falls__, _advantage, 
                  _exp_actions, _G_t, datas, skip_num):
        """
            For things like HRL I am using a cheap trick to throw out extra data samples
            This will most likely mess up the advantage estimation and G_t
        """
        import numpy as np
        if (skip_num > 1):
            
            removes = list(range(len(states__)))
            removes.reverse()       
            for tar in removes:
                # print ("traj length: ", len(states__[tar]))
                if (len(states__[tar]) < skip_num): ### Sometimes trajectories are too short...
                    del states__[tar]
                    del actions__[tar]
                    del rewards__[tar]
                    del result_states__[tar]
                    del falls__[tar]
                    del _advantage[tar]
                    del _exp_actions[tar]
                    del _G_t[tar]
                    del datas[tar]
                
            for tar in range(len(states__)):
                trim = -1
                if ( (len(states__[tar]) % skip_num) == 0  ):
                    trim = len(states__[tar])
                tmp_len = len(states__[tar])
                split_indices = [i for i  in range(skip_num, tmp_len, skip_num) ]# math.floor(a.shape[axis] / chunk_shape[axis]))]
                states__tr = states__[tar][:tmp_len][0::skip_num][:trim]
                
                result_states__tar =  result_states__[tar][:tmp_len][0::skip_num][:trim]
                if ("policy_connections" in self.getSettings()):
                    ### The actions for this policy will have already been replaced with the connectect policy actions.
                    # and (any([model._settings["agent_id"] == m[1] for m in self.getSettings()["policy_connections"]])) ):
                    ### Stack them instead of skipping them
                    # actions___[tar] =  actions__[tar][0::skip_num]
                    # print ("actions__[tar]: ", np.shape(actions___[tar]))
                    action_split = np.array_split(actions__[tar], split_indices, axis=0)[:trim]
                    actions__[tar] =  [np.array(rs).flatten() for rs in action_split]
                    # print ("actions__[tar] ", actions__[tar])
                    # print ("actions__[tar]: ", np.shape(actions__[tar]))
                    # actions__[tar] =  actions__[tar][0::skip_num]
                    ### stack llp_states as well
                    state_split_index = self.getSettings()["state_split_index"]
                    llp_state_split = np.array_split(states__[tar], split_indices, axis=0)[:trim]
                    llp_state_long =  [np.array(rs)[:,-state_split_index:].flatten() for rs in llp_state_split]
                    states__[tar] =  [np.concatenate((s[:-state_split_index],slp), axis=-1) for s,slp in zip(states__tr, llp_state_long)]

                    ### Add to result state as well.
                    llp_state_split = np.array_split(result_states__[tar], split_indices, axis=0)[:trim]
                    llp_state_long =  [np.array(rs)[:,-state_split_index:].flatten() for rs in llp_state_split]
                    result_states__[tar] =  [np.concatenate((s[:-state_split_index],slp), axis=-1) for s,slp in zip(result_states__tar, llp_state_long)]
                else:
                    states__[tar] = states__tr
                    actions__[tar] =  actions__[tar][:tmp_len][0::skip_num]
                    result_states__[tar] =  result_states__tar
                axis = 0
                first_split = np.array_split(rewards__[tar], split_indices, axis=0)
                assert len(rewards__[tar][0::skip_num]) == len(first_split), "len(rewards__[tar][0::skip_num]) == len(first_split): " + str(len(rewards__[tar][0::skip_num])) + " == " + str(len(first_split))
                ### Average reward over LLP steps.
                rewards__[tar] =  [[np.mean(rs)] for rs in first_split][:trim]
                
                falls__[tar] =  falls__[tar][:tmp_len][0::skip_num][:trim]
                _advantage[tar] =  _advantage[tar][:tmp_len][0::skip_num][:trim]
                _exp_actions[tar] =  _exp_actions[tar][:tmp_len][0::skip_num][:trim]
                _G_t[tar] =  _G_t[tar][:tmp_len][0::skip_num][:trim]
                
                for key in datas[tar]:
                    datas[tar][key] = datas[tar][key][:tmp_len][0::skip_num][:trim]
                
                # print("np.array(states__[tar]: ", np.array(states__[tar]).shape)
                path = {"states": np.array(states__[tar]),
                        "reward": np.array(rewards__[tar]),
                        "falls": np.array(falls__[tar]), 
                        "agent_id": np.array(datas[tar]["agent_id"]),
                        "terminated": False}
                ### Recompute advantage now that some states may be skipped
                paths = compute_advantage_(model, [path], model._settings["discount_factor"], model._settings['GAE_lambda'])
                adv__ = paths["advantage"]
                _advantage[tar] = adv__
                
                assert (np.array(states__[tar]).shape == np.array(result_states__[tar]).shape), "np.array(states__[tar]).shape == np.array(result_states__[tar]).shape: " + str( np.array(states__[tar]).shape) + " == " + str(np.array(result_states__[tar]).shape) 
                # assert np.ceil(tmp_len/skip_num) == len(states__[tar]), "np.ceil(tmp_len/skip_num) == len(states__[tar])" + str(np.ceil(tmp_len/skip_num)) + " == " + str(len(states__[tar]))            
        
        return  (states__, actions__, rewards__, result_states__, falls__, _advantage, 
                  _exp_actions, _G_t, datas)
        
    def sampleGoals(self, states, actions):
        ### get goal
        states = np.array(copy.deepcopy(states))
        actions = np.array(actions)
        probs = []
        goal = states[:,-self._settings["goal_slice_index"]:]
        # print ("old goals: ", goal)
        goals = []
        
        actions2 = self.getAgents()[1].getPolicy().predict(states)
        prob_ = -0.5 * np.sum(np.square(actions2 - actions))
        probs.append(prob_)
        goals.append(goal)

        num_goals_to_resample = (16 if not "num_goals_to_resample" in self.getSettings() else
                                 self.getSettings()["num_goals_to_resample"])
        for i in range(num_goals_to_resample):
            noise = [np.random.normal(0, 0.05, size=self._settings["goal_slice_index"])]
            noise = np.repeat(noise, len(states), axis=0)
            new_goals = goal + noise
            
            ### Copy in the new goals
            new_goals = np.array(new_goals)
            states[:,-self._settings["goal_slice_index"]:] = new_goals
            
            actions2 = self.getAgents()[1].getPolicy().predict(states)
            ### Distance between action sequence
            prob_ = -0.5 * np.sum(np.square(actions2 - actions))
            probs.append(prob_)
            goals.append(new_goals)
            
        
        # print ("probs: ", probs)
        indx = np.argmax(probs)
        return goals[indx]
        
    def applyHIRO(self, _states, _actions, _rewards, _result_states, _falls, _advantage, 
              _exp_actions, _G_t, datas):
        import numpy as np
        if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):  
            print ("Applying HIRO")
        ### Relable trajectory goal to new goals that have higher prob after LLP update
        ### Get the data for both policies
        (states__h, actions__h, rewards__h, result_states__h,
                falls__h, advantage__h, exp_actions__h, G_t__h, datas__h) = self.getSingleAgentData(_states, 
                    _actions, _rewards, _result_states, _falls, _advantage, _exp_actions, _G_t, datas,
                    agent_num=0)
        old_shape = np.array(actions__h).shape
        (states__l, actions__l, rewards__l, result_states__l,
                falls__l, advantage__l, exp_actions__l, G_t__l, datas__l) = self.getSingleAgentData(_states, 
                    _actions, _rewards, _result_states, _falls, _advantage, _exp_actions, _G_t, datas,
                    agent_num=1)
        ### For each trajectory calculate new goal
        trajectories = len(states__l)
        for traj in range(trajectories):
            ### Get the new goal(s) for the trajectory
            new_goals = []
            step=0
            steps_ = int(len(states__l[traj])/self._settings["hlc_timestep"])
            for g in range(steps_):
                ### get chunk of data
                statesl = states__l[traj][step:step+self._settings["hlc_timestep"]]
                actionsl = actions__l[traj][step:step+self._settings["hlc_timestep"]]
                new_goals = self.sampleGoals(statesl, actionsl)
                
                ### Copy in the new goals
                new_goals = np.array(new_goals)
                old_acts = np.array(actions__h[traj][step:step+self._settings["hlc_timestep"]])
                # states = np.array(copy.deepcopy(_states[traj]))
                # result_states = np.array(copy.deepcopy(_result_states[traj]))
                # states[:,-self._settings["goal_slice_index"]:] = new_goals
                # result_states[:,-self._settings["goal_slice_index"]:] = new_goals
                actions__h[traj][step:step+self._settings["hlc_timestep"]] = new_goals
                # rewards = copy.deepcopy(_rewards[traj])
                # diff = -np.fabs(result_states[:,:self._settings["goal_slice_index"]] - new_goals)
                # rewards = np.sum(diff, axis=-1, keepdims=True)
                step = step + self._settings["hlc_timestep"]
            
            
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
        assert old_shape == np.array(actions__h).shape
        return (states__h, actions__h, rewards__h, result_states__h,
                falls__h, advantage__h, exp_actions__h, G_t__h, datas__h)

    def applyHAC(self, _states, _actions, _rewards, _result_states, _falls, _advantage,
                 _exp_actions, _G_t, datas):
        import numpy as np
        # Relabel trajectory goal to new goals that are actually achieved,
        # and add penalty to rewards if goals are not achieved
        # Get the data for both policies
        (states__h, actions__h, rewards__h, result_states__h,
         falls__h, advantage__h, exp_actions__h, G_t__h, datas__h) = self.getSingleAgentData(_states,
                                                                                   _actions, _rewards, _result_states,
                                                                                   _falls, _advantage, _exp_actions,
                                                                                   _G_t, datas=datas,
                                                                                   agent_num=0)
        old_shape = np.array(actions__h).shape
        (states__l, actions__l, rewards__l, result_states__l,
         falls__l, advantage__l, exp_actions__l, G_t__l, datas__l) = self.getSingleAgentData(_states,
                                                                                   _actions, _rewards, _result_states,
                                                                                   _falls, _advantage, _exp_actions,
                                                                                   _G_t, datas=datas,
                                                                                   agent_num=1)
        hindsight_relabel_probability = 1.0 if not "hindsight_relabel_probability" in self.getSettings() else self.getSettings()["hindsight_relabel_probability"]
        subgoal_testing_probability = 1.0 if not "subgoal_testing_probability" in self.getSettings() else self.getSettings()["subgoal_testing_probability"]

        # For each trajectory calculate new goal
        trajectories = len(states__l)
        for traj in range(trajectories):
            # Get the new goal(s) for the trajectory
            step = 0
            steps_ = int(len(states__l[traj]) / self._settings["hlc_timestep"])
            for g in range(steps_):
                # get the goal that was achieved by the policy
                statesl = states__l[traj][step:step + self._settings["hlc_timestep"]]
                achieved_goals = statesl[-1][:self.getSettings()["goal_slice_index"]]
                new_goals = np.array(achieved_goals)

                # compute a penalty for not achieving goals proposed by the upper level with some probability
                if "hac_goal_threshold" in self.getSettings() and np.random.uniform() < hindsight_relabel_probability:
                    distance = np.linalg.norm(
                        new_goals - np.array(actions__h[traj][step]))
                    penalties = 0.0 if distance < self.getSettings()["hac_goal_threshold"] else self._settings["hlc_timestep"]
                    for t in range(self._settings["hlc_timestep"]):
                        rewards__h[traj][step + t] = np.array(rewards__h[traj][step + t]) - penalties

                # Copy in the new goals with some probability
                elif np.random.uniform() < subgoal_testing_probability:
                    for t in range(self._settings["hlc_timestep"]):
                        actions__h[traj][step + t] = new_goals

                step = step + self._settings["hlc_timestep"]

        assert old_shape == np.array(actions__h).shape
        return (states__h, actions__h, rewards__h, result_states__h,
                falls__h, advantage__h, exp_actions__h, G_t__h, datas__h)
        
    def getSingleAgentData(self, _states, _actions, _rewards, _result_states, _falls, _advantage, 
              _exp_actions, _G_t, datas, agent_num):
        states__ = [state_[agent_num::len(self.getAgents())] for state_ in _states]
        result_states__ = [state_[agent_num::len(self.getAgents())] for state_ in _result_states]
        if ("policy_connections" in self.getSettings()
            and (any([agent_num == m[1] for m in self.getSettings()["policy_connections"]])) ):
            ### Replace the state and action data with the other agent data.
            other_agent_id = 1
            actions__ = [state_[other_agent_id::len(self.getAgents())] for state_ in _actions]
            states__llp = [state_[other_agent_id::len(self.getAgents())] for state_ in _states]
            res_states__llp = [state_[other_agent_id::len(self.getAgents())] for state_ in _result_states]
            states___ = []
            res_states___ = []
            ### append llp states to 
            for state_, state_llp, res_state_, res_state_llp in zip(states__, states__llp, result_states__, res_states__llp):
                states____  = [np.concatenate((s_, slp), axis=-1) for s_,slp in zip(state_, state_llp)]
                res_states____  = [np.concatenate((s_, slp), axis=-1) for s_,slp in zip(res_state_, res_state_llp)]
                states___.append(states____)
                res_states___.append(res_states____)
            states__ = states___
            result_states__ = res_states___
        else:
            actions__ = [state_[agent_num::len(self.getAgents())] for state_ in _actions]
        rewards__ = [state_[agent_num::len(self.getAgents())] for state_ in _rewards]
        # result_states_tmp = [state_[agent_num::len(self.getAgents())] for state_ in _result_states]
        falls__ = [state_[agent_num::len(self.getAgents())] for state_ in _falls]
        advantage__ = [state_[agent_num::len(self.getAgents())] for state_ in _advantage]
        exp_actions__ = [state_[agent_num::len(self.getAgents())] for state_ in _exp_actions]
        G_t__ = [state_[agent_num::len(self.getAgents())] for state_ in _G_t]
        
        datas__ = []
        for tra in datas:
            datas___ = {}
            for key in tra:
                datas___[key] = tra[key][agent_num::len(self.getAgents())]
            datas__.append(datas___)
        
        assert (len(states__) == len(actions__))
        assert (np.array(states__).shape == np.array(result_states__).shape), "np.array(states__).shape == np.array(result_states__).shape: " + str(np.array(states__).shape) + " == " + str(np.array(result_states__).shape)
        return (states__, actions__, rewards__, result_states__,
                falls__, advantage__, exp_actions__, G_t__, datas__)
    # @profile(precision=5)
    def train(self, _states, _actions, _rewards, _result_states, _falls, _advantage=None, 
              _exp_actions=None, _G_t=None, p=1.0, datas=None):
        import numpy as np 
        
        # for agent_ in range(len(self.getAgents())):
        ### I want to start with the LLC (count down)
        for agent_ in range(len(self.getAgents())-1, 0 - 1, -1):
            ### Pull out the state for each agent, start at agent index and skip every number of agents 
            (states__, actions__, rewards__, result_states__,
                falls__, advantage__, exp_actions__, G_t__, datas__) = self.getSingleAgentData(_states, 
                    _actions, _rewards, _result_states, _falls, _advantage, _exp_actions, _G_t, datas=datas,
                    agent_num=agent_)
            result_states_tmp = copy.deepcopy(result_states__)

            if ( "use_centralized_critic" in self.getSettings()
                 and ((self.getSettings()["use_centralized_critic"] == True)
                      # or (agent_ in self.getSettings()["use_centralized_critic"])
                      )
                 ):
                ### Add other agent data 
                for agent__ in [i for i,x in enumerate(self.getAgents()) if i!=agent_]: ### Add the states for other agents
                    states___ = [state_[agent__::len(self.getAgents())] for state_ in _states]
                    result_states___ = [state_[agent__::len(self.getAgents())] for state_ in _states]
                    ### For each trajectory concatenate the states of the other agents onto this agents state
                    for tar in range(len(states__)):
                        for s in range(len(states__[tar])):
                            states__[tar][s] = np.concatenate((states__[tar][s],states___[tar][s]), axis=0)
                            ### Also collect the nextstate state for the other agents
                            result_states__[tar][s] = np.concatenate((result_states__[tar][s],result_states___[tar][s]), axis=0)
                            ### Create a tmp next state data structure because we need to ask for the other agent target action later
                            result_states_tmp[tar][s] = np.concatenate((result_states_tmp[tar][s],result_states___[tar][s]), axis=0) 
                        
                ### Collect the actions of the other agents as additional state info.
                for agent__ in [i for i,x in enumerate(self.getAgents()) if i!=agent_]:
                    actions___ = [state_[agent__::len(self.getAgents())] for state_ in _actions]
                    for tar in range(len(states__)):
                        for s in range(len(states__[tar])):
                            state___ = np.concatenate((states__[tar][s],actions___[tar][s]), axis=0)
                            states__[tar][s] = state___
                            ### Add garbage action to this tmp next state to create corect size state for other agent target action request
                            result_states_tmp[tar][s] = np.concatenate((result_states_tmp[tar][s],actions___[tar][s]), axis=0)
                
                ### Now that we have data of the correct size to ask for target actions of other agents, get those target actions.
                for agent__ in [i for i,x in enumerate(self.getAgents()) if i!=agent_]:
                    ### Get result state for this other agent
                    result_states_tmp_agent = np.array([state_[agent__::len(self.getAgents())] for state_ in _result_states])
                    result_states_tmp = np.array(result_states_tmp)
                    for tar in range(len(result_states_tmp_agent)):
                        concat_index = np.array(result_states_tmp_agent[tar][:]).shape[-1]
                        replace_data = np.array(result_states_tmp[tar])
                        replace_data[:,:concat_index] = np.array(result_states_tmp_agent[tar][:])
                        result_states_tmp[tar] = replace_data
                        target_actions = self.getAgents()[agent__].predict_target(result_states_tmp[tar])
                        for s in range(len(result_states___[tar])):
                            target_res_state = np.concatenate((result_states__[tar][s],target_actions[s]), axis=0)
                            result_states__[tar][s] = np.array(target_res_state)
            
            if (    "use_hindsight_relabeling" in self._settings
                    and ("HIRO" in self._settings["use_hindsight_relabeling"])
                    and (self.getSettings()["hlc_index"] == agent_)):
                (states__, actions__, rewards__, result_states__, falls__, advantage__,
                 exp_actions__, G_t__, datas__) = self.applyHIRO(_states, _actions, _rewards, _result_states, _falls, _advantage,
                                                        _exp_actions, _G_t, datas)

            if ("use_hindsight_relabeling" in self._settings
                    and ("HAC" in self._settings["use_hindsight_relabeling"])
                    and (self.getSettings()["hlc_index"] == agent_)):
                (states__, actions__, rewards__, result_states__, falls__, advantage__,
                 exp_actions__, G_t__, datas__) = self.applyHAC(_states, _actions, _rewards, _result_states, _falls, _advantage,
                                                       _exp_actions, _G_t, datas)
              
            if ("use_hindsight_relabeling" in self._settings and
                    ("HER" in self._settings["use_hindsight_relabeling"] or
                     "HAC" in self._settings["use_hindsight_relabeling"])and
                "goal_slice_index" in self._settings
                and (self.getSettings()["llc_index"] == agent_)):
                # print("Applying HER")
                (states__, actions__, rewards__, result_states__, falls__, advantage__, 
                 exp_actions__, G_t__, datas__) = self.applyHER(states__, actions__, rewards__, 
                                                       result_states__, falls__, advantage__, exp_actions__, G_t__, datas__)

            if ("hlc_index" in self.getSettings()
                and (self.getSettings()["hlc_index"] == agent_)):
                (states__, actions__, rewards__, result_states__, falls__, advantage__, exp_actions__, G_t__, datas__) = self.dataSkip(self.getAgents()[agent_], states__, 
                                        actions__, rewards__, result_states__, falls__, advantage__, exp_actions__, G_t__, datas__, skip_num=self.getSettings()["hlc_timestep"])
                ### Adjust the max_epoch length to match the true length for the HLC
                self.getAgents()[agent_]._settings["max_epoch_length"] = np.ceil(self.getSettings()["max_epoch_length"]/self.getSettings()["hlc_timestep"])
            if ("llc_episode_length" in self.getSettings()
                and "llc_index" in self.getSettings()
                and (self.getSettings()["llc_index"] == agent_)):
                states__, actions__, rewards__, result_states__, falls__, advantage__, exp_actions__, G_t__
                
                advantage__ = self.processRewards(self.getAgents()[agent_], states__, actions__, rewards__, 
                                                result_states__, falls__, advantage__, exp_actions__, G_t__, 
                                                datas__, 
                                                ep_len=self.getSettings()["llc_episode_length"])
                ### Adjust the max_epoch length to match the true length for the HLC
                self.getAgents()[agent_]._settings["max_epoch_length"] = np.ceil(self.getSettings()["max_epoch_length"]/self.getSettings()["hlc_timestep"])
            if ( "ignore_MRL_agents" in self.getSettings()
                 and (agent_ in self.getSettings()["ignore_MRL_agents"])):
                self.getAgents()[agent_]._settings["train_actor"] = False
                self.getAgents()[agent_]._settings["train_critic"] = False
            else:
                pass
            """
            if ("policy_connections" in self.getSettings()):
                for c in range(len(self.getSettings()["policy_connections"])):
                    if (self.getSettings()["policy_connections"][c][1] == agent_):
                        ### Replace actions with actions of other agent
                        (_, actions__, _, _,
                        _, _, _, _) = self.getSingleAgentData(_states, 
                            _actions, _rewards, _result_states, _falls, _advantage, _exp_actions, _G_t, datas=datas
                            agent_num=self.getSettings()["policy_connections"][c][0])
                        ### Update the llp model weights
                        self.getAgents()[self.getSettings()["policy_connections"][c][1]].updateFrontPolicy(
                            self.getAgents()[self.getSettings()["policy_connections"][c][0]])
                        break
            """
            self.getAgents()[agent_].train(states__, actions__, rewards__, result_states__, falls__, _advantage=advantage__, 
              _exp_actions=exp_actions__, _G_t=G_t__, p=p, datas=datas__)
        
    def recomputeRewards(self, _states, _actions, _rewards, _result_states, _falls, _advantage, 
              _exp_actions, _G_t):
        """
            While learning a reward function re compute the rewards after performing critic and fd updates before policy update
        """
        self.putDataInExpMem(_states, _actions, _rewards, _result_states, _falls, _advantage, _exp_actions, _G_t, recomputeRewards=True)
        
    def getcentralizedPolicyState(self, m, state):
        import numpy as np
        state_ = copy.deepcopy(list(state[m]))
        for bounds_ in [x for i,x in enumerate(state) if i!=m]:
            state_.extend(bounds_)
        ### Add action bounds for other agents
        for bounds_ in [x for i,x in enumerate(self.getSettings()["action_bounds"]) if i!=m]:
            state_.extend(bounds_[0])
                    
        state_ = np.array(state_)
        return state_
    
    def getcentralizedCriticState(self, state):
        import numpy as np
        
        state_ = []
        ### Append other states for each agent
        for agent_ in range(len(self.getAgents())):
            state_.append(copy.deepcopy(list(state[agent_])))
            for state___ in [x for i,x in enumerate(state) if i!=agent_]:
                # states___ = [state_[agent__::len(self.getAgents())] for state_ in _states]
                state_[agent_].extend(state___)
                
        ### Add action for other agents
        for agent_ in range(len(self.getAgents())):
            for bounds_ in [x for i,x in enumerate(self.getSettings()["action_bounds"]) if i!=agent_]:
                act_state = self.getcentralizedPolicyState(agent_, state)
                act_ = self.getAgents()[agent_].predict([act_state], evaluation_=False, p=1.0, sim_index=0, bootstrapping=False)[0] 
                state_[agent_].extend(act_)
                    
        state_ = np.array(state_)
        return state_
        
    def predict(self, state, evaluation_=False, p=None, sim_index=None, bootstrapping=False, use_mbrl=False):
        if self._useLock:
            self._accesLock.acquire()
        
        act = []
        if ( "use_centralized_critic" in self.getSettings()
             and (self.getSettings()["use_centralized_critic"] == True)):
            ### Need to assemble centralized states
            for m in range(len(state)):
                state_ = self.getcentralizedPolicyState(m, state)
                act.append(self.getAgents()[m].predict([state_], evaluation_=evaluation_, p=p, sim_index=sim_index, bootstrapping=bootstrapping)[0])
        else:
            act = [p_.predict([state_], evaluation_=evaluation_, p=p, sim_index=sim_index, bootstrapping=bootstrapping)[0] for p_, state_ in zip(self.getAgents(), state)]
        if self._useLock:
            self._accesLock.release()
        return act
    
    def sample(self, state, evaluation_=False, p=None, sim_index=None, bootstrapping=False, use_mbrl=False, 
               epsilon=1.0, sampling=False, time_step=0):
        if self._useLock:
            self._accesLock.acquire()
        # print ("MARL sample: ", repr(state))
        state_and_goals = []

        act = []
        exp_action = []
        entropy = []
        m_ = len(state)
        if ("perform_multiagent_training" in self.getSettings()):
            m_ = self.getSettings()['perform_multiagent_training']
            
        if ( "use_hrl_logic" in self.getSettings() ### Add LLP state
             and (self.getSettings()["use_hrl_logic"]) == "full" ):
            state = self.addHRLData(state)
        for m in range(m_):
            # by convention the highest levels in the hierarchy come first
            """
            if ( "use_centralized_critic" in self.getSettings()
                 and (self.getSettings()["use_centralized_critic"] == True)):
                ### Need to assemble centralized states
                state_ = self.getcentralizedPolicyState(m, state)

            else:
            """
            state_ = state[m]
            ### This needs to work for multi agent and single policy MultiAgent stuff
            m = min(m, self.getSettings()["perform_multiagent_training"]-1)

            """ Brandon:
            This code manages converting actions from higher agents into goals of lower agents.
            Assume that state does not have a goal concatenates to it.
            """
            state_ = np.array(state_)
            if ( "use_hrl_logic" in self.getSettings()
                 and (self.getSettings()["use_hrl_logic"])
                 and (m == 1) ):
                # if index is not the highest level policy, which has no goal to concat.
                # concat on the action of the higher level onto the state
                goal = np.array(self.latest_actions[m - 1][0])
                """
                while len(goal.shape) < len(state_.shape):
                    # if the state has extra dimensions, then copy those dimensions
                    # assume that the goal and the state always have a batch dimension first
                    goal = goal[:, None, ...]
                    next_expansion_dim = len(state_.shape) - len(goal.shape) - 1
                    goal = np.tile(goal, [state_.shape[next_expansion_dim]] + [1 for _i in goal.shape])
                # by this point state and goals should be the same along every axis except the last
                # if this is false then one is likely not a simple vector
                assert all([i == j for i, j in zip(state_.shape[:-1], goal.shape[:-1])])
                """
                # print ("state: ", state_, " goal: ",  goal)
                state_ = np.concatenate([np.array(state_), goal], -1)

            use_hle = ( "hlc_index" in self.getSettings()
                and "llc_index" in self.getSettings()
                and "high_level_exploration_samples" in self.getSettings()
                and self.getSettings()["hlc_index"] == m
                and "use_high_level_exploration" in self.getSettings()
                and self.getSettings()["use_high_level_exploration"] == True
                and ("exploration_processing" in self.getSettings()
                     and self.getSettings()["exploration_processing"] is not False))

            if use_hle:
                num_samples = self.getSettings()["high_level_exploration_samples"]
                state_ = np.array([state_ for i in range(num_samples)])
                candidate_actions, candidate_exp_acts, entropys, _ = self.getAgents()[m].sample(
                    state_,
                    evaluation_=evaluation_, p=p, sim_index=sim_index, bootstrapping=bootstrapping,
                    epsilon=epsilon, sampling=sampling)

                ### Assume that z_k is at the end of the state
                ### These shapes don't quite line up. The llp state can be a different size than the hlp shape.
                state_llp = copy.deepcopy(np.array([state[self.getSettings()["llc_index"]] for i in range(num_samples)]) )               ### Replace the last few coli
                state_llp[:, -len(candidate_actions[0]):] =  candidate_actions
                llc_values = self.getAgents()[self.getSettings()["llc_index"]].getPolicy().q_values2(state_llp)
                
                if ("exploration_processing" in self.getSettings()
                     and self.getSettings()["exploration_processing"] == "argmax"):
                    
                    idx_ = np.argmax(llc_values)
                elif ("exploration_processing" in self.getSettings()
                     and self.getSettings()["exploration_processing"] == "reweight"):
                    ### Other options
                    llc_values_ = llc_values.flatten() - np.min(llc_values)
                    llc_weights = llc_values_ / np.sum(llc_values_)
                    idx_ = np.random.choice(range(num_samples), p=llc_weights)
                else:
                    ### Equally weighted
                    idx_ = np.random.choice(range(num_samples))

                action = [candidate_actions[idx_]]
                exp_act = [candidate_exp_acts[idx_]]
                entropy_ = entropys[idx_]

            else:
                # print(state_.shape, m)
                state_tmp = state_
                if ( "use_centralized_critic" in self.getSettings()
                 and (self.getSettings()["use_centralized_critic"] == True)):
                ### Need to assemble centralized states
                    state_tmp = state_
                    state_ = self.getcentralizedPolicyState(m, state)
                    
                (action, exp_act, entropy_, _) = self.getAgents()[m].sample(
                    [state_],
                    evaluation_=evaluation_, p=p, sim_index=sim_index, bootstrapping=bootstrapping,
                    epsilon=epsilon, sampling=sampling)
                state_ = state_tmp 

            if ("use_hrl_logic" in self.getSettings()
                 and (self.getSettings()["use_hrl_logic"]) 
                 and ((time_step == 0) or (time_step % self.time_skips[m] == 0) )):
                # if this value is true then this level in the hierarchy is active
                # otherwise use the actions exp actions and entropy from previous steps
                self.latest_actions[m] = action
                self.latest_exp_act[m] = exp_act
                self.latest_entropy[m] = entropy_
            else:
                self.latest_actions[m] = action
                self.latest_exp_act[m] = exp_act
                self.latest_entropy[m] = entropy_
            """
            if (self.getSettings()["policy_connections"][0][1] == m):
                state[self.getSettings()["policy_connections"][0][0]][
                    self.getSettings()["goal_slice_index"]:] =  self.latest_actions[m]
            """
            state_and_goals.append(state_)
            act.append(self.latest_actions[m][0])
            exp_action.append(self.latest_exp_act[m][0])
            entropy.append(self.latest_entropy[m])

        if self._useLock:
            self._accesLock.release()
        # print ("act: ", repr(act))
        # print ("exp_action: ", repr(exp_action))

        return (act, exp_action, entropy, state_and_goals)
    
    def addHRLData(self, observation):
        ### The the state data for the LLP
        obs = [observation[0]]
        LLP_state = observation[0][:self.getSettings()['goal_slice_index']]
        # llp_goal = self.latest_actions[m - 1][0]
        # state_ = np.concatenate([np.array(state_), goal], -1)
        # observation = np.concatenate([observation, LLP_state], 0)
        obs.append(LLP_state)
        # print ("obs :", obs)
        return obs
    
    def addHRLReward(self, observation, nextObservation, reward_, done, info):
        ### THe next Observation is not going to have the HRL state structure yet.
        ### The the state data for the LLP
        r_ = [[reward_]]
        LLP_state = nextObservation[0][:self.getSettings()['goal_slice_index']]
        llp_goal = observation[self.getSettings()['llc_index']][0]
        # state_ = np.concatenate([np.array(state_), goal], -1)
        reward = -np.sum(np.fabs((LLP_state-llp_goal)))
        # observation.append(LLP_state)
        # reward_.append([reward])
        r_.append([reward])
        # print("reward_ :", r_)
        return r_
        
        
    def predict_std(self, state, evaluation_=False, p=1.0):
        if self._useLock:
            self._accesLock.acquire()
        
        std = []
        if ( "use_centralized_critic" in self.getSettings()
             and (self.getSettings()["use_centralized_critic"] == True)):
            ### Need to assemble centralized states
            import numpy as np
            for m in range(len(state)):
                state_ = self.getcentralizedPolicyState(m, state)
                std.append(self.getAgents()[m].predict_std([state_], evaluation_=evaluation_, p=p)[0])
        else:
            std = [p_.predict_std([state_], p=p)[0] for p_, state_ in zip(self.getAgents(), state) ]
        if self._useLock:
            self._accesLock.release()
        return std
    
    def predictWithDropout(self, state):
        if self._useLock:
            self._accesLock.acquire()
        act = [p_.predictWithDropout([state_])[0] for p_, state_ in zip(self.getAgents(), state) ]
        if self._useLock:
            self._accesLock.release()
        return act
    
    def predictNextState(self, state, action):
        return self._fd.predict(state, action)
    
    def setNoise(self, noise):
        """
            Normalized states for learning
        """
        if self._useLock:
            self._accesLock.acquire()
        q = [p_.setNoise(state_) for p_, state_ in zip(self.getAgents(), noise) ]
        if self._useLock:
            self._accesLock.release()
        return None
    
    def q_value(self, state):
        """
            Non normalized state in the env space
        """
        if self._useLock:
            self._accesLock.acquire()
        q = []
        if ( "use_centralized_critic" in self.getSettings()
             and (self.getSettings()["use_centralized_critic"] == True)):
            ### Need to assemble centralized states
            import numpy as np
            for m in range(len(state)):
                state_ = self.getcentralizedPolicyState(m, state)
                q.append(self.getAgents()[m].q_value([state_])[0])
        else:
            q = [p_.q_value([state_])[0] for p_, state_ in zip(self.getAgents(), state) ]
        if self._useLock:
            self._accesLock.release()
        return q
    
    def q_values(self, state):
        """
            Normalized states for learning
        """
        if self._useLock:
            self._accesLock.acquire()
        q = [p_.q_values([state_])[0] for p_, state_ in zip(self.getAgents(), state) ]
        if self._useLock:
            self._accesLock.release()
        return q
    
    def q_values2(self, state, agent_id):
        """
            
        """
        state_ = np.array(state)
        if ( "use_centralized_critic" in self.getSettings()
             and (self.getSettings()["use_centralized_critic"] == True)):
            ### Need to assemble centralized states
            # state_ = self.getcentralizedCriticState(state)[agent_id[0][0]]
            q = self.getAgents()[agent_id[0][0]].q_values2(state_, agent_id)
        else:
            q = self.getAgents()[agent_id[0][0]].q_values2(state_, agent_id)
        if self._useLock:
            self._accesLock.release()
        return q
    
    def bellman_error(self):
        if self._useLock:
            self._accesLock.acquire()
        errors_ = []
        for m in range(len(self.getAgents())):
        # err = [p_.bellman_error(state, action, reward, result_state, fall) for p_, state_, action_, reward_, result_state_, fall_ in zip(self.getAgents(), state, action, reward, result_state, fall)]
            err = self.getAgents()[m].bellman_error()
            errors_.append(err)  
        if self._useLock:
            self._accesLock.release()
        return errors_
        
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
            
    def getStateBounds(self):
        return [p.getStateBounds() for p in self.getAgents()]
    def getActionBounds(self):
        return [p.getActionBounds() for p in self.getAgents()]
    def getRewardBounds(self):
        return [p.getRewardBounds() for p in self.getAgents()]
    
    def getFDStateBounds(self):
        return [p.getFDStateBounds() for p in self.getAgents()]
    def getFDActionBounds(self):
        return [p.getFDActionBounds() for p in self.getAgents()]
    def getFDRewardBounds(self):
        return [p.getFDRewardBounds() for p in self.getAgents()]
    
    def setStateBounds(self, bounds):
        [p.setStateBounds(bounds_) for p, bounds_ in zip(self.getAgents(), bounds)]
    def setActionBounds(self, bounds):
        [p.setActionBounds(bounds_) for p, bounds_ in zip(self.getAgents(), bounds)]
    def setRewardBounds(self, bounds):
        [p.setRewardBounds(bounds_) for p, bounds_ in zip(self.getAgents(), bounds)]
        
    def setFDStateBounds(self, bounds):
        [p.setFDStateBounds(bounds_) for p, bounds_ in zip(self.getAgents(), bounds)]
    def setFDActionBounds(self, bounds):
        [p.setFDActionBounds(bounds_) for p, bounds_ in zip(self.getAgents(), bounds)]
    def setFDRewardBounds(self, bounds):
        [p.setFDRewardBounds(bounds_) for p, bounds_ in zip(self.getAgents(), bounds)]
        
    def _updateScaling(self):
        [p._updateScaling() for p in self.getAgents()]
            
    def insertTuple(self, tuple):
        ([state], [action], [resultState], [reward], [fall], [G_t], [exp_action], [adv], data) = tuple
        # print("data: ", data)
        self.getAgents()[data["agent_id"][0]].insertTuple(tuple)
        
    def insertTrajectory(self, states, actions, result_states, rewards, falls, G_ts, advantage, exp_actions, data):
        self.getAgents()[data["agent_id"][0][0]].insertTrajectory(states, actions, result_states, rewards, falls, G_ts, advantage, exp_actions, data)
        
    def insertFDTuple(self, tuple):
        ([state], [action], [resultState], [reward], [fall], [G_t], [exp_action], [adv], data) = tuple
        self.getAgents()[data["agent_id"][0]].insertFDTuple(tuple)
    def insertFDTrajectory(self, states, actions, result_states, rewards, falls, G_ts, advantage, exp_actions, data):
        self.getAgents()[data["agent_id"][0][0]].insertFDTrajectory(states, actions, result_states, rewards, falls, G_ts, advantage, exp_actions, data)
        
    def samples(self):
        return self.getAgents()[0].samples()
    
    def get_batch(self, batch_size, m=0):
        return self.getAgents()[m].get_batch(batch_size)
    
    def get_multitask_trajectory_batch(self, batch_size, m=0):
        return self.getAgents()[m].get_multitask_trajectory_batch(batch_size)
    
    def getFDBatch(self, batch_size, m=0):
        return self.getAgents()[m].getFDBatch(batch_size)
    
    def getFDmultitask_trajectory_batch(self, batch_size, m=0):
        return self.getAgents()[m].getFDmultitask_trajectory_batch(batch_size)
        
    def saveTo(self, directory, bestPolicy=False, bestFD=False, suffix=""):
        from util.SimulationUtil import getAgentName
        suffix = ""
        # self.getPolicy().saveTo(directory+getAgentName()+suffix )
        [self.getAgents()[i].saveTo(directory, bestPolicy=bestPolicy, bestFD=bestFD, suffix=str(i)  ) for i in range(len(self.getAgents()))]
        
    def loadExperience(self, directory):
        [self.getAgents()[i].loadExperience(directory+str(i)  ) for i in range(len(self.getAgents()))]
        
    def loadFDExperience(self, directory):
        [self.getAgents()[i].loadFDExperience(directory+str(i)  ) for i in range(len(self.getAgents()))]
        
    def loadFrom(self, directory, best=False):
        import dill
        from util.SimulationUtil import getAgentName
        suffix = ".pkl"
        if ( bestPolicy == True):
            suffix = "_Best.pkl"
        file_name=directory+getAgentName()+"_Best.pkl"
        f = open(file_name, 'rb')
        self.setPolicy(dill.load(f))
        f.close()
        
        if (self._settings['train_forward_dynamics']):
            file_name_dynamics=directory+"forward_dynamics_"+suffix
            f = open(file_name_dynamics, 'rb')
            self.setForwardDynamics(dill.load(f))
            f.close()
            
            
    def finish(self):
        if (self.getSampler() is not None):
            self.getSampler().finish()
        

