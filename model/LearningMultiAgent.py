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
# np.set_printoptions(threshold=np.nan)

class LearningMultiAgent(LearningAgent):
    
    def __init__(self, settings_):
        super(LearningMultiAgent,self).__init__(settings_=settings_)
        self._useLock = False
        if self._useLock:
            self._accesLock = threading.Lock()
        self._agents = [LearningAgent(self.getSettings()) for i in range(self.getSettings()["perform_multiagent_training"])]
        
    def getAgents(self):
        return self._agents
    
    def reset(self):
        [p.reset() for p in self.getAgents()]
        
    def getPolicy(self):
        if self._useLock:
            self._accesLock.acquire()
        pol = self._pol
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
        fd = self._fd
        if self._useLock:
            self._accesLock.release()
        return fd
                
    def setForwardDynamics(self, fd):
        if self._useLock:
            self._accesLock.acquire()
        self._fd = fd
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
        # self.getPolicy().setSettings(settings)
        # self.getForwardDynamics().setSettings(settings)
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
    
    # @profile(precision=5)
    def train(self, _states, _actions, _rewards, _result_states, _falls, _advantage=None, 
              _exp_actions=None, _G_t=None, p=1.0):
        import numpy as np 
        
        if ( "use_centralized_critic" in self.getSettings()
             and (self.getSettings()["use_centralized_critic"] == True)):
            
            for agent_ in range(len(self.getAgents())):
                # print ("_states: ", np.array(_states).shape)
                # print ("_states[0][0]: ", np.array(_states[0][0]).shape)
                # print ("_states[0][1]: ", np.array(_states[0][1]).shape)
                # print ("_states[0][2]: ", np.array(_states[0][2]).shape)
                ### Pull out the state for each agent, start at agent index and skip every number of agents 
                states__ = [state_[agent_::len(self.getAgents())] for state_ in _states]
                actions__ = [state_[agent_::len(self.getAgents())] for state_ in _actions]
                rewards__ = [state_[agent_::len(self.getAgents())] for state_ in _rewards]
                result_states__ = [state_[agent_::len(self.getAgents())] for state_ in _result_states]
                result_states_tmp = [state_[agent_::len(self.getAgents())] for state_ in _result_states]
                falls__ = [state_[agent_::len(self.getAgents())] for state_ in _falls]
                advantage__ = [state_[agent_::len(self.getAgents())] for state_ in _advantage]
                exp_actions__ = [state_[agent_::len(self.getAgents())] for state_ in _exp_actions]
                G_t__ = [state_[agent_::len(self.getAgents())] for state_ in _G_t]
                
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
                            # states__[tar][s] = np.array(list(states__[tar][s]).extend(states___[tar][s]))
                            # print ("states__[tar][s]: ", np.array(states__[tar][s]).shape)
                            # print ("states__[tar][s]: ", states__[tar][s])
                        # print ("states__[s]: ", np.array(states__[tar]).shape)
                        
                ### Collect the actions of the other agents as additional state info.
                for agent__ in [i for i,x in enumerate(self.getAgents()) if i!=agent_]:
                    actions___ = [state_[agent__::len(self.getAgents())] for state_ in _actions]
                    for tar in range(len(states__)):
                        for s in range(len(states__[tar])):
                            # states__[tar][s] = np.array(list(states__[tar][s]).extend(actions___[tar][s]))
                            state___ = np.concatenate((states__[tar][s],actions___[tar][s]), axis=0)
                            # print ("state: ", np.array(state___).shape)
                            states__[tar][s] = state___
                            ### Add garbage action to this tmp next state to create corect size state for other agent target action request
                            result_states_tmp[tar][s] = np.concatenate((result_states_tmp[tar][s],actions___[tar][s]), axis=0)
                        # print ("states__[s]: ", np.array(states__[tar]).shape)
                
                ### Now that we have data of the correct size to ask for target actions of other agents, get those target actions.
                for agent__ in [i for i,x in enumerate(self.getAgents()) if i!=agent_]:
                    # actions___ = [state_[agent__::len(self.getAgents())] for state_ in _actions]
                    # result_states___ = [state_[agent__::len(self.getAgents())] for state_ in result_states_tmp]
                    ### Get result state for this other agent
                    result_states_tmp_agent = np.array([state_[agent__::len(self.getAgents())] for state_ in _result_states])
                    result_states_tmp = np.array(result_states_tmp)
                    # print ("result_states_tmp_agent: ", result_states_tmp_agent.shape)
                    # result_states_tmp[]
                    for tar in range(len(result_states_tmp)):
                        # print ("result_states_tmp_agent[tar,:]: ", result_states_tmp_agent[tar,:].shape)
                        # print ("result_states_tmp[tar,:,:result_states_tmp_agent.shape[-1]]: ", np.array(result_states_tmp[tar,:,:result_states_tmp_agent.shape[-1]]).shape)
                        result_states_tmp[tar,:,:result_states_tmp_agent.shape[-1]] = result_states_tmp_agent[tar,:]
                        # print ("result_states___[tar]: ", np.array(result_states_tmp[tar]).shape)
                        # print ("result_states__[s] before : ", np.array(result_states__[tar]).shape)
                        target_actions = self.getAgents()[agent__].predict_target(result_states_tmp[tar])
                        # print ("target_actions: ", np.array(target_actions).shape)
                        for s in range(len(result_states___[tar])):
                            # states__[tar][s] = np.array(list(states__[tar][s]).extend(actions___[tar][s]))
                            # states__[tar][s] = np.concatenate((states__[tar][s],actions___[tar][s]), axis=0)
                            target_res_state = np.concatenate((result_states__[tar][s],target_actions[s]), axis=0)
                            # print ("target_res_state: ", np.array(target_res_state).shape)
                            # print ("target_res_state: ", target_res_state)
                            result_states__[tar][s] = np.array(target_res_state)
                        # print ("result_states__[s]: ", np.array(result_states__[tar]).shape) 
                
                # print ("states__: ", np.array(states__).shape)
                # print ("result_states__: ", np.array(result_states__).shape) 
                self.getAgents()[agent_].train(states__, actions__, rewards__, result_states__, falls__, _advantage=advantage__, 
                  _exp_actions=exp_actions__, _G_t=G_t__, p=p)
        else:
            for agent_ in range(len(self.getAgents())):
                # print ("_states: ", np.array(_states).shape)
                # print ("_states[0][0]: ", np.array(_states[0][0]).shape)
                # print ("_states[0][1]: ", np.array(_states[0][1]).shape)
                # print ("_states[0][2]: ", np.array(_states[0][2]).shape)
                ### Pull out the state for each agent, start at agent index and skip every number of agents 
                states__ = [state_[agent_::len(self.getAgents())] for state_ in _states]
                actions__ = [state_[agent_::len(self.getAgents())] for state_ in _actions]
                rewards__ = [state_[agent_::len(self.getAgents())] for state_ in _rewards]
                result_states__ = [state_[agent_::len(self.getAgents())] for state_ in _result_states]
                falls__ = [state_[agent_::len(self.getAgents())] for state_ in _falls]
                advantage__ = [state_[agent_::len(self.getAgents())] for state_ in _advantage]
                exp_actions__ = [state_[agent_::len(self.getAgents())] for state_ in _exp_actions]
                G_t__ = [state_[agent_::len(self.getAgents())] for state_ in _G_t]
                
                # print ("states__: ", np.array(states__).shape) 
                self.getAgents()[agent_].train(states__, actions__, rewards__, result_states__, falls__, _advantage=advantage__, 
                  _exp_actions=exp_actions__, _G_t=G_t__, p=p)
        
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
        q = self.getAgents()[agent_id[0][0]].q_values2(state, agent_id)
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
        
    def _updateScaling(self):
        [p._updateScaling() for p in self.getAgents()]
            
    def insertTuple(self, tuple):
        ([state], [action], [resultState], [reward], [fall], [G_t], [exp_action], [adv]) = tuple
        self.getAgents()[fall[0]].insertTuple(tuple)
        
    def insertTrajectory(self, states, actions, result_states, rewards, falls, G_ts, advantage, exp_actions):
        self.getAgents()[falls[0][0]].insertTrajectory(states, actions, result_states, rewards, falls, G_ts, advantage, exp_actions)
        
    def samples(self):
        return self.getAgents()[0].samples()
    
    def get_batch(self, size_, m):
        return self.getAgents()[m].get_batch(size_)
        
    def saveTo(self, directory, bestPolicy=False, bestFD=False, suffix=""):
        from util.SimulationUtil import getAgentName
        suffix = ""
        # self.getPolicy().saveTo(directory+getAgentName()+suffix )
        [self.getAgents()[i].saveTo(directory, bestPolicy=bestPolicy, bestFD=bestFD, suffix=str(i)  ) for i in range(len(self.getAgents()))]
        
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
        

