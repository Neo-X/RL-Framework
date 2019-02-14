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
        self._pol = None
        self._fd = None
        self._sampler = None
        self._expBuff = None
        self._expBuff_FD = None
        
    def reset(self):
        [p.reset() for p in self.getPolicy()]
        if (self._settings['train_forward_dynamics']):
            [f.reset() for f in self.getForwardDynamics()]
        
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
        self._pol = pol
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
        return [p.getNetworkParameters() for p in self.getPolicy()]
    def getFDNetworkParameters(self):
        return [p.getNetworkParameters() for p in self.getForwardDynamics()]
    def getRewardNetworkParameters(self):
        return [p.getNetworkParameters() for p in self.getRewardModel()]
    
    def setPolicyNetworkParameters(self, params):
        return [p.setNetworkParameters(param) for p, param in zip(self.getPolicy(), params)]
    def setFDNetworkParameters(self, params):
        return [p.setNetworkParameters(param) for p, param in zip(self.getForwardDynamics(), params)]
    def setRewardNetworkParameters(self, params):
        return [p.setNetworkParameters(param) for p, param in zip(self.getRewardModel(), params)]
    
    
        
    def setExperience(self, experienceBuffer):
        self._expBuff = experienceBuffer 
    def getExperience(self):
        return self._expBuff
    
    def setFDExperience(self, experienceBuffer):
        self._expBuff_FD = experienceBuffer 
    def getFDExperience(self):
        return self._expBuff_FD  
    
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
        
    def recomputeRewards(self, _states, _actions, _rewards, _result_states, _falls, _advantage, 
              _exp_actions, _G_t):
        """
            While learning a reward function re compute the rewards after performing critic and fd updates before policy update
        """
        self.putDataInExpMem(_states, _actions, _rewards, _result_states, _falls, _advantage, _exp_actions, _G_t, recomputeRewards=True)
        
    def predict(self, state, evaluation_=False, p=None, sim_index=None, bootstrapping=False, use_mbrl=False):
        if self._useLock:
            self._accesLock.acquire()
        # import numpy as np
        # print ("state: ", np.array(state).shape, state)
        # print ("state: ", state)
        state = self.processState(state)
        # print ("state after: ", np.array(state).shape, state)
        if (use_mbrl):
            action = self.getSampler().predict(state, p=p, sim_index=sim_index, bootstrapping=bootstrapping)
            act = [action]
        else:
            """
            act = []
            for p_, state_ in zip(self.getPolicy(), state):
                act_ = p_.predict([state_], evaluation_=evaluation_, p=p, sim_index=sim_index, bootstrapping=bootstrapping)
                act.append(act_)
            """ 
            act = [p_.predict([state_], evaluation_=evaluation_, p=p, sim_index=sim_index, bootstrapping=bootstrapping)[0] for p_, state_ in zip(self.getPolicy(), state)]
        if self._useLock:
            self._accesLock.release()
        return act
    
    def predict_std(self, state, evaluation_=False, p=1.0):
        if self._useLock:
            self._accesLock.acquire()
        state = self.processState(state)
        std = [p_.predict_std([state_], p=p)[0] for p_, state_ in zip(self.getPolicy(), state) ]
        if self._useLock:
            self._accesLock.release()
        return std
    
    def predictWithDropout(self, state):
        if self._useLock:
            self._accesLock.acquire()
        state = self.processState(state)
        # act = self.getPolicy().predictWithDropout(state)
        act = [p_.predictWithDropout([state_])[0] for p_, state_ in zip(self.getPolicy(), state) ]
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
        state = self.processState(state)
        # q = self.getPolicy().q_value(state)
        q = [p_.q_value([state_])[0] for p_, state_ in zip(self.getPolicy(), state) ]
        if self._useLock:
            self._accesLock.release()
        return q
    
    def q_values(self, state):
        """
            Normalized states for learning
        """
        if self._useLock:
            self._accesLock.acquire()
        state = self.processState(state)
        # q = self.getPolicy().q_values(state)
        q = [p_.q_values([state_])[0] for p_, state_ in zip(self.getPolicy(), state) ]
        if self._useLock:
            self._accesLock.release()
        return q
    
    def q_values2(self, state):
        state = self.processState(state)
        # q = self.getPolicy().q_values2(state)
        q = [p_.q_values2([state_])[0] for p_, state_ in zip(self.getPolicy(), state) ]
        if self._useLock:
            self._accesLock.release()
        return q
    
    def bellman_error(self, state, action, reward, result_state, fall):
        if self._useLock:
            self._accesLock.acquire()
        """
        if ("use_dual_state_representations" in self.getSettings()
            and (self.getSettings()["use_dual_state_representations"] == True)):
            print ("State: ", state)
            state = state[0]
            print ("State: ", state)
        """
        if ("use_hack_state_trans" in self.getSettings()
            and (self.getSettings()["use_hack_state_trans"] == True)):
            import numpy as np
            state = np.array(state)
            state = state[:,:len(self.getStateBounds()[0])]
        # err = self.getPolicy().bellman_error(state, action, reward, result_state, fall)
        err = [p_.bellman_error(state, action, reward, result_state, fall) for p_, state_, action_, reward_, result_state_, fall_ in zip(self.getPolicy(), state, action, reward, result_state, fall)] 
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
            
    def getStateBounds(self):
        return [p.getStateBounds() for p in self.getPolicy()]
    def getActionBounds(self):
        return [p.getActionBounds() for p in self.getPolicy()]
    def getRewardBounds(self):
        return [p.getRewardBounds() for p in self.getPolicy()]
    
    def getFDStateBounds(self):
        return [p.getStateBounds() for p in self.getForwardDynamics()]
    def getFDActionBounds(self):
        return [p.getActionBounds() for p in self.getForwardDynamics()]
    def getFDRewardBounds(self):
        return [p.getRewardBounds() for p in self.getForwardDynamics()]
    
    def setStateBounds(self, bounds):
        import numpy as np
        bounds = np.array(bounds)
        [p.setStateBounds(bounds_) for p, bounds_ in zip(self.getPolicy(), bounds)] 
        if (self._settings['train_forward_dynamics']):
            if ("use_dual_state_representations" in self._settings
                and (self._settings["use_dual_state_representations"] == True)):
                pass
            else:
                # self.getForwardDynamics().setStateBounds(bounds)
                [p.setStateBounds(bounds_) for p, bounds_ in zip(self.getForwardDynamics(), bounds)] 
                if ( 'keep_seperate_fd_exp_buffer' in self._settings 
                     and (self._settings['keep_seperate_fd_exp_buffer'] == True)
                     and (self.getFDExperience() is not None)):
                    assert False ### Not supported
                    # self.getForwardDynamics().setStateBounds(self.getFDExperience().getStateBounds())
                    # self.getFDExperience().setStateBounds(bounds)

    def setActionBounds(self, bounds):
        import numpy as np
        bounds = np.array(bounds)
        # self.getPolicy().setActionBounds(bounds)
        [p.setActionBounds(bounds_) for p, bounds_ in zip(self.getPolicy(), bounds)]
        if (self._settings['train_forward_dynamics']):
            
            if ( 'keep_seperate_fd_exp_buffer' in self._settings 
                 and (self._settings['keep_seperate_fd_exp_buffer'])
                 and (self.getFDExperience() is not None)):
                # self.getFDExperience().setActionBounds(bounds)
                assert False
                # self.getForwardDynamics().setActionBounds(self.getFDExperience().getActionBounds())
            else:
                # self.getForwardDynamics().setActionBounds(bounds)
                [p.setActionBounds(bounds_) for p, bounds_ in zip(self.getForwardDynamics(), bounds)]
                
    def setRewardBounds(self, bounds):
        import numpy as np
        bounds = np.array(bounds)
        # self.getPolicy().setRewardBounds(bounds)
        [p.setRewardBounds(bounds_) for p, bounds_ in zip(self.getPolicy(), bounds)]
        if (self._settings['train_forward_dynamics']):
            # self.getForwardDynamics().setRewardBounds(bounds)
            [p.setRewardBounds(bounds_) for p, bounds_ in zip(self.getForwardDynamics(), bounds)]
            if ( 'keep_seperate_fd_exp_buffer' in self._settings 
                 and (self._settings['keep_seperate_fd_exp_buffer'])
                 and (self.getFDExperience() is not None)):
                # self.getFDExperience().setRewardBounds(bounds)
                assert False
                # self.getForwardDynamics().setRewardBounds(self.getFDExperience().getRewardBounds())
            
    def saveTo(self, directory, bestPolicy=False, bestFD=False):
        from util.SimulationUtil import getAgentName
        suffix = ""
        if ( bestPolicy == True):
            suffix = "_Best"
        self.getPolicy().saveTo(directory+getAgentName()+suffix )
        [self.getPolicy()[i].saveTo(directory+getAgentName()+suffix+str(i) ) for i in range(len(self.getPolicy()))]
        
        suffix = ""
        if ( bestFD == True):
            suffix = "_Best"
        if (self._settings['train_forward_dynamics']):
            # self.getForwardDynamics().saveTo(directory+"forward_dynamics"+suffix)
            [self.getForwardDynamics()[i].saveTo(directory+"forward_dynamics"+suffix+str(i) ) for i in range(len(self.getForwardDynamics()))]
        
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
        

