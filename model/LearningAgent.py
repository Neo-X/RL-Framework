"""
    An interface class for Agents to be used in the system.

"""
from multiprocessing import Process, Queue
# from pathos.multiprocessing import Pool
import threading
import time
from model.AgentInterface import AgentInterface
from model.ModelUtil import *
import os
import numpy
import copy
# numpy.set_printoptions(threshold=numpy.nan)

class LearningAgent(AgentInterface):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):
        super(LearningAgent,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        self._useLock = False
        if self._useLock:
            self._accesLock = threading.Lock()
        self._pol = None
        self._fd = None
        
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
        
    def setSettings(self, settings):
        self._settings = settings
        
    def setExperience(self, experienceBuffer):
        self._expBuff = experienceBuffer 
    def getExperience(self):
        return self._expBuff 
    
    def train(self, _states, _actions, _rewards, _result_states, _falls):
        if self._useLock:
            self._accesLock.acquire()
        cost = 0
        for i in range(self._settings['critic_updates_per_actor_update']):
            _states, _actions, _result_states, _rewards, _falls = self._expBuff.get_batch(self._settings["batch_size"])
            # print ("Updating Critic")
            cost = self._pol.trainCritic(states=_states, actions=_actions, rewards=_rewards, result_states=_result_states, falls=_falls)
            if not np.isfinite(cost) or (cost > 500) :
                print ("States: " + str(_states) + " ResultsStates: " + str(_result_states) + " Rewards: " + str(_rewards) + " Actions: " + str(_actions))
                print ("Training cost is Odd: ", cost)
        if (self._settings['train_actor']):
            cost_ = self._pol.trainActor(states=_states, actions=_actions, rewards=_rewards, result_states=_result_states, falls=_falls)
        dynamicsLoss = 0 
        if (self._settings['train_forward_dynamics']):
            dynamicsLoss = self._fd.train(states=_states, actions=_actions, result_states=_result_states)
        if self._useLock:
            self._accesLock.release()
        return (cost, dynamicsLoss) 
    
    def predict(self, state):
        if self._useLock:
            self._accesLock.acquire()
        act = self._pol.predict(state)
        if self._useLock:
            self._accesLock.release()
        return act
    
    def predictWithDropout(self, state):
        if self._useLock:
            self._accesLock.acquire()
        act = self._pol.predictWithDropout(state)
        if self._useLock:
            self._accesLock.release()
        return act
    
    def predictNextState(self, state, action):
        return self._fd.predict(state, action)
    
    def q_value(self, state):
        if self._useLock:
            self._accesLock.acquire()
        q = self._pol.q_value(state)
        if self._useLock:
            self._accesLock.release()
        return q
    
    def bellman_error(self, state, action, reward, result_state, fall):
        if self._useLock:
            self._accesLock.acquire()
        err = self._pol.bellman_error(state, action, reward, result_state, fall)
        if self._useLock:
            self._accesLock.release()
        return err
        
    def initEpoch(self, exp):
        pass

import copy
# class LearningWorker(threading.Thread):
class LearningWorker(Process):
    def __init__(self, input_exp_queue, agent, namespace, learningNamespace):
        super(LearningWorker, self).__init__()
        self._input_queue= input_exp_queue
        self._namespace = namespace
        self._agent = agent
        self._learningNamespace = learningNamespace
        
        
    def run(self):
        print ('Worker started')
        # do some initialization here
        step_ = 0
        iterations_=0
        # self._agent._expBuff = self._namespace.experience
        while True:
            tmp = self._input_queue.get()
            if tmp == None:
                break
            #if len(tmp) == 6:
                # self._input_queue.put(tmp)
            #    continue # don't learn from eval tuples
            (state_, action, reward, resultState, fall) = tmp
            # print ("Learner Size of state input Queue: " + str(self._input_queue.qsize()))
            # self._agent._expBuff = self._namespace.experience
            if self._agent._settings['action_space_continuous']:
                # self._agent._expBuff.insert(norm_state(state_, self._agent._state_bounds), 
                #                            norm_action(action, self._agent._action_bounds), norm_state(resultState, self._agent._state_bounds), [reward])
                self._agent._expBuff.insert(state_, action, resultState, [reward], [fall])
                # print ("Experience buffer size: " + str(self._namespace.experience.samples()))
                # print ("Reward Scale: ", self._agent._reward_bounds)
                # print ("Reward Scale Model: ", self._agent._pol.getRewardBounds())
            else:
                self._agent._expBuff.insert(self._agent._state_bounds, [action], resultState, [reward], [fall])
            # print ("Learning agent experience size: " + str(self._agent._expBuff.samples()))
            step_ += 1
            if self._agent._expBuff.samples() > self._agent._settings["batch_size"] and ((step_ >= self._agent._settings['sim_action_per_training_update']) ):
                __states, __actions, __result_states, __rewards, __falls = self._agent._expBuff.get_batch(self._agent._settings["batch_size"])
                # print ("States: " + str(__states) + " ResultsStates: " + str(__result_states) + " Rewards: " + str(__rewards) + " Actions: " + str(__actions))
                (cost, dynamicsLoss) = self._agent.train(_states=__states, _actions=__actions, _rewards=__rewards, _result_states=__result_states, _falls=__falls)
                # print ("Master Agent Running training step, cost: " + str(cost) + " PID " + str(os.getpid()))
                # print ("Updated parameters: " + str(self._agent._pol.getNetworkParameters()[3]))
                if not np.isfinite(cost):
                    print ("States: " + str(__states) + " ResultsStates: " + str(__result_states) + " Rewards: " + str(__rewards) + " Actions: " + str(__actions))
                    print ("Training cost is Nan: ", cost)
                    sys.exit()
                # if (step_ % 10) == 0: # to help speed things up
                self._namespace.agentPoly = copy.deepcopy(self._agent.getPolicy().getNetworkParameters())
                if (self._agent._settings['train_forward_dynamics']):
                    self._namespace.forwardNN = copy.deepcopy(self._agent.getForwardDynamics().getNetworkParameters())
                self._learningNamespace.experience = copy.deepcopy(self._agent._expBuff)
                step_=0
            iterations_+=1
        print ("Learning Worker Complete:")
        
    def updateExperience(self):
        self._agent._expBuff = self._learningNamespace.experience
        
    def updateModel(self):
        print ("Updating model to: ", self._namespace.model)
        self._agent.setPolicy(copy.deepcopy(self._namespace.model))