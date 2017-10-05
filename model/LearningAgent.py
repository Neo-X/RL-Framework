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
        self._sampler = None
        
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
        
    def setSettings(self, settings):
        self._settings = settings
        
    def setExperience(self, experienceBuffer):
        self._expBuff = experienceBuffer 
    def getExperience(self):
        return self._expBuff 
    
    def train(self, _states, _actions, _rewards, _result_states, _falls, _advantage=None, _exp_actions=None):
        if self._useLock:
            self._accesLock.acquire()
        cost = 0
        if self._settings['on_policy']:
            if ( ('clear_exp_mem_on_poli' in self._settings) and 
                 self._settings['clear_exp_mem_on_poli']):
                self._expBuff.clear()
            
            ### Validate data
            tmp_states = []
            tmp_actions = []
            tmp_result_states = [] 
            tmp_rewards = []
            tmp_falls = []
            tmp_advantage = []
            tmp_exp_action = []
            # print ("Advantage:", _advantage)
            # print ("rewards:", _rewards)
            # print("Batch size: ", len(_states), len(_actions), len(_result_states), len(_rewards), len(_falls), len(_advantage))
            num_samples_=0
            for (state__, action__, next_state__, reward__, fall__, advantage__) in zip(_states, _actions, _result_states, _rewards, _falls, _advantage):
                if (checkValidData(state__, action__, next_state__, reward__)):
                    tmp_states.append(state__)
                    tmp_actions.append(action__)
                    tmp_result_states.append(next_state__)
                    tmp_rewards.append(reward__)
                    tmp_falls.append(fall__)
                    tmp_advantage.append(advantage__)
                    tmp_exp_action.append([0])## Doesn't really matter for on policy methods
                    tup = (state__, action__, next_state__, reward__, fall__, advantage__, [0])
                    self._expBuff.insertTuple(tup)
                    num_samples_ = num_samples_ + 1
                # else:
                    # print ("Tuple invalid:")
            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):        
                print ("self._expBuff.samples(): ", self._expBuff.samples())
            _states = np.array(norm_action(np.array(tmp_states), self._state_bounds), dtype=self._settings['float_type'])
            if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
                _actions = np.array((np.array(tmp_actions)), dtype=self._settings['float_type'])
            else:
                _actions = np.array(norm_action(np.array(tmp_actions), self._action_bounds), dtype=self._settings['float_type'])
            _result_states = np.array(norm_action(np.array(tmp_result_states), self._state_bounds), dtype=self._settings['float_type'])
            _rewards = np.array(tmp_rewards, dtype=self._settings['float_type'])
            _falls = np.array(tmp_falls, dtype='int8')
            _advantage = np.array(tmp_advantage, dtype=self._settings['float_type'])
            # print("Not Falls: ", _falls)
            # print("Rewards: ", _rewards)
            # print ("Actions after: ", _actions)
            cost = 0
            if (self._settings['train_critic']):
                if (self._settings['critic_updates_per_actor_update'] > 1):
                    for i in range(self._settings['critic_updates_per_actor_update']):
                        # print ("Number of samples:", self._expBuff.samples())
                        states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__ = self._expBuff.get_batch(min(self._settings["batch_size"], self._expBuff.samples()))
                        loss = self._pol.trainCritic(states=states__, actions=actions__, rewards=rewards__, result_states=result_states__, falls=falls__)
                        # cost = self._pol.trainCritic(states=_states, actions=_actions, rewards=_rewards, result_states=_result_states, falls=_falls)
                        if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                            print("Value function loss: ", loss)
                        if not np.isfinite(cost) or (cost > 500) :
                            numpy.set_printoptions(threshold=numpy.nan)
                            print ("States: " + str(states__) + " ResultsStates: " + str(result_states__) + " Rewards: " + str(rewards__) + " Actions: " + str(actions__))
                            print ("Training cost is Odd: ", cost)
                else:
                    # print ("Number of samples:", self._expBuff.samples())
                    states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__ = self._expBuff.get_batch(self._settings["batch_size"])
                    # cost = self._pol.trainCritic(states=states__, actions=actions__, rewards=rewards__, result_states=result_states__, falls=falls__)
                    cost = self._pol.trainCritic(states=_states, actions=_actions, rewards=_rewards, result_states=_result_states, falls=_falls)
                    if not np.isfinite(cost) or (cost > 500) :
                        numpy.set_printoptions(threshold=numpy.nan)
                        print ("States: " + str(_states) + " ResultsStates: " + str(_result_states) + " Rewards: " + str(_rewards) + " Actions: " + str(_actions))
                        print ("Training cost is Odd: ", cost)
                # self._expBuff.clear()
            if (self._settings['train_actor']):
                if ( 'use_multiple_policy_updates' in self._settings and ( self._settings['use_multiple_policy_updates']) ):
                    for i in range(self._settings['critic_updates_per_actor_update']):
                    
                        _states, _actions, _result_states, _rewards, _falls, _advantage, exp_actions__ = self._expBuff.get_batch(self._settings["batch_size"])
                        # states__, actions__, result_states__, rewards__, falls__, G_ts__ = self._expBuff.get_batch(self._settings["batch_size"])
                        cost_ = self._pol.trainActor(states=_states, actions=_actions, rewards=_rewards, result_states=_result_states, falls=_falls, advantage=_advantage)
                else:
                    cost_ = self._pol.trainActor(states=_states, actions=_actions, rewards=_rewards, result_states=_result_states, falls=_falls, advantage=_advantage)
                    """
                    if not np.isfinite(cost_) or (cost_ > 500) :
                        numpy.set_printoptions(threshold=numpy.nan)
                        print ("States: " + str(_states) + " ResultsStates: " + str(_result_states) + " Rewards: " + str(_rewards) + " Actions: " + str(_actions))
                        print ("Training cost is Odd: ", cost_)
                    """
            dynamicsLoss = 0 
            if (self._settings['train_forward_dynamics']):
                for i in range(self._settings['critic_updates_per_actor_update']):
                    states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__ = self._expBuff.get_batch(self._settings["batch_size"])
                    dynamicsLoss = self._fd.train(states=states__, actions=actions__, result_states=result_states__, rewards=rewards__)
                    """
                    if not np.isfinite(dynamicsLoss) or (dynamicsLoss > 500) :
                        numpy.set_printoptions(threshold=numpy.nan)
                        print ("States: " + str(states__) + " ResultsStates: " + str(result_states__) + " Rewards: " + str(rewards__) + " Actions: " + str(actions__))
                        print ("Training cost is Odd: ", dynamicsLoss)
                    """
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print ("Forward Dynamics Loss: ", dynamicsLoss)
                    if (self._settings['train_critic_on_fd_output'] and 
                        (( self._pol.numUpdates() % self._settings['dyna_update_lag_steps']) == 0) and 
                        ( ( self._pol.numUpdates() %  self._settings['steps_until_target_network_update']) >= (self._settings['steps_until_target_network_update']/10)) and
                        ( ( self._pol.numUpdates() %  self._settings['steps_until_target_network_update']) <= (self._settings['steps_until_target_network_update'] - (self._settings['steps_until_target_network_update']/10)))
                        ):
                        predicted_result_states__ = self._fd.predict_batch(states=states__, actions=actions__)
                        cost = self._pol.trainDyna(predicted_states=predicted_result_states__, actions=actions__, rewards=rewards__, result_states=result_states__, falls=falls__)
                        """
                        if not np.isfinite(cost) or (cost > 500) :
                            numpy.set_printoptions(threshold=numpy.nan)
                            print ("States: " + str(states__) + " ResultsStates: " + str(result_states__) + " Rewards: " + str(rewards__) + " Actions: " + str(actions__))
                            print ("Training cost is Odd: ", cost)
                        """
                        if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                            print("Performing Dyna Update, loss: ", cost)
                        # print("Updated params: ", self._pol.getNetworkParameters()[0][0][0])
                if ( 'use_MBPG' in self._settings and (self._settings['use_MBPG'])):
                    for i in range(self._settings['critic_updates_per_actor_update']):
                        # if ( 'use_multiple_policy_updates' in self._settings and ( self._settings['use_multiple_policy_updates']) ):
                        _states, _actions, _result_states, _rewards, _falls, _advantage, exp_actions__ = self._expBuff.get_batch(self._settings["batch_size"])
                            # states__, actions__, result_states__, rewards__, falls__, G_ts__ = self._expBuff.get_batch(self._settings["batch_size"])
                        if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                            print ("Training MBPG")
                        cost_ = self._pol.trainActionGrad(states=_states, forwardDynamicsModel=self._fd)
                        if not np.isfinite(cost_) or (cost_ > 500) :
                            numpy.set_printoptions(threshold=numpy.nan)
                            print ("States: " + str(_states) + " ResultsStates: " + str(_result_states) + " Rewards: " + str(_rewards) + " Actions: " + str(_actions))
                            print ("Training cost is Odd: ", cost_)
        else: ## Off-policy
            # print("State Bounds LA:", self._pol.getStateBounds())
            # print("Action Bounds LA:", self._pol.getActionBounds())
            # print("Exp State Bounds LA: ", self._expBuff.getStateBounds())
            # print("Exp Action Bounds LA: ", self._expBuff.getActionBounds())
            
            for update in range(self._settings['training_updates_per_sim_action']): ## Even more training options...
                for i in range(self._settings['critic_updates_per_actor_update']):
                    _states, _actions, _result_states, _rewards, _falls, _G_ts, _exp_actions = self._expBuff.get_batch(self._settings["batch_size"])
                    # print ("Updating Critic")
                    loss = self._pol.trainCritic(states=_states, actions=_actions, rewards=_rewards, result_states=_result_states, falls=_falls)
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print("Value function loss: ", loss)
                    if not np.isfinite(cost) or (cost > 500) :
                        numpy.set_printoptions(threshold=numpy.nan)
                        print ("States: " + str(_states) + " ResultsStates: " + str(_result_states) + " Rewards: " + str(_rewards) + " Actions: " + str(_actions))
                        print ("Training cost is Odd: ", cost)
                if (self._settings['train_actor']):
                    cost_ = self._pol.trainActor(states=_states, actions=_actions, rewards=_rewards, result_states=_result_states, falls=_falls, advantage=_advantage, exp_actions=_exp_actions)
                dynamicsLoss = 0 
                if (self._settings['train_forward_dynamics']):
                    dynamicsLoss = self._fd.train(states=_states, actions=_actions, result_states=_result_states, rewards=_rewards)
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print ("Forward Dynamics Loss: ", dynamicsLoss)
                    if (self._settings['train_critic_on_fd_output'] and 
                        (( self._pol.numUpdates() % self._settings['dyna_update_lag_steps']) == 0) and 
                        ( ( self._pol.numUpdates() %  self._settings['steps_until_target_network_update']) >= (self._settings['steps_until_target_network_update']/10)) and
                        ( ( self._pol.numUpdates() %  self._settings['steps_until_target_network_update']) <= (self._settings['steps_until_target_network_update'] - (self._settings['steps_until_target_network_update']/10)))
                        ):
                        
                        predicted_result_states__ = self._fd.predict_batch(states=_states, actions=_actions)
                        cost = self._pol.trainDyna(predicted_states=predicted_result_states__, actions=_actions, rewards=_rewards, result_states=_result_states, falls=_falls)
                        if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                            print ( "Dyna training cost: ", cost)
                        if not np.isfinite(cost) or (cost > 500) :
                            numpy.set_printoptions(threshold=numpy.nan)
                            print ("States: " + str(_states) + " ResultsStates: " + str(_result_states) + " Rewards: " + str(_rewards) + " Actions: " + str(_actions))
                            print ("Training cost is Odd: ", cost)
            # import lasagne
            # val_params = lasagne.layers.helper.get_all_param_values(self._pol.getModel().getCriticNetwork())
            # pol_params = lasagne.layers.helper.get_all_param_values(self._pol.getModel().getActorNetwork())
            # fd_params = lasagne.layers.helper.get_all_param_values(self._fd.getModel().getForwardDynamicsNetwork())
            # print ("Learning Agent: Model pointers: val, ", self._pol.getModel(), " poli, ", self._pol.getModel(),  " fd, ", self._fd.getModel())
            # print("pol first layer params: ", pol_params[1])
            # print("val first layer params: ", val_params[1])
            # print("fd first layer params: ", fd_params[1])               
                            
        if self._useLock:
            self._accesLock.release()
        return (cost, dynamicsLoss) 
    
    def predict(self, state, evaluation_=False):
        if self._useLock:
            self._accesLock.acquire()
        act = self._pol.predict(state)
        if self._useLock:
            self._accesLock.release()
        return act
    
    def predict_std(self, state, evaluation_=False):
        if self._useLock:
            self._accesLock.acquire()
        std = self._pol.predict_std(state)
        if self._useLock:
            self._accesLock.release()
        return std
    
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
        return self.getPolicy().getStateBounds()
    def getActionBounds(self):
        return self.getPolicy().getActionBounds()
    def getRewardBounds(self):
        return self.getPolicy().getRewardBounds()
    
    def setStateBounds(self, bounds):
        self.getPolicy().setStateBounds(bounds)
        if (self._settings['train_forward_dynamics']):
            self.getForwardDynamics().setStateBounds(bounds)
    def setActionBounds(self, bounds):
        self.getPolicy().setActionBounds(bounds)
        if (self._settings['train_forward_dynamics']):
            self.getForwardDynamics().setActionBounds(bounds)
    def setRewardBounds(self, bounds):
        self.getPolicy().setRewardBounds(bounds)
        if (self._settings['train_forward_dynamics']):
            self.getForwardDynamics().setRewardBounds(bounds)

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
        print ('Worker started')
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
            # print ("Learner Size of state input Queue: " + str(self._input_queue.qsize()))
            # self._agent._expBuff = self.__learningNamespace.experience
            if self._agent._settings['action_space_continuous']:
                # self._agent._expBuff.insert(norm_state(state_, self._agent._state_bounds), 
                #                            norm_action(action, self._agent._action_bounds), norm_state(resultState, self._agent._state_bounds), [reward])
                self._agent._expBuff.insertTuple(tmp)
                # print ("Experience buffer size: " + str(self.__learningNamespace.experience.samples()))
                # print ("Reward Scale: ", self._agent._reward_bounds)
                # print ("Reward Scale Model: ", self._agent._pol.getRewardBounds())
            else:
                self._agent._expBuff.insertTuple(tmp)
            # print ("Learning agent experience size: " + str(self._agent._expBuff.samples()))
            step_ += 1
            if self._agent._expBuff.samples() > self._agent._settings["batch_size"] and ((step_ >= self._agent._settings['sim_action_per_training_update']) ):
                __states, __actions, __result_states, __rewards, __falls, __G_ts, __exp_actions = self._agent._expBuff.get_batch(self._agent._settings["batch_size"])
                # print ("States: " + str(__states) + " ResultsStates: " + str(__result_states) + " Rewards: " + str(__rewards) + " Actions: " + str(__actions))
                (cost, dynamicsLoss) = self._agent.train(_states=__states, _actions=__actions, _rewards=__rewards, _result_states=__result_states, _falls=__falls)
                # print ("Master Agent Running training step, cost: " + str(cost) + " PID " + str(os.getpid()))
                # print ("Updated parameters: " + str(self._agent._pol.getNetworkParameters()[3]))
                if not np.isfinite(cost):
                    print ("States: " + str(__states) + " ResultsStates: " + str(__result_states) + " Rewards: " + str(__rewards) + " Actions: " + str(__actions))
                    print ("Training cost is Nan: ", cost)
                    sys.exit()
                # if (step_ % 10) == 0: # to help speed things up
                # self._learningNamespace.agentPoly = self._agent.getPolicy().getNetworkParameters()
                data = (self._agent._expBuff, self._agent.getPolicy().getNetworkParameters())
                if (self._agent._settings['train_forward_dynamics']):
                    # self._learningNamespace.forwardNN = self._agent.getForwardDynamics().getNetworkParameters()
                    data = (self._agent._expBuff, self._agent.getPolicy().getNetworkParameters(), self._agent.getForwardDynamics().getNetworkParameters())
                # self._learningNamespace.experience = self._agent._expBuff
                ## put and do not block
                try:
                    # print ("Sending network params:")
                    if (not (self._output_message_queue.full())):
                        self._output_message_queue.put(data, False)
                    else:
                        ## Pull out an old one
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
        