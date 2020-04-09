
# import cPickle
import dill
import sys
import gc
# from theano.compile.io import Out
sys.setrecursionlimit(50000)
# from sim.PendulumEnvState import PendulumEnvState
# from sim.PendulumEnv import PendulumEnv
from multiprocessing import Process, Queue
# from pathos.multiprocessing import Pool
import threading
import time
import copy
from model.ModelUtil import *
from util.utils import current_mem_usage
# import memory_profiler
# import resources
from simulation.simEpoch import simEpoch
import traceback
import sys

# class SimWorker(threading.Thread):
class SimWorker(Process):
    
    def __init__(self, input_queue, output_queue, actor, exp, model, discount_factor, action_space_continuous, 
                 settings, print_data, p, validation, eval_episode_data_queue, process_random_seed,
                 message_que, worker_id):
        super(SimWorker, self).__init__()
        self._input_queue= input_queue
        self._output_queue = output_queue
        self._eval_episode_data_queue = eval_episode_data_queue
        self._actor = actor
        self._exp = exp
        self._model = model
        self._discount_factor = discount_factor
        self._action_space_continuous= action_space_continuous
        self._settings= settings
        self._print_data=print_data
        self._p= p
        self._validation=validation
        self._max_iterations = settings['rounds'] + settings['epochs'] * 32
        self._iteration = 0
        # self._namespace = namespace # A way to pass messages between processes
        self._process_random_seed = process_random_seed
        ## Used to receive special messages like update your model parameters to this now!
        self._message_queue = message_que
        self._worker_id = worker_id
        
    def createNewModel(self):
        from util.SimulationUtil import createRLAgent
        print ("Creating new model with different session")
        model = createRLAgent(self._settings['agent_name'], self._settings['state_bounds'], self._settings['action_bounds'], 
                              self._settings['reward_bounds'], self._settings)
        print ("done creating model")
        return model
    
    def createNewFDModel(self, env, setting_):
        from util.SimulationUtil import createNewFDModel
        print ("Creating new FD model with different session")
        forwardDynamicsModel = createNewFDModel(setting_, env, self._model.getPolicy())
        return forwardDynamicsModel
    
    def createSampler(self, poli, fd, exp_, actor):
        from util.SimulationUtil import getDataDirectory, createForwardDynamicsModel, createSampler, createActor
        print("Creating simulation sampler")
        sampler = createSampler(self._settings, exp_)
        ## This should be some kind of copy of the simulator not a network
        if (self._settings['forward_dynamics_predictor'] == "network"):
            forwardDynamicsModel = fd
        else:
            state_bounds = np.array(self._settings['state_bounds'])
            action_bounds = np.array(self._settings['action_bounds']) 
            forwardDynamicsModel = createForwardDynamicsModel(self._settings, state_bounds, action_bounds, actor, exp_, agentModel=None, print_info=True)
        sampler.setForwardDynamics(forwardDynamicsModel)
        sampler.setPolicy(poli)
        return sampler
    
    def setEnvironment(self, exp_):
        """
            Set the environment instance to use
        """
        self._exp = exp_
        self._model.setEnvironment(self._exp)
        
    def updateAgent(self, data):
        
        setLearningData(self._model, self._settings, data)
        
        self._p = data[1]
        
    def run(self):
        try:
            return self._run()
        except:
            raise Exception("".join(traceback.format_exception(*sys.exc_info())))

    # @profile(precision=5)
    def _run(self):
        # from pympler import summary
        # from pympler import muppy
        import os
        from util.SimulationUtil import setupEnvironmentVariable, setupLearningBackend, updateSettings, processBounds
        ### Flag so simulation models can be a little different.
        self._settings["simulation_model"] = True
        ### Keep forward models on the CPU
        setupEnvironmentVariable(self._settings, eval=True)

        import numpy as np
        np.random.seed(self._process_random_seed)
        
        setupLearningBackend(self._settings)
        
        timeout_ = 60 * 10 ### 5 min timeout
        if ("simulation_timeout" in self._settings):
            timeout_ = self._settings["simulation_timeout"]
        
        ## This is not needed if there is one thread only...
        if (int(self._settings["num_available_threads"]) > 0): 
            from util.SimulationUtil import createEnvironment
            print ("************************************Creating simulation environments for simulation workers")
            self._exp = createEnvironment(self._settings["sim_config_file"], self._settings['environment_type'], self._settings, 
                                          render=self._settings['shouldRender'], index=self._worker_id)
            self._exp.setActor(self._actor)
            self._exp.getActor().init()   
            self._exp.init()
            self._exp.setRandomSeed(self._process_random_seed)
            (_, _, self._settings) = processBounds(self._settings['state_bounds'], self._settings['action_bounds'], self._settings, self._exp)
            
            np.random.seed(self._process_random_seed)
            ## The sampler might need this new model if threads > 1
            self._model.setEnvironment(self._exp)
            print("Creating new policy in process:")
            self._model.setPolicy(self.createNewModel())
            if ("train_forward_dynamics" in self._settings and
             (self._settings["train_forward_dynamics"])):
                self._model.setForwardDynamics(self.createNewFDModel(self._exp, self._settings))
            if ("train_reward_distance_metric" in self._settings and
             (self._settings["train_reward_distance_metric"] == True)):
                settings_ = copy.deepcopy(self._settings)
                settings_ = updateSettings(settings_, settings_["reward_metric_settings"])
                self._model.setRewardModel(self.createNewFDModel(self._exp, settings_))
            if ( "use_simulation_sampling" in self._settings
                 and (self._settings['use_simulation_sampling'] )):
                self._model.setSampler(self.createSampler(self._model.getPolicy(),
                                                          self._model.getForwardDynamics(), 
                                                          self._exp, self._actor))
        else:
            print ("sim thread exp: ", self._exp)
        
        
        ## This get is fine, it is the first one that I want to block on.
        print ("Waiting for initial policy update.", self._message_queue)
        episodeData = self._message_queue.get(timeout=timeout_)
        print ("Received initial policy update.")
        if (episodeData == None):
            if ("learning_backend" in self._settings and
            (self._settings["learning_backend"] == "tensorflow")):
                import keras
                sess = keras.backend.get_session()
                keras.backend.clear_session()
                sess.close()
                del sess
            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                print ("Died before starting..")
            self._exp.finish()
            gc.collect()
            return
        message = episodeData['type']
        if message == "Update_Policy":
            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                print ("First Message: ", message)
            data = episodeData['data']
            self.updateAgent(data)
            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                print ("Sim worker:", os.getpid(), " State Bounds: ", self._model.getStateBounds())
            print ("Initial policy ready:")
            # print ("sim worker p: " + str(self._p))
        if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
            print ('Worker: started')
        # do some initialization here
        while True:
            eval=False
            sim_on_poli = False
            bootstrapping = False
            # print ("Worker: getting data")
            if (self._settings['on_policy'] == True):
                episodeData = self._message_queue.get(timeout=timeout_)
                if episodeData == None:
                    print ("Terminating worker: " , os.getpid(), " Size of state input Queue: " + str(self._input_queue.qsize()))
                    break
                elif ( episodeData['type'] == "Update_Policy" ):
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print ("Message: ", message)
                    data = episodeData['data']
                    # print ("Recieves p: ", data[1])
                    # print ("New model parameters: ", data[2][1][0])
                    ### Update scaling parameters
                    self.updateAgent(data)
                    
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print ("Sim worker:", os.getpid(), " Size of state input Queue: " + str(self._input_queue.qsize()))
                        print('\tWorker maximum memory usage: %.2f (mb)' % (current_mem_usage()))
                
                    continue
                
                elif ( episodeData['type'] == "Get_Net_Params" ):
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print ("Message: ", message)
                    data = episodeData['data']
                    # print ("New model parameters: ", data[2][1][0])
                    ### Update scaling parameters
                    data = getLearningData(self._model, self._settings, self._p)
                    self._eval_episode_data_queue.put(data, timeout=timeout_)
                
                    continue
                
                elif episodeData['type'] == "eval":
                    eval=True
                    episodeData = episodeData['data']
                    # "Sim worker evaluating episode"
                elif ( episodeData['type'] == 'sim_on_policy'):
                    sim_on_poli = True
                    episodeData = episodeData['data']
                elif ( episodeData['type'] == 'keep_alive'):
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print ("Keep Sim worker:", os.getpid(), " alive.")
                    self._eval_episode_data_queue.put("returning_keep_alive", timeout=timeout_)
                    continue
                elif ( episodeData['type'] == 'bootstrapping'):
                    bootstrapping = True
                else:
                    episodeData = episodeData['data']
                # print("self._p: ", self._p)
                # print ("Worker: Evaluating episode")
                # print ("Nums samples in worker: ", self._namespace.experience.samples())
                if (eval): ## No action exploration
                    out = self.simEpochParallel(actor=self._actor, exp=self._exp, model=self._model, discount_factor=self._discount_factor, 
                            anchors=episodeData, action_space_continuous=self._action_space_continuous, settings=self._settings, 
                            print_data=self._print_data, p=0.0, validation=True, evaluation=eval)
                elif (sim_on_poli): ### Normal trajectory rollout with noise
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        print("Simulating a normal episode ??with exploration?? on policy")
                    settings_ = copy.deepcopy(self._settings)
                    """
                    r = np.random.rand(1)[0]
                    if ( ('perform_mbae_episode_sampling' in self._settings)
                         and (self._settings['perform_mbae_episode_sampling'] == True)
                        and (r > self._settings['model_based_action_omega']) ): ## regular
                        settings_['model_based_action_omega'] = 0.0
                    elif ( ('perform_mbae_episode_sampling' in self._settings)
                         and (self._settings['perform_mbae_episode_sampling'] == True) 
                         ):
                        ## This will result in an entire episode sampled from MBAE
                        settings_['model_based_action_omega'] = 1.0
                        """
                    out = self.simEpochParallel(actor=self._actor, exp=self._exp, model=self._model, discount_factor=self._discount_factor, 
                            anchors=episodeData, action_space_continuous=self._action_space_continuous, settings=settings_, 
                            print_data=self._print_data, p=self._p, validation=self._validation, evaluation=eval)
                elif (bootstrapping): ### Special bootstrapping case
                    out = self.simEpochParallel(actor=self._actor, exp=self._exp, model=self._model, discount_factor=self._discount_factor, 
                            anchors=episodeData, action_space_continuous=self._action_space_continuous, settings=self._settings, 
                            print_data=self._print_data, p=self._p, validation=self._validation, evaluation=False,
                            bootstrapping=bootstrapping)
                else:
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        print("Simulating a normal episode")
                    settings_ = copy.deepcopy(self._settings)
                    """
                    r = np.random.rand(1)[0]
                    if ( ('perform_mbae_episode_sampling' in self._settings)
                         and (self._settings['perform_mbae_episode_sampling'] == True)
                        and (r > self._settings['model_based_action_omega']) ): ## regular
                        settings_['model_based_action_omega'] = 0.0
                    elif ( ('perform_mbae_episode_sampling' in self._settings)
                         and (self._settings['perform_mbae_episode_sampling'] == True) 
                         ):
                        ## This will result in an entire episode sampled from MBAE
                        settings_['model_based_action_omega'] = 1.0    
                    """
                    out = self.simEpochParallel(actor=self._actor, exp=self._exp, model=self._model, discount_factor=self._discount_factor, 
                            anchors=episodeData, action_space_continuous=self._action_space_continuous, settings=settings_, 
                            print_data=self._print_data, p=self._p, validation=self._validation, evaluation=eval)
                self._iteration += 1
                # if self._p <= 0.0:
                
                #    self._output_queue.put(out, timeout=timeout_)
                (tuples, discounted_sum, q_value, evalData) = out
                # (states, actions, result_states, rewards, falls) = tuples
                tuples[8]["mem_usage_sim"] = np.zeros_like(tuples[8]["agent_id"]) + current_mem_usage()
                out = (tuples, discounted_sum, q_value, evalData)
                ## Hack for now just update after ever episode
                # print ("Worker: send sim results: ")
                if (eval or sim_on_poli or bootstrapping):
                    self._eval_episode_data_queue.put(out, timeout=timeout_)
                else:
                    pass
            elif (self._settings['on_policy'] == "fast"):
                ### This will process trajectories in parallel
                ## Check if any messages in the queue
                episodeData = self._input_queue.get(timeout=timeout_)
                ### Pull updated network parameters, if there are some
                if self._message_queue.qsize() > 0:
                    # print ("Getting updated network parameters:")
                    while (not self._message_queue.empty()):
                        ## Don't block
                        try:
                            data_ = self._message_queue.get(False)
                        except Exception as inst:
                            # print ("SimWorker model parameter message queue empty.")
                            pass
                        if (not (data_ is None)):
                            data = data_
                    # print ("Got updated network parameters:")
                    # print("episodeData: ", episodeData)
                    if (data != None and (isinstance(data,dict))):
                        # message = episodeData[0]## Check if any messages in the queue
                        if data['type'] == "Update_Policy":
                            data = data['data']
                            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                print ("Message: ", episodeData['type'])
                                
                            self.updateAgent(data)
                            """
                            self._model.setStateBounds(data[2])
                            self._model.setActionBounds(data[3])
                            self._model.setRewardBounds(data[4])
                            # if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                # print("Scaling State params: ", self._model.getStateBounds())
                                # print("Scaling Action params: ", self._model.getActionBounds())
                                # print("Scaling Reward params: ", self._model.getRewardBounds())
                            self._model.getPolicy().setNetworkParameters(data[5])
                            if (self._settings['train_forward_dynamics']):
                                self._model.getForwardDynamics().setNetworkParameters(data[6])
                            p = data[1]
                            self._p = p
                            """
                            
                            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                print ("Sim worker:", os.getpid(), " Size of state input Queue: " + str(self._input_queue.qsize()))
                                print('\tWorker maximum memory usage: %.2f (mb)' % (current_mem_usage()))
                # print ("Worker: got data", episodeData)
                if episodeData == None:
                    print ("Terminating worker: " , os.getpid(), " Size of state input Queue: " + str(self._input_queue.qsize()))
                    break
                if episodeData['type'] == "eval":
                    eval=True
                    episodeData = episodeData['data']
                    # "Sim worker evaluating episode"
                elif ( episodeData['type'] == 'sim_on_policy'):
                    sim_on_poli = True
                elif ( episodeData['type'] == 'bootstrapping'):
                    bootstrapping = True
                elif ( episodeData['type'] == "Get_Net_Params" ):
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print ("Message: ", message)
                    data = episodeData['data']
                    print ("Requesting network parameters: ")
                    ### Update scaling parameters
                    data = getLearningData(self._model, self._settings, self._p)
                    self._eval_episode_data_queue.put(data, timeout=timeout_)
                
                    continue
                elif ( episodeData['type'] == 'keep_alive'):
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print ("Keep Sim worker:", os.getpid(), " alive.")
                    self._eval_episode_data_queue.put("returning_keep_alive", timeout=timeout_)
                    continue
                else:
                    episodeData = episodeData['data']
                # print("self._p: ", self._p)
                # print ("Worker: Evaluating episode")
                # print ("Nums samples in worker: ", self._namespace.experience.samples())
                if (eval): ## No action exploration
                    out = self.simEpochParallel(actor=self._actor, exp=self._exp, model=self._model, discount_factor=self._discount_factor, 
                            anchors=episodeData, action_space_continuous=self._action_space_continuous, settings=self._settings, 
                            print_data=self._print_data, p=0.0, validation=True, evaluation=eval)
                elif (sim_on_poli):
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        print("Simulating a normal episode ??with exploration?? on policy")
                    settings_ = copy.deepcopy(self._settings)
                    """
                    r = np.random.rand(1)[0]
                    if ( ('perform_mbae_episode_sampling' in self._settings)
                         and (self._settings['perform_mbae_episode_sampling'] == True)
                        and (r > self._settings['model_based_action_omega']) ): ## regular
                        settings_['model_based_action_omega'] = 0.0
                    elif ( ('perform_mbae_episode_sampling' in self._settings)
                         and (self._settings['perform_mbae_episode_sampling'] == True) 
                         ):
                        ## This will result in an entire episode sampled from MBAE
                        settings_['model_based_action_omega'] = 1.0
                        """
                    out = self.simEpochParallel(actor=self._actor, exp=self._exp, model=self._model, discount_factor=self._discount_factor, 
                            anchors=episodeData, action_space_continuous=self._action_space_continuous, settings=settings_, 
                            print_data=self._print_data, p=self._p, validation=self._validation, evaluation=eval)
                elif (bootstrapping):
                    out = self.simEpochParallel(actor=self._actor, exp=self._exp, model=self._model, discount_factor=self._discount_factor, 
                            anchors=episodeData, action_space_continuous=self._action_space_continuous, settings=self._settings, 
                            print_data=self._print_data, p=self._p, validation=self._validation, evaluation=False,
                            bootstrapping=bootstrapping)
                else:
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        print("Simulating a normal episode")
                    settings_ = copy.deepcopy(self._settings)
                    """
                    r = np.random.rand(1)[0]
                    if ( ('perform_mbae_episode_sampling' in self._settings)
                         and (self._settings['perform_mbae_episode_sampling'] == True)
                        and (r > self._settings['model_based_action_omega']) ): ## regular
                        settings_['model_based_action_omega'] = 0.0
                    elif ( ('perform_mbae_episode_sampling' in self._settings)
                         and (self._settings['perform_mbae_episode_sampling'] == True) 
                         ):
                        ## This will result in an entire episode sampled from MBAE
                        settings_['model_based_action_omega'] = 1.0    
                    """
                    out = self.simEpochParallel(actor=self._actor, exp=self._exp, model=self._model, discount_factor=self._discount_factor, 
                            anchors=episodeData, action_space_continuous=self._action_space_continuous, settings=settings_, 
                            print_data=self._print_data, p=self._p, validation=self._validation, evaluation=eval)
                self._iteration += 1
                
                (tuples, discounted_sum, q_value, evalData) = out
                tuples[8]["mem_usage_sim"] = np.zeros_like(tuples[8]["agent_id"]) + current_mem_usage()
                out = (tuples, discounted_sum, q_value, evalData)
                if (eval or sim_on_poli or bootstrapping):
                    self._eval_episode_data_queue.put(out, timeout=timeout_)
                else:
                    pass
                
            else: ## off policy, all threads sharing the same queue
                episodeData = self._input_queue.get(timeout=timeout_)
                ## Check if any messages in the queue
                # print ("Worker: got data", episodeData)
                if episodeData == None:
                    print ("Terminating worker: " , os.getpid(), " Size of state input Queue: " + str(self._input_queue.qsize()))
                    break
                if episodeData['type'] == "eval":
                    eval=True
                    episodeData = episodeData['data']
                    # "Sim worker evaluating episode"
                elif ( episodeData['type'] == 'sim_on_policy'):
                    sim_on_poli = True
                elif ( episodeData['type'] == 'bootstrapping'):
                    bootstrapping = True
                else:
                    episodeData = episodeData['data']
                # print("self._p: ", self._p)
                # print ("Worker: Evaluating episode")
                # print ("Nums samples in worker: ", self._namespace.experience.samples())
                if (eval): ## No action exploration
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print("Running evaluation episode")
                    out = self.simEpochParallel(actor=self._actor, exp=self._exp, model=self._model, discount_factor=self._discount_factor, 
                            anchors=episodeData, action_space_continuous=self._action_space_continuous, settings=self._settings, 
                            print_data=self._print_data, p=0.0, validation=True, evaluation=eval)
                elif (sim_on_poli): ### With exploration // I don't think this is can EVER be called anymore...
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        print("Simulating a normal episode ??with exploration??")
                    sys.exit()
                    out = self.simEpochParallel(actor=self._actor, exp=self._exp, model=self._model, discount_factor=self._discount_factor, 
                            anchors=episodeData, action_space_continuous=self._action_space_continuous, settings=self._settings, 
                            print_data=self._print_data, p=self._p, validation=self._validation, evaluation=eval)
                elif (bootstrapping): ## With exploration and noise
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        print ("Running boostraping episode")
                    out = self.simEpochParallel(actor=self._actor, exp=self._exp, model=self._model, discount_factor=self._discount_factor, 
                            anchors=episodeData, action_space_continuous=self._action_space_continuous, settings=self._settings, 
                            print_data=self._print_data, p=self._p, validation=self._validation, evaluation=False,
                            bootstrapping=bootstrapping)
                else: ##Normal??
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        print("Simulating a normal episode")
                    settings_ = copy.deepcopy(self._settings)
                    """
                    r = np.random.rand(1)[0]
                    if ( ('perform_mbae_episode_sampling' in self._settings)
                         and (self._settings['perform_mbae_episode_sampling'] == True)
                        and (r > self._settings['model_based_action_omega']) ): ## regular
                        settings_['model_based_action_omega'] = 0.0
                    elif ( ('perform_mbae_episode_sampling' in self._settings)
                         and (self._settings['perform_mbae_episode_sampling'] == True) 
                         ):
                        ## This will result in an entire episode sampled from MBAE
                        settings_['model_based_action_omega'] = 1.0
                        """
                    out = self.simEpochParallel(actor=self._actor, exp=self._exp, model=self._model, discount_factor=self._discount_factor, 
                            anchors=episodeData, action_space_continuous=self._action_space_continuous, settings=settings_, 
                            print_data=self._print_data, p=self._p, validation=self._validation, evaluation=eval)
                self._iteration += 1
                # if self._p <= 0.0:
                
                (tuples, discounted_sum, q_value, evalData) = out
                tuples[8]["mem_usage_sim"] = np.zeros_like(tuples[8]["agent_id"]) + current_mem_usage()
                out = (tuples, discounted_sum, q_value, evalData)
                if (eval or sim_on_poli or bootstrapping):
                    self._eval_episode_data_queue.put(out, timeout=timeout_)
                else:
                    pass
                
                ### Pull updated network parameters
                if self._message_queue.qsize() > 0:
                    data = None
                    # print ("Getting updated network parameters:")
                    while (not self._message_queue.empty()):
                        ## Don't block
                        try:
                            data_ = self._message_queue.get(False)
                        except Exception as inst:
                            # print ("SimWorker model parameter message queue empty.")
                            pass
                        if (not (data_ is None)):
                            episodeData = data_
                    # print ("Got updated network parameters:")
                    # print("episodeData: ", episodeData)
                    if (episodeData != None and (isinstance(episodeData,dict))):
                        # message = episodeData[0]## Check if any messages in the queue
                        message = episodeData['type']
                        if message == "Update_Policy":
                            data = episodeData['data']
                            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                print ("Message: ", message)
                            # print ("New model parameters: ", data[2][1][0])
                            self._model.setStateBounds(data[2])
                            self._model.setActionBounds(data[3])
                            self._model.setRewardBounds(data[4])
                            # if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                # print("Scaling State params: ", self._model.getStateBounds())
                                # print("Scaling Action params: ", self._model.getActionBounds())
                                # print("Scaling Reward params: ", self._model.getRewardBounds())
                            self._model.getPolicy().setNetworkParameters(data[5])
                            if (self._settings['train_forward_dynamics']):
                                self._model.getForwardDynamics().setNetworkParameters(data[6])
                            p = data[1]
                            self._p = p
                            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                print ("Sim worker:", os.getpid(), " Size of state input Queue: " + str(self._input_queue.qsize()))
                                print('\tWorker maximum memory usage: %.2f (mb)' % (current_mem_usage()))
                    
                # print ("Actions: " + str(actions))
                # all_objects = muppy.get_objects()
                # sum1 = summary.summarize(all_objects)
                # summary.print_(sum1)
        print ("Simulation Worker Complete: ", os.getpid())
        if ("learning_backend" in self._settings and
            (self._settings["learning_backend"] == "tensorflow")):
            import keras
            sess = keras.backend.get_session()
            keras.backend.clear_session()
            sess.close()
            del sess
        self._exp.finish()
        gc.collect()
        return
        
    def simEpochParallel(self, actor, exp, model, discount_factor, anchors=None, action_space_continuous=False, settings=None, print_data=False, p=0.0, validation=False, epoch=0, evaluation=False, 
                         bootstrapping=False):
        try:
            out = simEpoch(actor, exp, model, discount_factor, anchors=anchors, action_space_continuous=action_space_continuous, settings=settings,
                           print_data=print_data, p=p, validation=validation, epoch=epoch, evaluation=evaluation, _output_queue=self._output_queue, epsilon=settings['epsilon'],
                           bootstrapping=bootstrapping,
                           worker_id=self._worker_id)
            return out
        except:
            raise Exception("".join(traceback.format_exception(*sys.exc_info())))
