
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
# import memory_profiler
# import resources
from simulation.simEpoch import simEpoch


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
    
    def createNewFDModel(self):
        from util.SimulationUtil import getDataDirectory, createForwardDynamicsModel, createSampler, createActor
        print ("Creating new FD model with different session")
        state_bounds = self._settings['state_bounds']
        if ("use_dual_dense_state_representations" in self._settings
            and (self._settings["use_dual_dense_state_representations"] == True)):
            state_bounds = self._settings['state_bounds']
        elif ("use_dual_state_representations" in self._settings
            and (self._settings["use_dual_state_representations"] == True)
            and (not (self._settings["forward_dynamics_model_type"] == "SingleNet"))):
            state_bounds = [[0] * self._settings["fd_num_terrain_features"], 
                                     [1] * self._settings["fd_num_terrain_features"]]
        # if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
        #     print("fd state bounds:", state_bounds)
        action_bounds = self._settings['action_bounds']
        
        forwardDynamicsModel = None
        if (self._settings['train_forward_dynamics']):
            actor = createActor(self._settings['environment_type'], self._settings, None)
            if ( self._settings['forward_dynamics_model_type'] == "SingleNet"
                 and (self._settings['use_single_network'] == True)):
                print ("Creating forward dynamics network: Using single network model")
                forwardDynamicsModel = createForwardDynamicsModel(self._settings, state_bounds, action_bounds, None, None, agentModel=self._model.getPolicy())
                # forwardDynamicsModel = model
            else:
                print ("Creating forward dynamics network")
                # forwardDynamicsModel = ForwardDynamicsNetwork(state_length=len(state_bounds[0]),action_length=len(action_bounds[0]), state_bounds=state_bounds, action_bounds=action_bounds, settings_=settings)
                forwardDynamicsModel = createForwardDynamicsModel(self._settings, state_bounds, action_bounds, None, None, agentModel=None)
            # masterAgent.setForwardDynamics(forwardDynamicsModel)
            forwardDynamicsModel.setActor(actor)
            # forwardDynamicsModel.setEnvironment(exp)
            forwardDynamicsModel.init(len(state_bounds[0]), len(action_bounds[0]), state_bounds, action_bounds, actor, None, self._settings)
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
    
    def current_mem_usage(self):
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
        except ImportError:
            return 0
        # return 0

    def setEnvironment(self, exp_):
        """
            Set the environment instance to use
        """
        self._exp = exp_
        self._model.setEnvironment(self._exp)
        
    # @profile(precision=5)
    def run(self):
        # from pympler import summary
        # from pympler import muppy
        import os
        from util.SimulationUtil import setupEnvironmentVariable, setupLearningBackend
        ### Flag so simulation models can be a little different.
        self._settings["simulation_model"] = True
        ### Keep forward models on the CPU
        setupEnvironmentVariable(self._settings)

        if ("GPU_BUS_Index" in self._settings 
            and ("force_sim_net_to_cpu" in self._settings
                and (self._settings["force_sim_net_to_cpu"] == True))):
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
        
        import numpy as np
        np.random.seed(self._process_random_seed)
        
        setupLearningBackend(self._settings)
        
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
            if (self._settings['state_bounds'] == "ask_env"):
                print ("Getting state bounds from environment")
                s_min = self._exp.getEnvironment().observation_space.getMinimum()
                s_max = self._exp.getEnvironment().observation_space.getMaximum()
                print (self._exp.getEnvironment().observation_space.getMinimum())
                self._settings['state_bounds'] = [s_min,s_max]
                # print ("*************new state bounds: ", np.array(self._settings['state_bounds']).shape)
            np.random.seed(self._process_random_seed)
            ## The sampler might need this new model if threads > 1
            self._model.setEnvironment(self._exp)
            print("Creating new policy in process:")
            self._model.setPolicy(self.createNewModel())
            self._model.setForwardDynamics(self.createNewFDModel())
            if ( self._settings['use_simulation_sampling'] ):
                self._model.setSampler(self.createSampler(self._model.getPolicy(),
                                                          self._model.getForwardDynamics(), 
                                                          self._exp, self._actor))
        else:
            print ("sim thread exp: ", self._exp)
        
        
        ## This get is fine, it is the first one that I want to block on.
        print ("Waiting for initial policy update.", self._message_queue)
        episodeData = self._message_queue.get()
        print ("Received initial policy update.")
        message = episodeData['type']
        if message == "Update_Policy":
            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                print ("First Message: ", message)
            data = episodeData['data']
            """
            poli_params = []
            for i in range(len(data[5])):
                print ("poli params", data[5][i])
                net_params=[]
                for j in range(len(data[5][i])):
                    net_params.append(np.array(data[5][i][j], dtype='float32'))
                poli_params.append(net_params)
                """
            # print("Setting net params")
            self._model.getPolicy().setNetworkParameters(data[5])
            # print ("First Message: ", "Updated policy parameters")
            if (self._settings['train_forward_dynamics']):
                self._model.getForwardDynamics().setNetworkParameters(data[6])
            self._p = data[1]
            self._model.setStateBounds(data[2])
            self._model.setActionBounds(data[3])
            self._model.setRewardBounds(data[4])
            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
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
                episodeData = self._message_queue.get()
                if episodeData == None:
                    print ("Terminating worker: " , os.getpid(), " Size of state input Queue: " + str(self._input_queue.qsize()))
                    break
                elif ( episodeData['type'] == "Update_Policy" ):
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print ("Message: ", message)
                    data = episodeData['data']
                    # print ("New model parameters: ", data[2][1][0])
                    ### Update scaling parameters
                    self._model.setStateBounds(data[2])
                    self._model.setActionBounds(data[3])
                    self._model.setRewardBounds(data[4])
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        print("Scaling State params: ", self._model.getStateBounds())
                        print("Scaling Action params: ", self._model.getActionBounds())
                        print("Scaling Reward params: ", self._model.getRewardBounds())        
                    self._model.getPolicy().setNetworkParameters(data[5])
                    if (self._settings['train_forward_dynamics']):
                        self._model.getForwardDynamics().setNetworkParameters(data[6])
                    p = data[1]
                    # if p < 0.1:
                    #     p = 0.1
                    self._p = p
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print ("Sim worker:", os.getpid(), " Size of state input Queue: " + str(self._input_queue.qsize()))
                        print('\tWorker maximum memory usage: %.2f (mb)' % (self.current_mem_usage()))
                elif episodeData['type'] == "eval":
                    eval=True
                    episodeData = episodeData['data']
                    # "Sim worker evaluating episode"
                elif ( episodeData['type'] == 'sim_on_policy'):
                    sim_on_poli = True
                    episodeData = episodeData['data']
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
                # if self._p <= 0.0:
                
                #    self._output_queue.put(out)
                (tuples, discounted_sum, q_value, evalData) = out
                # (states, actions, result_states, rewards, falls) = tuples
                ## Hack for now just update after ever episode
                # print ("Worker: send sim results: ")
                if (eval or sim_on_poli or bootstrapping):
                    self._eval_episode_data_queue.put(out)
                else:
                    pass
            elif (self._settings['on_policy'] == "fast"):
                ### This will process trajectories in parallel
                episodeData = self._input_queue.get()
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
                # if self._p <= 0.0:
                
                #    self._output_queue.put(out)
                (tuples, discounted_sum, q_value, evalData) = out
                # (states, actions, result_states, rewards, falls) = tuples
                ## Hack for now just update after ever episode
                # print ("Worker: send sim results: ")
                if (eval or sim_on_poli or bootstrapping):
                    # print ("Putting episode data in queue")
                    self._eval_episode_data_queue.put(out)
                else:
                    pass
                
                ### Pull updated network parameters, if there are some
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
                                print('\tWorker maximum memory usage: %.2f (mb)' % (self.current_mem_usage()))
            else: ## off policy, all threads sharing the same queue
                episodeData = self._input_queue.get()
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
                
                #    self._output_queue.put(out)
                (tuples, discounted_sum, q_value, evalData) = out
                # (states, actions, result_states, rewards, falls) = tuples
                ## Hack for now just update after ever episode
                # print ("Worker: send sim results: ")
                if (eval or sim_on_poli or bootstrapping):
                    # print ("Putting episode data in queue")
                    self._eval_episode_data_queue.put(out)
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
                                print('\tWorker maximum memory usage: %.2f (mb)' % (self.current_mem_usage()))
                    
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
        out = simEpoch(actor, exp, model, discount_factor, anchors=anchors, action_space_continuous=action_space_continuous, settings=settings, 
                       print_data=print_data, p=p, validation=validation, epoch=epoch, evaluation=evaluation, _output_queue=self._output_queue, epsilon=settings['epsilon'],
                       bootstrapping=bootstrapping,
                       worker_id=self._worker_id)
        return out
