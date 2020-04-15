
"""
    Class to setup and control the environment sampling process
"""
import multiprocessing
from util.SimulationUtil import createActor
from model.LearningMultiAgent import LearningMultiAgent
from simulation.SimWorker import SimWorker

class Sampler(object):
    
    def __init__(self, settings, log):
        self._log = log
        self._settings = settings

        if (settings['num_available_threads'] == -1):
            # No threading
            pass
        else:
            # Threading.
            self._input_anchor_queue = multiprocessing.Queue(settings['num_available_threads'])
            self._input_anchor_queue_eval = multiprocessing.Queue(settings['num_available_threads'])
            self._output_experience_queue = multiprocessing.Queue(settings['num_available_threads'])
            self._eval_episode_data_queue = multiprocessing.Queue(settings['num_available_threads'])

        # Tag_FullObserve_SLAC_mini.json: True            
        if (settings['on_policy']):
            ## So that off-policy agent does not learn
            self._output_experience_queue = None
            
        exp_val = None
        self._timeout_ = 60 * 10 ### 10 min timeout
        # Tag_FullObserve_SLAC_mini.json: True, 1800        
        if ("simulation_timeout" in settings): self._timeout_ = settings["simulation_timeout"]
        
        
        ### These are the workers for training
        (self._sim_workers, self._sim_work_queues) = self.createSimWorkers(self._settings, self._input_anchor_queue, 
                                              self._output_experience_queue, self._eval_episode_data_queue, 
                                              [], [], exp_val)

        self._eval_sim_workers = self._sim_workers
        self._eval_sim_work_queues = self._sim_work_queues
        if ( 'override_sim_env_id' in settings and (settings['override_sim_env_id'] != False)):
            (self._eval_sim_workers, self._eval_sim_work_queues) = self.createSimWorkers(settings, self._input_anchor_queue_eval, 
                                                            self._output_experience_queue, self._eval_episode_data_queue, 
                                                            None, forwardDynamicsModel, exp_val,
                                                            default_sim_id=settings['override_sim_env_id'])
        else:
            self._input_anchor_queue_eval = self._input_anchor_queue
        
        
        if (int(settings["num_available_threads"]) > 0):
            for sw in self._sim_workers:
                print ("Sim worker")
                print (sw)
                sw.start()
            if ( 'override_sim_env_id' in settings and (settings['override_sim_env_id'] != False)):
                for sw in self._eval_sim_workers:
                    print ("Sim worker")
                    print (sw)
                    sw.start()
        
        
        ### This is for a single-threaded Synchronous sim only.
        if (int(settings["num_available_threads"]) == -1): # This is okay if there is one thread only...
            self._sim_workers[0].setEnvironment(exp_val)
            self._sim_workers[0].start()
            if ( 'override_sim_env_id' in settings and (settings['override_sim_env_id'] != False)):
                self._eval_sim_workers[0].setEnvironment(exp_val)
                self._eval_sim_workers[0].start()
        
    # python -m memory_profiler example.py
    # @profile(precision=5)
    def createSimWorkers(self, settings, input_anchor_queue, output_experience_queue, eval_episode_data_queue, model, forwardDynamicsModel, exp_val, default_sim_id=None):
        """
            Creates a number of simulation workers and the message queues that
            are used to tell them what to simulate.
        """    
        
        sim_workers = []
        sim_work_queues = []
        for process in range(abs(settings['num_available_threads'])):
            # this is the process that selects which game to play
            exp_=None
            
            if (int(settings["num_available_threads"]) == -1): # This is okay if there is one thread only...
                print ("Assigning same EXP")
                exp_ = exp_val # This should not work properly for many simulations running at the same time. It could try and evalModel a simulation while it is still running samples 
            print ("original exp: ", exp_)
                # sys.exit()
            ### Using a wrapper for the type of actor now
            actor = createActor(settings['environment_type'], settings, None)
            
            agent = LearningMultiAgent(settings_=settings)
            agent.setSettings(settings)
            agent.setPolicy(model)
            if (settings['train_forward_dynamics']):
                agent.setForwardDynamics(forwardDynamicsModel)
            
            elif ( "use_simulation_sampling" in settings
                   and settings['use_simulation_sampling'] ):
                
                sampler = createSampler(settings, exp_)
                ## This should be some kind of copy of the simulator not a network
                forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, actor, exp_, agentModel=None, print_info=True)
                sampler.setForwardDynamics(forwardDynamicsModel)
                # sampler.setPolicy(model)
                agent.setSampler(sampler)
                print ("thread together exp: ", sampler._exp)
            
            ### Check if this is to be a multi-task simulation
            if type(settings['sim_config_file']) is list:
                if (default_sim_id != None):
                    print("Setting sim_id to default id")
                    sim_id = default_sim_id
                else:
                    print("Setting sim_id to process number")
                    sim_id = process
            else:
                print("Not Multi task simulation")
                sim_id = None
                
            print("Setting sim_id to:" , sim_id)
            if (settings['on_policy']):
                message_queue = multiprocessing.Queue(1)
            else:
                message_queue = multiprocessing.Queue(settings['num_available_threads'])
            sim_work_queues.append(message_queue)
            w = SimWorker(input_anchor_queue, output_experience_queue, actor, exp_, agent, settings["discount_factor"], action_space_continuous=settings['action_space_continuous'], 
                    settings=settings, print_data=False, p=0.0, validation=True, eval_episode_data_queue=eval_episode_data_queue, process_random_seed=settings['random_seed']+process + 1,
                    message_que=message_queue, worker_id=sim_id )
            # w.start()
            sim_workers.append(w)
    
        return (sim_workers, sim_work_queues)
    
    def sendKeepAlive(self):
        
        ### Send keep alive to sim processes
        if (masterAgent.getSettings()['on_policy'] == "fast"):
            out = simModelMoreParrallel( sw_message_queues=self._sim_work_queues
                                       ,model=masterAgent, settings=settings__ 
                                       ,eval_episode_data_queue=self._eval_episode_data_queue 
                                       ,anchors=settings['num_on_policy_rollouts']
                                       ,type='keep_alive'
                                       ,p=1
                                       )
        else:
            out = simModelParrallel( sw_message_queues=self._sim_work_queues,
                                   model=masterAgent, settings=settings__, 
                                   eval_episode_data_queue=self._eval_episode_data_queue, 
                                   anchors=settings__['num_on_policy_rollouts'],
                                   type='keep_alive',
                                   p=1)
            
    def obtainSamples(self, agent, rollouts, p):
        from simulation.simEpoch import simModelParrallel, simModelMoreParrallel, simEpoch
        if (self._settings['on_policy'] == "fast"):
            out = simModelMoreParrallel( sw_message_queues=self._input_anchor_queue,
                                       model=agent, settings=self._settings, 
                                       eval_episode_data_queue=self._eval_episode_data_queue, 
                                       anchors=rollouts
                                       ,p=p)
        else:
            out = simModelParrallel( sw_message_queues=self._sim_work_queues,
                                       model=agent, settings=self._settings, 
                                       eval_episode_data_queue=self._eval_episode_data_queue, 
                                       anchors=rollouts
                                       ,p=p)
            
        return out
        
        
        
    
    def updateParameters(self, agent, p=1.0):
        from model.ModelUtil import getLearningData
        ### Copy the learning agents network parameters to the simulation agents
        message={}
        data = getLearningData(agent, self._settings, tmp_p=p)
        message['type'] = 'Update_Policy'
        message['data'] = data
        for m_q in self._sim_work_queues:
            print("trainModel: Sending current network parameters: ", m_q)
            m_q.put(message, timeout=self._timeout_)
            
        if ( 'override_sim_env_id' in self._settings and (settings['override_sim_env_id'] != False)):
            for m_q in self._eval_sim_work_queues:
                ## block on full queue
                m_q.put(message, timeout=self._timeout_)
                
                
    def finish(self):
        
        print ("Terminating Workers")
        if (self.settings['on_policy'] == True):
            for m_q in self._sim_work_queues:
                ## block on full queue
                m_q.put(None, timeout=self._timeout_)
            if ( 'override_sim_env_id' in self._settings and (self._settings['override_sim_env_id'] != False)):
                for m_q in self._eval_sim_work_queues:
                    ## block on full queue
                    m_q.put(None, timeout=self._timeout_)
            for sw in self._sim_workers: # Should update these more often
                sw.join()
            if ( 'override_sim_env_id' in self._settings and (self._settings['override_sim_env_id'] != False)):
                for sw in self._eval_sim_workers: # Should update these more often
                    sw.join() 
        else:
            for sw in self._sim_workers: 
                self._input_anchor_queue.put(None, timeout=self._timeout_)
            if ( 'override_sim_env_id' in self._settings and (self._settings['override_sim_env_id'] != False)):
                for sw in self._eval_sim_workers: 
                    self._input_anchor_queue_eval.put(None, timeout=self._timeout_)
            print ("Joining Workers"        )
            for sw in self._sim_workers: # Should update these more often
                sw.join()
            if ( 'override_sim_env_id' in self._settings and (self._settings['override_sim_env_id'] != False)):
                for sw in self._eval_sim_workers: # Should update these more often
                    sw.join() 
        
        # input_anchor_queue.close()            
        # input_anchor_queue_eval.close()
        
        if (not self._settings['on_policy']):    
            print ("Terminating learners"        )
            if ( self._output_experience_queue != None):
                for lw in self._learning_workers: # Should update these more often
                    self._output_experience_queue.put(None, timeout=self._timeout_)
                    self._output_experience_queue.put(None, timeout=self._timeout_)
                self._output_experience_queue.close()
            print ("Joining learners"        )  
            """
            for m_q in sim_work_queues:  
                print(masterAgent_message_queue.get(False))
                # print(masterAgent_message_queue.get(False))
            while (not masterAgent_message_queue.empty()):
                ## Don't block
                try:
                    data = masterAgent_message_queue.get(False)
                except Exception as inst:
                    print ("training: In model parameter message queue empty: ", masterAgent_message_queue.qsize())
            """
            for i in range(len(self._learning_workers)): # Should update these more often
                print ("Joining learning worker ", i , " of ", len(self._learning_workers))
                self._learning_workers[i].join()
        
        for i in range(len(self._sim_work_queues)):
            print ("sim_work_queues size: ", self._sim_work_queues[i].qsize())
            while (not self._sim_work_queues[i].empty()): ### Empty the queue
                ## Don't block
                try:
                    data_ = self._sim_work_queues[i].get(False)
                except Exception as inst:
                    # print ("SimWorker model parameter message queue empty.")
                    pass
            # sim_work_queues[i].cancel_join_thread()
            print ("sim_work_queues size: ", self._sim_work_queues[i].qsize())
            
            
        for i in range(len(self._eval_sim_work_queues)):
            print ("eval_sim_work_queues size: ", self._eval_sim_work_queues[i].qsize())
            while (not self._eval_sim_work_queues[i].empty()): ### Empty the queue
                ## Don't block
                try:
                    data_ = self._eval_sim_work_queues[i].get(False)
                except Exception as inst:
                    # print ("SimWorker model parameter message queue empty.")
                    pass
            print ("eval_sim_work_queues size: ", self._eval_sim_work_queues[i].qsize())
        
        
        print ("Finish sim")
        if (int(settings["num_available_threads"]) == -1): # This is okay if there is one thread only...
            self._exp_val.finish()
    
    