import copy
import sys
# from pygments.lexers.theorem import LeanLexer
sys.setrecursionlimit(50000)
import os
import json
sys.path.append("../")
sys.path.append("../characterSimAdapter/")
import math
import numpy as np

import random
# import cPickle
import dill
import dill as pickle
import dill as cPickle

import cProfile, pstats, io
# import memory_profiler
# import psutil
import gc
# from guppy import hpy; h=hpy()
# from memprof import memprof

# import pathos.multiprocessing
import multiprocessing


sim_processes = []
learning_processes = []
_input_anchor_queue = None
_output_experience_queue = None
_eval_episode_data_queue = None
_sim_work_queues = []

# python -m memory_profiler example.py
# @profile(precision=5)
def trainModelParallel(settingsFileName):
        
    # pr = cProfile.Profile()
    # pr.enable()
    # try:
        file = open(settingsFileName)
        settings = json.load(file)
        print ("Settings: " + str(json.dumps(settings)))
        file.close()
        import os    
        os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device="+settings['training_processor_type']+",floatX="+settings['float_type']
        
        ## Theano needs to be imported after the flags are set.
        # from ModelEvaluation import *
        # from model.ModelUtil import *
        # print ( "theano.config.mode: ", theano.config.mode)
        from ModelEvaluation import SimWorker, evalModelParrallel, collectExperience, simEpoch, evalModel, simModelParrallel
        from model.ModelUtil import validBounds
        from model.LearningAgent import LearningAgent, LearningWorker
        from util.SimulationUtil import validateSettings
        from util.SimulationUtil import createEnvironment
        from util.SimulationUtil import createRLAgent
        from util.SimulationUtil import createActor
        from util.SimulationUtil import getDataDirectory, createForwardDynamicsModel, createSampler
        
        
        from util.ExperienceMemory import ExperienceMemory
        from RLVisualize import RLVisualize
        from NNVisualize import NNVisualize
        
        from sim.PendulumEnvState import PendulumEnvState
        from sim.PendulumEnv import PendulumEnv
        from sim.BallGame2DEnv import BallGame2DEnv
        settings = validateSettings(settings)
        
        model_type= settings["model_type"]
        directory= getDataDirectory(settings)
        rounds = settings["rounds"]
        epochs = settings["epochs"]
        # num_states=settings["num_states"]
        epsilon = settings["epsilon"]
        discount_factor=settings["discount_factor"]
        reward_bounds=np.array(settings["reward_bounds"])
        # reward_bounds = np.array([[-10.1],[0.0]])
        batch_size=settings["batch_size"]
        train_on_validation_set=settings["train_on_validation_set"]
        state_bounds = np.array(settings['state_bounds'])
        discrete_actions = np.array(settings['discrete_actions'])
        num_actions= discrete_actions.shape[0] # number of rows
        print ("Sim config file name: " + str(settings["sim_config_file"]))
        # c = characterSim.Configuration(str(settings["sim_config_file"]))
        # c = characterSim.Configuration("../data/epsilon0Config.ini")
        action_space_continuous=settings['action_space_continuous']

        if (settings['num_available_threads'] == 1):
            input_anchor_queue = multiprocessing.Queue(settings['queue_size_limit'])
            output_experience_queue = multiprocessing.Queue(settings['queue_size_limit'])
            eval_episode_data_queue = multiprocessing.Queue(settings['queue_size_limit'])
        else:
            input_anchor_queue = multiprocessing.Queue(settings['epochs'])
            output_experience_queue = multiprocessing.Queue(settings['queue_size_limit'])
            eval_episode_data_queue = multiprocessing.Queue(settings['eval_epochs'])
            
        if (settings['on_policy']): ## So that off policy agent does not learn
            output_experience_queue = None
            
        sim_work_queues = []
        
        action_space_continuous=settings['action_space_continuous']
        if action_space_continuous:
            action_bounds = np.array(settings["action_bounds"], dtype=float)
            
        ### Using a wrapper for the type of actor now
        actor = createActor(settings['environment_type'], settings, None)
        exp_val = None
        if (not validBounds(action_bounds)):
            # Check that the action bounds are spcified correctly
            print("Action bounds invalid: ", action_bounds)
            sys.exit()
        if (not validBounds(state_bounds)):
            # Probably did not collect enough bootstrapping samples to get good state bounds.
            print("State bounds invalid: ", state_bounds)
            sys.exit()
        if (not validBounds(reward_bounds)):
            print("Reward bounds invalid: ", reward_bounds)
            sys.exit()
        
        if settings['action_space_continuous']:
            experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), settings['expereince_length'], continuous_actions=True, settings=settings)
        else:
            experience = ExperienceMemory(len(state_bounds[0]), 1, settings['expereince_length'])
            
        experience.setSettings(settings)
        
        if settings['visualize_learning']:    
            rlv = RLVisualize(directory+str(settings['agent_name']), settings)
            rlv.setInteractive()
            rlv.init()
        if (settings['train_forward_dynamics']):
            if settings['visualize_learning']:
                nlv = NNVisualize(title=str("Forward Dynamics Model") + " with " + str(settings["model_type"]), settings=settings)
                nlv.setInteractive()
                nlv.init()
        if (settings['train_reward_predictor']):
            if settings['visualize_learning']:
                rewardlv = NNVisualize(title=str("Reward Model") + " with " + str(settings["model_type"]), settings=settings)
                rewardlv.setInteractive()
                rewardlv.init()
                 
        if (settings['debug_critic']):
            criticLosses = []
            criticRegularizationCosts = [] 
            if (settings['visualize_learning']):
                critic_loss_viz = NNVisualize(title=str("Critic Loss") + " with " + str(settings["model_type"]))
                critic_loss_viz.setInteractive()
                critic_loss_viz.init()
                critic_regularization_viz = NNVisualize(title=str("Critic Regularization Cost") + " with " + str(settings["model_type"]))
                critic_regularization_viz.setInteractive()
                critic_regularization_viz.init()
            
        if (settings['debug_actor']):
            actorLosses = []
            actorRegularizationCosts = []            
            if (settings['visualize_learning']):
                actor_loss_viz = NNVisualize(title=str("Actor Loss") + " with " + str(settings["model_type"]))
                actor_loss_viz.setInteractive()
                actor_loss_viz.init()
                actor_regularization_viz = NNVisualize(title=str("Actor Regularization Cost") + " with " + str(settings["model_type"]))
                actor_regularization_viz.setInteractive()
                actor_regularization_viz.init()

        # mgr = multiprocessing.Manager()
        # namespace = mgr.Namespace()
        
        model = createRLAgent(settings['agent_name'], state_bounds, discrete_actions, reward_bounds, settings)
        
        if (settings['train_forward_dynamics']):
            if ( settings['forward_dynamics_model_type'] == "SingleNet"):
                print ("Creating forward dynamics network: Using single network model")
                forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, None, None, agentModel=model)
                # forwardDynamicsModel = model
            else:
                print ("Creating forward dynamics network")
                # forwardDynamicsModel = ForwardDynamicsNetwork(state_length=len(state_bounds[0]),action_length=len(action_bounds[0]), state_bounds=state_bounds, action_bounds=action_bounds, settings_=settings)
                forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, None, None, agentModel=None)
            # masterAgent.setForwardDynamics(forwardDynamicsModel)
            forwardDynamicsModel.setActor(actor)
            # forwardDynamicsModel.setEnvironment(exp)
            forwardDynamicsModel.init(len(state_bounds[0]), len(action_bounds[0]), state_bounds, action_bounds, actor, None, settings)
        
        learning_workers = []
        # for process in range(settings['num_available_threads']):
        for process in range(1):
            # this is the process that selects which game to play
            agent = LearningAgent(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                              action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
            
            agent.setSettings(settings)
            """
            if action_space_continuous:
                model = createRLAgent(settings['agent_name'], state_bounds, action_bounds, reward_bounds, settings)
            else:
                model = createRLAgent(settings['agent_name'], state_bounds, discrete_actions, reward_bounds, settings)
            model.setStateBounds(state_bounds)
            model.setActionBounds(action_bounds)
            model.setRewardBounds(reward_bounds)
            """
            # agent.setPolicy(model)
            # actor.setPolicy(model)
            # agent.setExperience(experience)
            # learningNamespace.agentPoly = agent.getPolicy().getNetworkParameters()
            # learningNamespace.experience = experience
            
            lw = LearningWorker(output_experience_queue, agent)
            # lw.start()
            learning_workers.append(lw)  
        masterAgent = agent
        # print ("NameSpace: " + str(namespace))
        # sys.exit(0)
        
        # this is the process that selects which game to play
        sim_workers = []
        for process in range(settings['num_available_threads']):
            # this is the process that selects which game to play
            exp_=None
            
            if (int(settings["num_available_threads"]) == 1): # This is okay if there is one thread only...
                print ("Assigning same EXP")
                exp_ = exp_val # This should not work properly for many simulations running at the same time. It could try and evalModel a simulation while it is still running samples 
            print ("original exp: ", exp_)
                # sys.exit()
        
            agent = LearningAgent(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                              action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
            
            agent.setSettings(settings)
            agent.setPolicy(model)
            if (settings['train_forward_dynamics']):
                agent.setForwardDynamics(forwardDynamicsModel)
            
            elif ( settings['use_simulation_sampling'] ):
                
                sampler = createSampler(settings, exp_)
                ## This should be some kind of copy of the simulator not a network
                forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, actor, exp_)
                sampler.setForwardDynamics(forwardDynamicsModel)
                # sampler.setPolicy(model)
                agent.setSampler(sampler)
                print ("thread together exp: ", sampler._exp)
            
            """
            if action_space_continuous:
                model = createRLAgent(settings['agent_name'], state_bounds, action_bounds, reward_bounds, settings)
            else:
                model = createRLAgent(settings['agent_name'], state_bounds, discrete_actions, reward_bounds, settings)
            """
            """
            model_ = copy.deepcopy(model)
            model_.setStateBounds(state_bounds)
            model_.setActionBounds(action_bounds)
            model_.setRewardBounds(reward_bounds)
            """
            # agent.setPolicy(model_)
            """
            if (settings['train_forward_dynamics']):
                # forwardDynamicsModel = ForwardDynamicsNetwork(state_length=len(state_bounds[0]),action_length=len(action_bounds[0]), state_bounds=state_bounds, action_bounds=action_bounds, settings_=settings)
                # forwardDynamicsModel_ = createForwardDynamicsModel(settings, state_bounds, action_bounds, None, None)
                forwardDynamicsModel_ = copy.deepcopy(forwardDynamicsModel)
                agent.setForwardDynamics(forwardDynamicsModel_)
                forwardDynamicsModel_.setActor(actor)
                # forwardDynamicsModel.setEnvironment(exp_)
                forwardDynamicsModel_.init(len(state_bounds[0]), len(action_bounds[0]), state_bounds, action_bounds, actor, exp_, settings)
            """
            if (settings['on_policy']):
                message_queue = multiprocessing.Queue(1)
            else:
                message_queue = multiprocessing.Queue(settings['epochs'])
            sim_work_queues.append(message_queue)
            w = SimWorker(input_anchor_queue, output_experience_queue, actor, exp_, agent, discount_factor, action_space_continuous=action_space_continuous, 
                    settings=settings, print_data=False, p=0.0, validation=True, eval_episode_data_queue=eval_episode_data_queue, process_random_seed=settings['random_seed']+process,
                    message_que=message_queue )
            # w.start()
            sim_workers.append(w)
        
        
        # paramSampler = exp_val.getActor().getParamSampler()
        best_eval=-100000000.0
        best_dynamicsLosses= best_eval*-1.0
            
        values = []
        discounted_values = []
        bellman_error = []
        reward_over_epoc = []
        dynamicsLosses = []
        dynamicsRewardLosses = []
        
        for lw in learning_workers:
            print ("Learning worker" )
            print (lw)
        
        if (int(settings["num_available_threads"]) > 1):
            for sw in sim_workers:
                print ("Sim worker")
                print (sw)
                sw.start()
        
        if not os.path.exists(directory):
            os.makedirs(directory)
            
            
        # copy settings file
        file = open(settingsFileName, 'r')
        out_file_name=directory+os.path.basename(settingsFileName)
        print ("Saving settings file with data: ", out_file_name)
        out_file = open(out_file_name, 'w')
        out_file.write(file.read())
        file.close()
        out_file.close()
        ## This needs to be done after the simulation worker processes are created
        exp_val = createEnvironment(str(settings["forwardDynamics_config_file"]), settings['environment_type'], settings, render=settings['shouldRender'])
        exp_val.setActor(actor)
        exp_val.getActor().init()
        exp_val.init()
        
        experience, state_bounds, reward_bounds, action_bounds = collectExperience(actor, exp_val, model, settings)
        masterAgent.setExperience(experience)
        
        if (not validBounds(action_bounds)):
            # Check that the action bounds are spcified correctly
            print("Action bounds invalid: ", action_bounds)
            sys.exit()
        if (not validBounds(state_bounds)):
            # Probably did not collect enough bootstrapping samples to get good state bounds.
            print("State bounds invalid: ", state_bounds)
            sys.exit()
        if (not validBounds(reward_bounds)):
            print("Reward bounds invalid: ", reward_bounds)
            sys.exit()
        
        if (int(settings["num_available_threads"]) == 1): # This is okay if there is one thread only...
            sim_workers[0].setEnvironment(exp_val)
            sim_workers[0].start()
        
        print ("Reward History: ", experience._reward_history)
        print ("Action History: ", experience._action_history)
        print ("Action Mean: ", np.mean(experience._action_history))
        
        if (settings["save_experience_memory"]):
            print ("Saving initial experience memory")
            file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+"expBufferInit.hdf5"
            experience.saveToFile(file_name)
        """
        if action_space_continuous:
            model = createRLAgent(settings['agent_name'], state_bounds, action_bounds, reward_bounds, settings)
        else:
            model = createRLAgent(settings['agent_name'], state_bounds, discrete_actions, reward_bounds, settings)
        """
        if ( not settings['load_saved_model'] ):
            model.setStateBounds(state_bounds)
            model.setActionBounds(action_bounds)
            model.setRewardBounds(reward_bounds)
        else: # continuation learning
            experience.setStateBounds(copy.deepcopy(model.getStateBounds()))
            experience.setRewardBounds(copy.deepcopy(model.getRewardBounds()))
            experience.setActionBounds(copy.deepcopy(model.getActionBounds()))
            model.setSettings(settings)
            
        # mgr = multiprocessing.Manager()
        # learningNamespace = mgr.Namespace()
        
        masterAgent_message_queue = multiprocessing.Queue(settings['epochs'])
        
        if (settings['train_forward_dynamics']):
            if ( not settings['load_saved_model'] ):
                forwardDynamicsModel.setStateBounds(state_bounds)
                forwardDynamicsModel.setActionBounds(action_bounds)
                forwardDynamicsModel.setRewardBounds(reward_bounds)
            masterAgent.setForwardDynamics(forwardDynamicsModel)
        
        ## Now everything related to the exp memory needs to be updated
        bellman_errors=[]
        masterAgent.setPolicy(model)
        # masterAgent.setForwardDynamics(forwardDynamicsModel)
        # learningNamespace.agentPoly = masterAgent.getPolicy().getNetworkParameters()
        # learningNamespace.model = model
        print("Master agent state bounds: ",  masterAgent.getPolicy().getStateBounds())
        # sys.exit()
        for sw in sim_workers: # Need to update parameter bounds for models
            # sw._model.setPolicy(copy.deepcopy(model))
            # sw.updateModel()
            # sw.updateForwardDynamicsModel()
            print ("exp: ", sw._exp)
            print ("sw modle: ", sw._model.getPolicy()) 
            
            
        # learningNamespace.experience = experience
        ## If not on policy
        if ( not settings['on_policy']):
            for lw in learning_workers:
                # lw._agent.setPolicy(copy.deepcopy(model))
                lw._agent.setPolicy(model)
                # lw.setLearningNamespace(learningNamespace)
                lw.setMasterAgentMessageQueue(masterAgent_message_queue)
                lw.updateExperience(experience)
                # lw.updateModel()
                print ("ls policy: ", lw._agent.getPolicy())
                
                lw.start()
            
        # del learningNamespace.model
        tmp_p=1.0
        if ( settings['load_saved_model'] ):
            tmp_p = settings['min_epsilon']
        data = ('Update_Policy', tmp_p, model.getStateBounds(), model.getActionBounds(), model.getRewardBounds(), 
                masterAgent.getPolicy().getNetworkParameters())
        if (settings['train_forward_dynamics']):
            # masterAgent.getForwardDynamics().setNetworkParameters(learningNamespace.forwardNN)
            data = ('Update_Policy', tmp_p, model.getStateBounds(), model.getActionBounds(), model.getRewardBounds(), 
                    masterAgent.getPolicy().getNetworkParameters(), masterAgent.getForwardDynamics().getNetworkParameters())
        for m_q in sim_work_queues:
            print("trainModel: Sending current network parameters: ", m_q)
            m_q.put(data)
            
        del model
        ## Give gloabl access to processes to they can be terminated when ctrl+c is pressed
        global sim_processes
        sim_processes = sim_workers
        global learning_processes
        learning_processes = learning_workers
        global _input_anchor_queue
        _input_anchor_queue = input_anchor_queue
        global _output_experience_queue
        _output_experience_queue = output_experience_queue
        global _eval_episode_data_queue
        _eval_episode_data_queue = eval_episode_data_queue
        global _sim_work_queues
        _sim_work_queues = sim_work_queues
            
        trainData = {}
        trainData["mean_reward"]=[]
        trainData["std_reward"]=[]
        trainData["mean_bellman_error"]=[]
        trainData["std_bellman_error"]=[]
        trainData["mean_discount_error"]=[]
        trainData["std_discount_error"]=[]
        trainData["mean_forward_dynamics_loss"]=[]
        trainData["std_forward_dynamics_loss"]=[]
        trainData["mean_forward_dynamics_reward_loss"]=[]
        trainData["std_forward_dynamics_reward_loss"]=[]
        trainData["mean_eval"]=[]
        trainData["std_eval"]=[]
        trainData["mean_critic_loss"]=[]
        trainData["std_critic_loss"]=[]
        trainData["mean_critic_regularization_cost"]=[]
        trainData["std_critic_regularization_cost"]=[]
        trainData["mean_actor_loss"]=[]
        trainData["std_actor_loss"]=[]
        trainData["mean_actor_regularization_cost"]=[]
        trainData["std_actor_regularization_cost"]=[]
        
        if (False ):
            print("State Bounds:", masterAgent.getStateBounds())
            print("Action Bounds:", masterAgent.getActionBounds())
            
            print("Exp State Bounds: ", experience.getStateBounds())
            print("Exp Action Bounds: ", experience.getActionBounds())
        
        print ("Starting first round")
        if (settings['on_policy']):
            sim_epochs_ = epochs
            epochs = 1
        for round_ in range(2,rounds+2):
            # p = math.fabs(settings['initial_temperature'] / (math.log(round_*round_) - round_) )
            # p = (settings['initial_temperature'] / (math.log(round_))) 
            # p = ((settings['initial_temperature']/math.log(round_))/math.log(rounds))
            p = ((settings['initial_temperature']/math.log(round_))) 
            # p = ((rounds - round_)/rounds) ** 2
            p = max(settings['min_epsilon'], min(settings['epsilon'], p)) # Keeps it between 1.0 and 0.2
            if ( settings['load_saved_model'] ):
                p = settings['min_epsilon']
                
            # print ("Model pointers: val, ", masterAgent._pol.getModel(), 
            #        " poli, ", masterAgent._pol.getModel(),  " fd, ", masterAgent._fd.getModel())
            
            # for sm in sim_workers:
                # sm.setP(p)
            # pr = cProfile.Profile()
            for epoch in range(epochs):
                if (settings['on_policy']):
                    
                    out = simModelParrallel( sw_message_queues=sim_work_queues,
                                                               model=masterAgent, settings=settings, eval_episode_data_queue=eval_episode_data_queue, anchors=sim_epochs_)
                    
                    # out = simEpoch(actor, exp_val, masterAgent, discount_factor, anchors=epoch, action_space_continuous=action_space_continuous, settings=settings, 
                    #    print_data=False, p=1.0, validation=False, epoch=epoch, evaluation=False, _output_queue=None, epsilon=settings['epsilon'])
                    (tuples, discounted_sum, q_value, evalData) = out
                    (__states, __actions, __result_states, __rewards, __falls, __G_ts, advantage__, exp_actions__) = tuples
                    # print("**** training states: ", np.array(__states).shape)
                    # print("**** training __result_states: ", np.array(__result_states).shape)
                    # print ("Actions before: ", __actions)
                    for i in range(1):
                        masterAgent.train(_states=__states, _actions=__actions, _rewards=__rewards, _result_states=__result_states, _falls=__falls, _advantage=advantage__)
                        
                    data = ('Update_Policy', p, masterAgent.getPolicy().getNetworkParameters())
                    message = {}
                    message['type'] = 'Update_Policy'
                    message['data'] = data
                    if (settings['train_forward_dynamics']):
                        # masterAgent.getForwardDynamics().setNetworkParameters(learningNamespace.forwardNN)
                        data = ('Update_Policy', p, masterAgent.getPolicy().getNetworkParameters(),
                                 masterAgent.getForwardDynamics().getNetworkParameters())
                        message['data'] = data
                    for m_q in sim_work_queues:
                        ## block on full queue
                        m_q.put(message)
                else:
                    episodeData = {}
                    episodeData['data'] = epoch
                    episodeData['type'] = 'sim'
                    input_anchor_queue.put(episodeData)
                
                # pr.enable()
                # print ("Current Tuple: " + str(learningNamespace.experience.current()))
                if masterAgent.getExperience().samples() > batch_size:
                    states, actions, result_states, rewards, falls, G_ts, exp_actions = masterAgent.getExperience().get_batch(batch_size)
                    print ("Batch size: " + str(batch_size))
                    error = masterAgent.bellman_error(states, actions, rewards, result_states, falls)
                    bellman_errors.append(error)
                    if (settings['debug_critic']):
                        loss__ = masterAgent.getPolicy()._get_critic_loss() # uses previous call batch data
                        criticLosses.append(loss__)
                        regularizationCost__ = masterAgent.getPolicy()._get_critic_regularization()
                        criticRegularizationCosts.append(regularizationCost__)
                        
                    if (settings['debug_actor']):
                        """
                        print( "Advantage: ", masterAgent.getPolicy()._get_advantage())
                        print("Policy prob: ", masterAgent.getPolicy()._q_action())
                        print("Policy log prob: ", masterAgent.getPolicy()._get_log_prob())
                        print( "Actor loss: ", masterAgent.getPolicy()._get_action_diff())
                        """
                        loss__ = masterAgent.getPolicy()._get_actor_loss() # uses previous call batch data
                        actorLosses.append(loss__)
                        regularizationCost__ = masterAgent.getPolicy()._get_actor_regularization()
                        actorRegularizationCosts.append(regularizationCost__)
                    
                    if not all(np.isfinite(error)):
                        print ("States: " + str(states) + " ResultsStates: " + str(result_states) + " Rewards: " + str(rewards) + " Actions: " + str(actions) + " Falls: ", str(falls))
                        print ("Bellman Error is Nan: " + str(error) + str(np.isfinite(error)))
                        sys.exit()
                    
                    error = np.mean(np.fabs(error))
                    if error > 10000:
                        print ("Error to big: ")
                        print (states, actions, rewards, result_states)
                        
                    if (settings['train_forward_dynamics']):
                        dynamicsLoss = masterAgent.getForwardDynamics().bellman_error(states, actions, result_states, rewards)
                        dynamicsLoss = np.mean(np.fabs(dynamicsLoss))
                        dynamicsLosses.append(dynamicsLoss)
                    if (settings['train_reward_predictor']):
                        dynamicsRewardLoss = masterAgent.getForwardDynamics().reward_error(states, actions, result_states, rewards)
                        dynamicsRewardLoss = np.mean(np.fabs(dynamicsRewardLoss))
                        dynamicsRewardLosses.append(dynamicsRewardLoss)
                    if (settings['train_forward_dynamics']):
                        print ("Round: " + str(round_) + " Epoch: " + str(epoch) + " p: " + str(p) + " With mean reward: " + str(np.mean(rewards)) + " bellman error: " + str(error) + " ForwardPredictionLoss: " + str(dynamicsLoss))
                    else:
                        print ("Round: " + str(round_) + " Epoch: " + str(epoch) + " p: " + str(p) + " With mean reward: " + str(np.mean(rewards)) + " bellman error: " + str(error))
                    # discounted_values.append(discounted_sum)
                    

                print ("Master agent experience size: " + str(masterAgent.getExperience().samples()))
                # print ("**** Master agent experience size: " + str(learning_workers[0]._agent._expBuff.samples()))
                
                if (not settings['on_policy']):
                    ## There could be stale policy parameters in here, use the last set put in the queue
                    data = None
                    while (not masterAgent_message_queue.empty()):
                        ## Don't block
                        try:
                            data = masterAgent_message_queue.get(False)
                        except Exception as inst:
                            print ("training: In model parameter message queue empty: ", masterAgent_message_queue.qsize())
                    if (not (data == None) ):
                        # print ("Data: ", data)
                        masterAgent.setExperience(data[0])
                        masterAgent.getPolicy().setNetworkParameters(data[1])
                        if (settings['train_forward_dynamics']):
                            masterAgent.getForwardDynamics().setNetworkParameters(data[2])
                
                        
                # experience = learningNamespace.experience
                # actor.setExperience(experience)
                """
                pr.disable()
                f = open('x.prof', 'a')
                pstats.Stats(pr, stream=f).sort_stats('time').print_stats()
                f.close()
                """
            
                # this->_actor->iterate();
            ## This will let me know which part of learning is going slower training updates or simulation
            print ("sim queue size: ", input_anchor_queue.qsize() )
            if ( output_experience_queue != None):
                print ("exp tuple queue size: ", output_experience_queue.qsize())
            
            if (not settings['on_policy']):
                # masterAgent.getPolicy().setNetworkParameters(learningNamespace.agentPoly)
                # masterAgent.setExperience(learningNamespace.experience)
                data = ('Update_Policy', p, masterAgent.getPolicy().getNetworkParameters())
                if (settings['train_forward_dynamics']):
                    # masterAgent.getForwardDynamics().setNetworkParameters(learningNamespace.forwardNN)
                    data = ('Update_Policy', p, masterAgent.getPolicy().getNetworkParameters(),
                             masterAgent.getForwardDynamics().getNetworkParameters())
                for m_q in sim_work_queues:
                    ## Don't block on full queue
                    try:
                        m_q.put(data, False)
                    except: 
                        print ("SimWorker model parameter message queue full: ", m_q.qsize())
              
            if (round_ % settings['plotting_update_freq_num_rounds']) == 0:
                # Running less often helps speed learning up.
                # Sync up sim actors
                
                # if (settings['on_policy'] or ((settings['num_available_threads'] == 1))):
                #     mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error, mean_eval, std_eval = evalModel(actor, exp_val, masterAgent, discount_factor, 
                #                                         anchors=settings['eval_epochs'], action_space_continuous=action_space_continuous, settings=settings)
                # else:
                if (settings['on_policy']):
                    mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error, mean_eval, std_eval = evalModelParrallel( input_anchor_queue=sim_work_queues,
                                                               model=masterAgent, settings=settings, eval_episode_data_queue=eval_episode_data_queue, anchors=settings['eval_epochs'])
                else:
                    mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error, mean_eval, std_eval = evalModelParrallel( input_anchor_queue=input_anchor_queue,
                                                               model=masterAgent, settings=settings, eval_episode_data_queue=eval_episode_data_queue, anchors=settings['eval_epochs'])
                """
                for sm in sim_workers:
                    sm.setP(0.0)
                for lw in learning_workers:
                    output_experience_queue.put(None)
                mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error = evalModelParrallel(input_anchor_queue, output_experience_queue, discount_factor, 
                                                    anchors=_anchors[:settings['eval_epochs']], action_space_continuous=action_space_continuous, settings=settings)
                                                    """
                print (mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error)
                if mean_bellman_error > 10000:
                    print ("Error to big: ")
                else:
                    if (settings['train_forward_dynamics']):
                        mean_dynamicsLosses = np.mean(dynamicsLosses)
                        std_dynamicsLosses = np.std(dynamicsLosses)
                        dynamicsLosses = []
                    if (settings['train_reward_predictor']):
                        mean_dynamicsRewardLosses = np.mean(dynamicsRewardLosses)
                        std_dynamicsRewardLosses = np.std(dynamicsRewardLosses)
                        dynamicsRewardLosses = []
                        
                    trainData["mean_reward"].append(mean_reward)
                    # print ("Mean Rewards: " + str(mean_rewards))
                    trainData["std_reward"].append(std_reward)
                    bellman_errors
                    # trainData["mean_bellman_error"].append(mean_bellman_error)
                    # trainData["std_bellman_error"].append(std_bellman_error)
                    trainData["mean_bellman_error"].append(np.mean(np.fabs(bellman_errors)))
                    trainData["std_bellman_error"].append(np.std(bellman_errors))
                    bellman_errors=[]
                    trainData["mean_discount_error"].append(mean_discount_error)
                    trainData["std_discount_error"].append(std_discount_error)
                    trainData["mean_eval"].append(mean_eval)
                    trainData["std_eval"].append(std_eval)
                    if (settings['train_forward_dynamics']):
                        trainData["mean_forward_dynamics_loss"].append(mean_dynamicsLosses)
                        trainData["std_forward_dynamics_loss"].append(mean_dynamicsLosses)
                    if (settings['train_reward_predictor']):
                        trainData["mean_forward_dynamics_reward_loss"].append(mean_dynamicsRewardLosses)
                        trainData["std_forward_dynamics_reward_loss"].append(mean_dynamicsRewardLosses)
                    if settings['visualize_learning']:
                        rlv.updateBellmanError(np.array(trainData["mean_bellman_error"]), np.array(trainData["std_bellman_error"]))
                        rlv.updateReward(np.array(trainData["mean_eval"]), np.array(trainData["std_eval"]))
                        rlv.updateDiscountError(np.fabs(trainData["mean_discount_error"]), np.array(trainData["std_discount_error"]))
                        rlv.redraw()
                        rlv.setInteractiveOff()
                        rlv.saveVisual(directory+"pendulum_agent_"+str(settings['agent_name']))
                        rlv.setInteractive()
                        # rlv.redraw()
                    if (settings['train_forward_dynamics'] and settings['visualize_learning']):
                        nlv.updateLoss(np.array(trainData["mean_forward_dynamics_loss"]), np.array(trainData["std_forward_dynamics_loss"]))
                        nlv.redraw()
                        nlv.setInteractiveOff()
                        nlv.saveVisual(directory+"trainingGraphNN")
                        nlv.setInteractive()
                    if (settings['train_reward_predictor'] and settings['visualize_learning']):
                        rewardlv.updateLoss(np.array(trainData["mean_forward_dynamics_reward_loss"]), np.array(trainData["std_forward_dynamics_reward_loss"]))
                        rewardlv.redraw()
                        rewardlv.setInteractiveOff()
                        rewardlv.saveVisual(directory+"rewardTrainingGraph")
                        rewardlv.setInteractive()
                    if (settings['debug_critic']):
                        
                        mean_criticLosses = np.mean(criticLosses)
                        std_criticLosses = np.std(criticLosses)
                        trainData["mean_critic_loss"].append(mean_criticLosses)
                        trainData["std_critic_loss"].append(std_criticLosses)
                        criticLosses = []
                        if (settings['visualize_learning']):
                            critic_loss_viz.updateLoss(np.array(trainData["mean_critic_loss"]), np.array(trainData["std_critic_loss"]))
                            critic_loss_viz.redraw()
                            critic_loss_viz.setInteractiveOff()
                            critic_loss_viz.saveVisual(directory+"criticLossGraph")
                            critic_loss_viz.setInteractive()
                        
                        mean_criticRegularizationCosts = np.mean(criticRegularizationCosts)
                        std_criticRegularizationCosts = np.std(criticRegularizationCosts)
                        trainData["mean_critic_regularization_cost"].append(mean_criticRegularizationCosts)
                        trainData["std_critic_regularization_cost"].append(std_criticRegularizationCosts)
                        criticRegularizationCosts = []
                        if (settings['visualize_learning']):
                            critic_regularization_viz.updateLoss(np.array(trainData["mean_critic_regularization_cost"]), np.array(trainData["std_critic_regularization_cost"]))
                            critic_regularization_viz.redraw()
                            critic_regularization_viz.setInteractiveOff()
                            critic_regularization_viz.saveVisual(directory+"criticRegularizationGraph")
                            critic_regularization_viz.setInteractive()
                        
                    if (settings['debug_actor']):
                        
                        mean_actorLosses = np.mean(actorLosses)
                        std_actorLosses = np.std(actorLosses)
                        trainData["mean_actor_loss"].append(mean_actorLosses)
                        trainData["std_actor_loss"].append(std_actorLosses)
                        actorLosses = []
                        if (settings['visualize_learning']):
                            actor_loss_viz.updateLoss(np.array(trainData["mean_actor_loss"]), np.array(trainData["std_actor_loss"]))
                            actor_loss_viz.redraw()
                            actor_loss_viz.setInteractiveOff()
                            actor_loss_viz.saveVisual(directory+"actorLossGraph")
                            actor_loss_viz.setInteractive()
                        
                        mean_actorRegularizationCosts = np.mean(actorRegularizationCosts)
                        std_actorRegularizationCosts = np.std(actorRegularizationCosts)
                        trainData["mean_actor_regularization_cost"].append(mean_actorRegularizationCosts)
                        trainData["std_actor_regularization_cost"].append(std_actorRegularizationCosts)
                        actorRegularizationCosts = []
                        if (settings['visualize_learning']):
                            actor_regularization_viz.updateLoss(np.array(trainData["mean_actor_regularization_cost"]), np.array(trainData["std_actor_regularization_cost"]))
                            actor_regularization_viz.redraw()
                            actor_regularization_viz.setInteractiveOff()
                            actor_regularization_viz.saveVisual(directory+"actorRegularizationGraph")
                            actor_regularization_viz.setInteractive()
                """for lw in learning_workers:
                    lw.start()
                   """     
                ## Visulaize some stuff if you want to
                exp_val.updateViz(actor, masterAgent, directory)
                
                
            if (round_ % settings['saving_update_freq_num_rounds']) == 0:
            
                if (settings['train_forward_dynamics']):
                    file_name_dynamics=directory+"forward_dynamics_"+str(settings['agent_name'])+".pkl"
                    f = open(file_name_dynamics, 'wb')
                    dill.dump(masterAgent.getForwardDynamics(), f)
                    f.close()
                    if mean_dynamicsLosses < best_dynamicsLosses:
                        best_dynamicsLosses = mean_dynamicsLosses
                        print ("Saving BEST current forward dynamics agent: " + str(best_dynamicsLosses))
                        file_name_dynamics=directory+"forward_dynamics_"+str(settings['agent_name'])+"_Best.pkl"
                        f = open(file_name_dynamics, 'wb')
                        dill.dump(masterAgent.getForwardDynamics(), f)
                        f.close()
                        
                if (mean_eval > best_eval):
                    best_eval = mean_eval
                    print ("Saving BEST current agent: " + str(best_eval))
                    file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+"_Best.pkl"
                    f = open(file_name, 'wb')
                    dill.dump(masterAgent.getPolicy(), f)
                    f.close()
                    
                if settings['save_trainData']:
                    fp = open(directory+"trainingData_" + str(settings['agent_name']) + ".json", 'w')
                    # print ("Train data: ", trainData)
                    ## because json does not serialize np.float32 
                    for key in trainData:
                        trainData[key] = [float(i) for i in trainData[key]]
                    json.dump(trainData, fp)
                    fp.close()
                    # draw data
                        
            # mean_reward = std_reward = mean_bellman_error = std_bellman_error = mean_discount_error = std_discount_error = None
            # if ( round_ % 10 ) == 0 :
                print ("Saving current masterAgent")
                
                file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+".pkl"
                f = open(file_name, 'wb')
                dill.dump(masterAgent.getPolicy(), f)
                f.close()
                
                f = open(directory+"trainingData_" + str(settings['agent_name']) + ".json", "w")
                json.dump(trainData, f, sort_keys=True, indent=4)
                f.close()
            gc.collect()    
            # print (h.heap())
            
        # bellman_error = np.fabs(np.array(bellman_error))
        # print ("Mean Bellman error: " + str(np.mean(np.fabs(bellman_error))))
        # print ("STD Bellman error: " + str(np.std(np.fabs(bellman_error))))
        
        # discounted_values = np.array(discounted_values)
        # values = np.array(values)
        
        # print ("Discounted reward difference: " + str(discounted_values - values))
        # print ("Discounted reward difference Avg: " +  str(np.mean(np.fabs(discounted_values - values))))
        # print ("Discounted reward difference STD: " +  str(np.std(np.fabs(discounted_values - values))))
        # reward_over_epoc = np.array(reward_over_epoc)
        
        print ("Terminating Workers"        )
        for sw in sim_workers: # Should update these more often
            input_anchor_queue.put(None)
            
        for sw in sim_workers: # Should update these more often
            sw.join()
            
        if ( output_experience_queue != None):
            for lw in learning_workers: # Should update these more often
                output_experience_queue.put(None)
            
        for lw in learning_workers: # Should update these more often
            lw.join()
        
        exp_val.finish()
        
        file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+".pkl"
        f = open(file_name, 'wb')
        dill.dump(masterAgent.getPolicy(), f)
        f.close()
        
        f = open(directory+"trainingData_" + str(settings['agent_name']) + ".json", "w")
        json.dump(trainData, f, sort_keys=True, indent=4)
        f.close()
        
        if (settings['train_forward_dynamics']):
            file_name_dynamics=directory+"forward_dynamics_"+str(settings['agent_name'])+".pkl"
            f = open(file_name_dynamics, 'wb')
            dill.dump(masterAgent.getForwardDynamics(), f)
            f.close()
        
        """except: # catch *all* exceptions
        e = sys.exc_info()[0]
        print ("Error: " + str(e))
        print ("State " + str(state_) + " action " + str(pa) + " newState " + str(resultState) + " Reward: " + str(reward))
        
        """ 
        
import inspect
def print_full_stack(tb=None):
    """
    Only good way to print stack trace yourself.
    http://blog.dscpl.com.au/2015/03/generating-full-stack-traces-for.html
    """
    if tb is None:
        tb = sys.exc_info()[2]

    print ('Traceback (most recent call last):')
    if (not (tb == None)):
        for item in reversed(inspect.getouterframes(tb.tb_frame)[1:]):
            print (' File "{1}", line {2}, in {3}\n'.format(*item),)
            for line in item[4]:
                print (' ' + line.lstrip(),)
            for item in inspect.getinnerframes(tb):
                print (' File "{1}", line {2}, in {3}\n'.format(*item),)
            for line in item[4]:
                print (' ' + line.lstrip(),)
            
import signal
import sys
def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        # global sim_processes
        # sim_processes = sim_workers
        # global learning_processes
        # learning_processes = learning_workers
        print("sim processes: ", sim_processes)
        print("learning_processes: ", learning_processes)
        
        # cancel_join_thread()
        ## cancel all the queues
        _input_anchor_queue.cancel_join_thread()
        _output_experience_queue.cancel_join_thread()
        _eval_episode_data_queue.cancel_join_thread()
        for sim_queue in _sim_work_queues:
            sim_queue.cancel_join_thread()
        
        
        for proc in sim_processes:
            if (not (proc == None)):
                print ("Killing process: ", proc)
                print ("process id: ", proc.pid())
                os.kill(proc.pid(), signal.SIGINT)
        for proc in learning_processes:
            if (not (proc == None)):
                print ("Killing process: ", proc.pid())
                os.kill(proc.pid(), signal.SIGINT)
            
        print_full_stack()
        sys.exit(0)
# signal.signal(signal.SIGINT, signal_handler)

if (__name__ == "__main__"):
    
    trainModelParallel(sys.argv[1])