import copy
import sys
sys.setrecursionlimit(50000)
import os
import json
from numpy import dtype
sys.path.append("../")
sys.path.append("../characterSimAdapter/")
import math
import numpy as np

# import characterSim
from env.BallGame2D import BallGame2D
from ModelEvaluation import *

from model.RLDeepNet import RLDeepNet
from model.DeepRLNet3 import DeepRLNet3  
from model.DeepCACLA import DeepCACLA
from model.DeepCACLADQ import DeepCACLADQ
from model.DeepCACLADV import DeepCACLADV
from model.DeepCACLADropout import DeepCACLADropout
from model.DeepDPG import DeepDPG
from model.DeepDPG2 import DeepDPG2
from model.DeepDPGDQ import DeepDPGDQ
from model.ForwardDynamicsNetwork import ForwardDynamicsNetwork
from model.LearningAgent import *
from util.ExperienceMemory import ExperienceMemory
from RLVisualize import RLVisualize
from NNVisualize import NNVisualize

from actor.ActorInterface import ActorInterface
from actor.BallGame2DActor import BallGame2DActor

from sim.PendulumEnvState import PendulumEnvState
from sim.PendulumEnv import PendulumEnv
from sim.BallGame2DEnv import BallGame2DEnv

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

from model.ModelUtil import *
from util.SimulationUtil import *

# @profile(precision=5)
# @memprof(plot = True)
def train(settingsFileName):
        
    # pr = cProfile.Profile()
    # pr.enable()
    # try:
        file = open(settingsFileName)
        settings = json.load(file)
        print ("Settings: " + str(json.dumps(settings)))
        file.close()
        settings = validateSettings(settings)
        anchor_data_file = open(settings["anchor_file"])
        _anchors = getAnchors(anchor_data_file)
        print ("Length of anchors epochs: " + str(len(_anchors)))
        anchor_data_file.close()
        train_forward_dynamics=True
        model_type= settings["model_type"]
        directory= getDataDirectory(settings)
        num_actions= settings["num_actions"]
        rounds = settings["rounds"]
        epochs = settings["epochs"]
        num_states=settings["num_states"]
        epsilon = settings["epsilon"]
        discount_factor=settings["discount_factor"]
        reward_bounds=np.array(settings["reward_bounds"])
        # reward_bounds = np.array([[-10.1],[0.0]])
        batch_size=settings["batch_size"]
        train_on_validation_set=settings["train_on_validation_set"]
        state_bounds = np.array(settings['state_bounds'])
        discrete_actions = np.array(settings['discrete_actions'])
        print ("Sim config file name: " + str(settings["sim_config_file"]))
        # c = characterSim.Configuration(str(settings["sim_config_file"]))
        # c = characterSim.Configuration("../data/epsilon0Config.ini")
        action_space_continuous=settings['action_space_continuous']
        
        input_anchor_queue = Queue(settings['queue_size_limit'])
        output_experience_queue = Queue(settings['queue_size_limit'])
        
        action_space_continuous=settings['action_space_continuous']
        if action_space_continuous:
            action_bounds = np.array(settings["action_bounds"], dtype=float)
            
        ### Using a wrapper for the type of actor now
        actor = createActor(settings['environment_type'], settings, None)
        exp_val = None
        
        exp_val = createEnvironment(str(settings["forwardDynamics_config_file"]), settings['environment_type'])

        exp_val.getActor().init()
        exp_val.getEnvironment().init()
        
        model = createRLAgent(settings['agent_name'], state_bounds, discrete_actions, reward_bounds, settings)
        experience, state_bounds, reward_bounds, action_bounds = collectExperience(actor, exp_val, model, settings)
        """
        if settings['action_space_continuous']:
            experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), settings['expereince_length'], continuous_actions=True)
        else:
            experience = ExperienceMemory(len(state_bounds[0]), 1, settings['expereince_length'])
        """
        actor.setExperience(experience)
        
        print ("Reward History: ", experience._reward_history)
        print ("Action History: ", experience._action_history)
        print ("Action Mean: ", np.mean(experience._action_history))
            
        if settings['visualize_learning']:    
            rlv = RLVisualize(directory+str(settings['agent_name']))
            rlv.setInteractive()
            rlv.init()
        if (settings['train_forward_dynamics']):
            if settings['visulaize_forward_dynamics']:
                nlv = NNVisualize(title=str("Forward Dynamics Model") + " with " + str(settings["model_type"]))
                nlv.setInteractive()
                nlv.init()
                
        if (settings['debug_critic']):
            critic_loss_viz = NNVisualize(title=str("Critic Loss") + " with " + str(settings["model_type"]))
            critic_loss_viz.setInteractive()
            critic_loss_viz.init()
            criticLosses = []
            critic_regularization_viz = NNVisualize(title=str("Critic Regularization Cost") + " with " + str(settings["model_type"]))
            critic_regularization_viz.setInteractive()
            critic_regularization_viz.init()
            criticRegularizationCosts = []

        mgr = multiprocessing.Manager()
        namespace = mgr.Namespace()
                        
        learning_workers = []
        # for process in range(settings['num_available_threads']):
        for process in range(1):
            # this is the process that selects which game to play
            agent = LearningAgent(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                              action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
            
            agent.setSettings(settings)
            
            if action_space_continuous:
                model = createRLAgent(settings['agent_name'], state_bounds, action_bounds, reward_bounds, settings)
            else:
                model = createRLAgent(settings['agent_name'], state_bounds, discrete_actions, reward_bounds, settings)
            model.setStateBounds(state_bounds)
            model.setActionBounds(action_bounds)
            model.setRewardBounds(reward_bounds)
            agent.setPolicy(model)
            if (settings['train_forward_dynamics']):
                print ("Created forward dynamics network")
                # forwardDynamicsModel = ForwardDynamicsNetwork(state_length=len(state_bounds[0]),action_length=len(action_bounds[0]), state_bounds=state_bounds, action_bounds=action_bounds, settings_=settings)
                forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, None, None)
                agent.setForwardDynamics(forwardDynamicsModel)
                forwardDynamicsModel.setActor(actor)
                # forwardDynamicsModel.setEnvironment(exp)
                forwardDynamicsModel.init(len(state_bounds[0]), len(action_bounds[0]), state_bounds, action_bounds, actor, None, settings)
                namespace.forwardNN = agent.getForwardDynamics().getNetworkParameters()
                actor.setForwardDynamicsModel(forwardDynamicsModel)
            actor.setPolicy(model)
            agent.setExperience(experience)
            namespace.agentPoly = agent.getPolicy().getNetworkParameters()
            namespace.experience = experience
            
            lw = LearningWorker(output_experience_queue, agent, namespace)
            lw.start()
            learning_workers.append(lw)  
        masterAgent = agent
        # print ("NameSpace: " + str(namespace))
        # sys.exit(0)
        
        # this is the process that selects which game to play
        sim_workers = []
        for process in range(settings['num_available_threads']):
            # this is the process that selects which game to play
            exp_=None
            
            if ((settings["num_available_threads"]) == 1): # This is okay if there is one thread only...
                exp_ = exp_val # This should not work properly for many simulations running at the same time. It could try and evalModel a simulation while it is still running samples 
            else:
               
                print ("Starting another worker ", process)
                exp_ = createEnvironment(str(settings["sim_config_file"]), settings['environment_type'])
                exp_.getActor().init()   
                exp_.getEnvironment().init()
                print ("Done starting worker ", process)
            
            agent = LearningAgent(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                              action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
            
            agent.setSettings(settings)
            """
            if action_space_continuous:
                model = createRLAgent(settings['agent_name'], state_bounds, action_bounds, reward_bounds, settings)
            else:
                model = createRLAgent(settings['agent_name'], state_bounds, discrete_actions, reward_bounds, settings)
            """
            model_ = copy.deepcopy(model)
            model_.setStateBounds(state_bounds)
            model_.setActionBounds(action_bounds)
            model_.setRewardBounds(reward_bounds)
            agent.setPolicy(model_)
            if (settings['train_forward_dynamics']):
                print ("Created forward dynamics network")
                # forwardDynamicsModel = ForwardDynamicsNetwork(state_length=len(state_bounds[0]),action_length=len(action_bounds[0]), state_bounds=state_bounds, action_bounds=action_bounds, settings_=settings)
                # forwardDynamicsModel_ = createForwardDynamicsModel(settings, state_bounds, action_bounds, None, None)
                forwardDynamicsModel_ = copy.deepcopy(forwardDynamicsModel)
                agent.setForwardDynamics(forwardDynamicsModel_)
                forwardDynamicsModel_.setActor(actor)
                # forwardDynamicsModel.setEnvironment(exp_)
                forwardDynamicsModel_.init(len(state_bounds[0]), len(action_bounds[0]), state_bounds, action_bounds, actor, exp_, settings)
            
            w = SimWorker(input_anchor_queue, output_experience_queue, actor, exp_, agent, discount_factor, action_space_continuous=action_space_continuous, 
                    settings=settings, print_data=False, p=0.0, validation=True)
            w.start()
            sim_workers.append(w)
        
        
        # paramSampler = exp_val.getActor().getParamSampler()
        best_eval=-100000000.0
        best_dynamicsLosses= best_eval*-1.0
            
        values = []
        discounted_values = []
        bellman_error = []
        reward_over_epoc = []
        dynamicsLosses = []
        
        trainData = {}
        trainData["mean_reward"]=[]
        trainData["std_reward"]=[]
        trainData["mean_bellman_error"]=[]
        trainData["std_bellman_error"]=[]
        trainData["mean_discount_error"]=[]
        trainData["std_discount_error"]=[]
        trainData["mean_forward_dynamics_loss"]=[]
        trainData["std_forward_dynamics_loss"]=[]
        trainData["mean_eval"]=[]
        trainData["std_eval"]=[]
        trainData["mean_critic_loss"]=[]
        trainData["std_critic_loss"]=[]
        trainData["mean_critic_regularization_cost"]=[]
        trainData["std_critic_regularization_cost"]=[]
        
        namespace.experience = experience
        
        for lw in learning_workers:
            print ("Learning worker" )
            print (lw)
            lw.updateExperience()
            
        for sw in sim_workers:
            print ("Sim worker")
            print (sw  )
        
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        if (settings["save_experience_memory"]):
            file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+"expBufferInit.hdf5"
            experience.saveToFile(file_name)
            # experience.loadFromFile(file_name)
            
        # copy settings file
        file = open(settingsFileName, 'r')
        out_file_name=directory+os.path.basename(settingsFileName)
        print ("Saving settings file with data: ", out_file_name)
        out_file = open(out_file_name, 'w')
        out_file.write(file.read())
        file.close()
        out_file.close()
        
        """
        exp_val = createEnvironment(str(settings["forwardDynamics_config_file"]), settings['environment_type'])

        exp_val.getActor().init()
        exp_val.getEnvironment().init()
        """
        bellman_errors=[]
        
        print ("Starting first round")
        for round_ in range(rounds):
            p = (rounds - round_) / float(rounds)
            for sm in sim_workers:
                sm.setP(p)
            # pr = cProfile.Profile()
            for epoch in range(epochs):
                input_anchor_queue.put(_anchors[epoch])
                    
                # pr.enable()
                # print ("Current Tuple: " + str(namespace.experience.current()))
                if experience.samples() > batch_size:
                    states, actions, result_states, rewards, falls = experience.get_batch(batch_size)
                    print ("Batch size: " + str(batch_size))
                    error = masterAgent.bellman_error(states, actions, rewards, result_states, falls)
                    bellman_errors.append(error)
                    if (settings['debug_critic']):
                        loss__ = masterAgent.getPolicy()._get_critic_loss() # uses previous call batch data
                        criticLosses.append(loss__)
                        regularizationCost__ = masterAgent.getPolicy()._get_critic_regularization()
                        criticRegularizationCosts.append(regularizationCost__)
                    
                    if not all(np.isfinite(error)):
                        print ("States: " + str(states) + " ResultsStates: " + str(result_states) + " Rewards: " + str(rewards) + " Actions: " + str(actions))
                        print ("Bellman Error is Nan: " + str(error) + str(np.isfinite(error)))
                        sys.exit()
                    
                    error = np.mean(np.fabs(error))
                    if error > 10000:
                        print ("Error to big: ")
                        print (states, actions, rewards, result_states)
                        
                    if (settings['train_forward_dynamics']):
                        dynamicsLoss = masterAgent.getForwardDynamics().bellman_error(np.array(states), np.array(actions), np.array(result_states))
                        dynamicsLoss = np.mean(np.fabs(dynamicsLoss))
                        dynamicsLosses.append(dynamicsLoss)
                    if (settings['train_forward_dynamics']):
                        print ("Round: " + str(round_) + " Epoch: " + str(epoch) + " With mean reward: " + str(np.mean(rewards)) + " bellman error: " + str(error) + " ForwardPredictionLoss: " + str(dynamicsLoss))
                    else:
                        print ("Round: " + str(round_) + " Epoch: " + str(epoch) + " With mean reward: " + str(np.mean(rewards)) + " bellman error: " + str(error))
                    # discounted_values.append(discounted_sum)

                print ("Master agent experience size: " + str(experience.samples()))
                # print ("**** Master agent experience size: " + str(learning_workers[0]._agent._expBuff.samples()))
                masterAgent.getPolicy().setNetworkParameters(namespace.agentPoly)
                if (settings['train_forward_dynamics']):
                    masterAgent.getForwardDynamics().setNetworkParameters(namespace.forwardNN)
                for sw in sim_workers: # Should update these more offten
                    sw._model.getPolicy().setNetworkParameters(namespace.agentPoly)
                    if (settings['train_forward_dynamics']):
                        sw._model.getForwardDynamics().setNetworkParameters(namespace.forwardNN)
                experience = namespace.experience
                actor.setExperience(experience)
                """
                pr.disable()
                f = open('x.prof', 'a')
                pstats.Stats(pr, stream=f).sort_stats('time').print_stats()
                f.close()
                """
            
                # this->_actor->iterate();
                
            if (round_ % settings['plotting_update_freq_num_rounds']) == 0:
                # Running less often helps speed learning up.
                # Sync up sim actors
                
                 
                mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error, mean_eval, std_eval = evalModel(actor, exp_val, model, discount_factor, 
                                                    anchors=_anchors[:settings['eval_epochs']], action_space_continuous=action_space_continuous, settings=settings)
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
                    if settings['visualize_learning']:
                        rlv.updateBellmanError(np.array(trainData["mean_bellman_error"]), np.array(trainData["std_bellman_error"]))
                        rlv.updateReward(np.array(trainData["mean_eval"]), np.array(trainData["std_eval"]))
                        rlv.updateDiscountError(np.fabs(trainData["mean_discount_error"]), np.array(trainData["std_discount_error"]))
                        rlv.redraw()
                        rlv.setInteractiveOff()
                        rlv.saveVisual(directory+"pendulum_agent_"+str(settings['agent_name']))
                        rlv.setInteractive()
                        # rlv.redraw()
                    if (settings['visulaize_forward_dynamics']):
                        nlv.updateLoss(np.array(trainData["mean_forward_dynamics_loss"]), np.array(trainData["std_forward_dynamics_loss"]))
                        nlv.redraw()
                        nlv.setInteractiveOff()
                        nlv.saveVisual(directory+"trainingGraphNN")
                        nlv.setInteractive()
                    if (settings['debug_critic']):
                        
                        mean_criticLosses = np.mean(criticLosses)
                        std_criticLosses = np.std(criticLosses)
                        trainData["mean_critic_loss"].append(mean_criticLosses)
                        trainData["std_critic_loss"].append(std_criticLosses)
                        criticLosses = []
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
                        critic_regularization_viz.updateLoss(np.array(trainData["mean_critic_regularization_cost"]), np.array(trainData["std_critic_regularization_cost"]))
                        critic_regularization_viz.redraw()
                        critic_regularization_viz.setInteractiveOff()
                        critic_regularization_viz.saveVisual(directory+"criticRegularizationGraph")
                        critic_regularization_viz.setInteractive()
                """for lw in learning_workers:
                    lw.start()
                   """     
            if (round_ % settings['saving_update_freq_num_rounds']) == 0:
            
                if (settings['train_forward_dynamics']):
                    file_name_dynamics=directory+"forward_dynamics_"+str(settings['agent_name'])+".pkl"
                    f = open(file_name_dynamics, 'w')
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
                    json.dump(trainData, fp)
                    fp.close()
                    # draw data
                        
            # mean_reward = std_reward = mean_bellman_error = std_bellman_error = mean_discount_error = std_discount_error = None
            # if ( round_ % 10 ) == 0 :
                print ("Saving current masterAgent")
                
                file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+".pkl"
                f = open(file_name, 'w')
                dill.dump(agent, f)
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
        for sw in sim_workers: # Should update these more offten
            input_anchor_queue.put(None)
            
        for sw in sim_workers: # Should update these more offten
            sw.join()
            
        for lw in learning_workers: # Should update these more offten
            output_experience_queue.put(None)
            
        for lw in learning_workers: # Should update these more offten
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
            f = open(file_name_dynamics, 'w')
            dill.dump(masterAgent.getForwardDynamics(), f)
            f.close()
        
        """except: # catch *all* exceptions
        e = sys.exc_info()[0]
        print ("Error: " + str(e))
        print ("State " + str(state_) + " action " + str(pa) + " newState " + str(resultState) + " Reward: " + str(reward))
        
        """ 
          
    
    
if (__name__ == "__main__"):
    
    train(sys.argv[1])