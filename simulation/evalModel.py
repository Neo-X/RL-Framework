
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
import numpy as np
from model.ModelUtil import *
# import memory_profiler
# import resources
from simulation.simEpoch import simEpoch, simModelParrallel

# @profile(precision=5)
def evalModel(actor, exp, model, discount_factor, anchors=None, action_space_continuous=False, 
              settings=None, print_data=False, p=0.0, evaluation=False, visualizeEvaluation=None,
              bootstrapping=False, sampling=False):
    if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['hyper_train']):
        print ("Evaluating model:")
    j=0
    discounted_values = []
    bellman_errors = []
    reward_over_epocs = []
    values = []
    evalDatas = []
    epoch_=0
    for i in range(anchors): # half the anchors
        (tuples, discounted_sum, value, evalData) = simEpoch(actor, exp, 
                model, discount_factor, anchors=i, action_space_continuous=action_space_continuous, 
                settings=settings, print_data=print_data, p=p, validation=True, epoch=epoch_, evaluation=evaluation,
                visualizeEvaluation=visualizeEvaluation, bootstrapping=bootstrapping, sampling=sampling, epsilon=settings['epsilon'])
        epoch_ = epoch_ + 1
        (states, actions, result_states, rewards, falls, G_t, advantage, exp_actions) = tuples
        # print (states, actions, rewards, result_states, discounted_sum, value)
        # print ("Evaluated Actions: ", actions)
        # print ("Evaluated Rewards: ", rewards)
        batch_size_ = settings['batch_size']
        if (settings["batch_size"] == "all"):
                batch_size_ = len(states)
        if model.getExperience().samples() >= batch_size_:
            _states, _actions, _result_states, _rewards, falls, _G_ts, exp_actions = model.getExperience().get_batch(settings['batch_size'])
            error = model.bellman_error(_states, _actions, _rewards, _result_states, falls)
        else :
            error = [[0]]
            print ("Error: not enough samples in experience to check bellman error: ", model.getExperience().samples(), " needed " , settings['batch_size'] )
        # states, actions, result_states, rewards = experience.get_batch(64)
        # error = model.bellman_error(states, actions, rewards, result_states)
        # print (states, actions, rewards, result_states, discounted_sum, value)
        error = np.mean(np.fabs(error))
        # print ("Round: " + str(round_) + " Epoch: " + str(epoch) + " With reward_sum: " + str(np.sum(rewards)) + " bellman error: " + str(error))
        discounted_values.extend(np.array(discounted_sum))
        values.extend(np.array(value))
        # print ("Rewards over eval epoch: ", rewards)
        # This works better because epochs can terminate early, which is bad.
        reward_over_epocs.append(np.mean(np.array(rewards)))
        bellman_errors.append(error)
        evalDatas.append(evalData)
    if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
        print ("Reward for best epoch: " + str(np.argmax(reward_over_epocs)) + " is " + str(np.max(reward_over_epocs)))
        print ("reward_over_epocs" + str(reward_over_epocs))
    if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['debug']):
        print ("Discounted sum: ", np.array(discounted_values))
        print ("Initial values: ", np.array(values))
        for i in range(len(discounted_values)):
            print ("len(discounted_values[",i,"]): ", np.array(discounted_values[i]).shape, " len(values[",i,"]): ", 
                   np.array(values[i]).shape)
    mean_reward = np.mean(reward_over_epocs)
    std_reward = np.std(reward_over_epocs)
    mean_bellman_error = np.mean(bellman_errors)
    std_bellman_error = np.std(bellman_errors)
    mean_discount_error = np.mean(np.array(discounted_values) - np.array(values))
    std_discount_error = np.std(np.array(discounted_values) - np.array(values))
    mean_eval = np.mean(evalDatas)
    std_eval = np.std(evalDatas)
    
    discounted_values = []
    reward_over_epocs = []
    bellman_errors = []
        
    return (mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error,
            mean_eval, std_eval)

# @profile(precision=5)
def evalModelParrallel(input_anchor_queue, eval_episode_data_queue, model, settings, anchors=None):
    if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['hyper_train']):
        print ("Evaluating model Parrallel:")
    
    if ( 'value_function_batch_size' in settings):
        batch_size=settings["value_function_batch_size"]
    else:
        batch_size=settings["batch_size"]
    j=0
    discounted_values = []
    bellman_errors = []
    reward_over_epocs = []
    values = []
    evalDatas = []
    epoch_=0
    i = 0 
    while i < anchors: # half the anchors
        
        j = 0
        while (j < abs(settings['num_available_threads'])) and ( (i + j) < anchors):
            episodeData = {}
            episodeData['data'] = i
            episodeData['type'] = 'eval'
            if (settings['on_policy']):
                input_anchor_queue[j].put(episodeData)
            else:
                input_anchor_queue.put(episodeData)
            j += 1
            
        # for anchs in anchors: # half the anchors
        j = 0
        while (j < abs(settings['num_available_threads'])) and ( (i + j) < anchors):
            (tuples, discounted_sum, value, evalData) =  eval_episode_data_queue.get()
            j += 1
            """
            simEpoch(actor, exp, 
                    model, discount_factor, anchors=anchs, action_space_continuous=action_space_continuous, 
                    settings=settings, print_data=print_data, p=0.0, validation=True, epoch=epoch_, evaluation=evaluation,
                    visualizeEvaluation=visualizeEvaluation)
            """
            epoch_ = epoch_ + 1
            (states, actions, result_states, rewards, falls, G_ts, advantage, exp_actions) = tuples
            # print (states, actions, rewards, result_states, discounted_sum, value)
            # print ("Evaluated Actions: ", actions)
            # print ("Evaluated Rewards: ", rewards)
            if model.getExperience().samples() >= batch_size:
                _states, _actions, _result_states, _rewards, falls, _G_ts, exp_actions, advantage = model.getExperience().get_batch(batch_size)
                error = model.bellman_error(_states, _actions, _rewards, _result_states, falls)
                # print("Episode bellman error: ", error)
            else :
                error = [[0]]
                print ("Error: not enough samples in experience to check bellman error: ", model.getExperience().samples(), " needed " , batch_size)
            # states, actions, result_states, rewards = experience.get_batch(64)
            # error = model.bellman_error(states, actions, rewards, result_states)
            # print (states, actions, rewards, result_states, discounted_sum, value)
            error = np.mean(np.fabs(error))
            # print ("Round: " + str(round_) + " Epoch: " + str(epoch) + " With reward_sum: " + str(np.sum(rewards)) + " bellman error: " + str(error))
            discounted_values.append(discounted_sum)
            values.append(value)
            # print ("Rewards over eval epoch: ", rewards)
            # This works better because epochs can terminate early, which is bad.
            reward_over_epocs.append(np.mean(np.array(rewards)))
            bellman_errors.append(error)
            evalDatas.append(evalData)
        i += j
    if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['train']):
        print ("Reward for best epoch: " + str(np.argmax(reward_over_epocs)) + " is " + str(np.max(reward_over_epocs)))
        print ("reward_over_epocs" + str(reward_over_epocs))
    if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"]['debug']):
        print ("Discounted sum: ", np.array(discounted_values))
        print ("Initial values: ", np.array(values))
        for i in range(len(discounted_values)):
            print ("len(discounted_values[",i,"]): ", np.array(discounted_values[i]).shape, " len(values[",i,"]): ", 
                   np.array(values[i]).shape)
    mean_reward = np.mean(reward_over_epocs)
    std_reward = np.std(reward_over_epocs)
    mean_bellman_error = np.mean(bellman_errors)
    std_bellman_error = np.std(bellman_errors)
    mean_discount_error = np.mean(np.array(discounted_values) - np.array(values))
    std_discount_error = np.std(np.array(discounted_values) - np.array(values))
    mean_eval = np.mean(evalDatas)
    std_eval = np.std(evalDatas)
    
    discounted_values = []
    reward_over_epocs = []
    bellman_errors = []
        
    return (mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error,
            mean_eval, std_eval)
