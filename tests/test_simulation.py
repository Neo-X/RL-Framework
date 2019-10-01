from nose.tools import timed
from numpy.testing import assert_allclose
import numpy as np
import pytest
import warnings
from model.ModelUtil import *
from simulation.simEpoch import simEpoch, simModelParrallel
from simulation.evalModel import evalModelParrallel, evalModel
import json
from actor.DoNothingActor import DoNothingActor
from sim.DummyEnv import DummyEnv
from algorithm.ModelDummy import ModelDummy
from trainModel import trainModelParallel
import sys

class TestSimulation(object):
    
    # @pytest.mark.timeout(600)
    def test_collect_tuples(self):
        filename = "tests/settings/particleSim/PPO/PPO_KERAS_Tensorflow.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'hyper_train'
        settings["simulation_timeout"] = 600
        
        settings["perform_multiagent_training"] = 1
        settings['state_bounds'] = [[-1] * 11, [1] * 11]
        agent = DoNothingActor(settings_=settings, experience=None)
        env = DummyEnv(exp=None, settings=settings)
        modelDummy = ModelDummy(model=None, n_in=11, n_out=len(settings['action_bounds'][0]), 
                                state_bounds=settings['state_bounds'], action_bounds=settings['action_bounds'], 
                                reward_bound=settings['reward_bounds'], settings_=settings)
        env.setMaxT(settings['max_epoch_length'])
        out = simEpoch(actor=agent, exp=env, model=modelDummy, discount_factor=0.9, settings=settings, anchors=None, action_space_continuous=True, 
                       print_data=False, p=0.0, validation=False, epoch=0, evaluation=False, _output_queue=None, bootstrapping=False, visualizeEvaluation=None,
             sampling=False, epsilon=0.2,
             worker_id=None)
        
        (tuples, tmp_discounted_sum, tmp_baselines_, evalData) = out
        (tmp_states, tmp_actions, tmp_res_states, tmp_rewards, tmp_falls, tmp_G_ts, tmp_advantage, tmp_exp_actions, datas) = tuples
        # print ('out: ', out)
        assert ( len(tmp_states) == settings['max_epoch_length'])
        assert ( len(tmp_actions) == settings['max_epoch_length'])
        assert ( len(tmp_res_states) == settings['max_epoch_length'])
        assert ( len(tmp_rewards) == settings['max_epoch_length'])
        assert ( len(tmp_falls) == settings['max_epoch_length'])
        assert ( len(tmp_G_ts) == settings['max_epoch_length'])
        assert ( len(tmp_advantage) == settings['max_epoch_length'])
        assert ( len(tmp_exp_actions) == settings['max_epoch_length'])
       
     
    # @pytest.mark.timeout(600)    
    def test_collect_tuples_discount_sum(self):
        filename = "tests/settings/particleSim/PPO/PPO_KERAS_Tensorflow.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'hyper_train'
        settings["simulation_timeout"] = 60
        
        settings["perform_multiagent_training"] = 1
        settings['state_bounds'] = [[-1] * 11, [1] * 11]
        agent = DoNothingActor(settings_=settings, experience=None)
        env = DummyEnv(exp=None, settings=settings)
        modelDummy = ModelDummy(model=None, n_in=11, n_out=len(settings['action_bounds'][0]),
                                state_bounds=settings['state_bounds'], action_bounds=settings['action_bounds'], 
                                reward_bound=settings['reward_bounds'], settings_=settings)
        env.setMaxT(settings['max_epoch_length'])
        out = simEpoch(actor=agent, exp=env, model=modelDummy, discount_factor=0.9, settings=settings, anchors=None, action_space_continuous=True, 
                       print_data=False, p=0.0, validation=False, epoch=0, evaluation=False, _output_queue=None, bootstrapping=False, visualizeEvaluation=None,
             sampling=False, epsilon=0.2,
             worker_id=None)
        
        (tuples, tmp_discounted_sum, tmp_baselines_, evalData) = out
        (tmp_states, tmp_actions, tmp_res_states, tmp_rewards, tmp_falls, tmp_G_ts, tmp_advantage, tmp_exp_actions, datas) = tuples
        # print ('out: ', out)
        assert ( len(tmp_discounted_sum) == settings['max_epoch_length'])
        assert ( len(tmp_baselines_) == settings['max_epoch_length'])
        assert ( len(evalData) == 1)   
    
    
    # @pytest.mark.timeout(600)   
    def test_collect_tuples_fidd_length_episodes(self):
        filename = "tests/settings/particleSim/PPO/PPO_KERAS_Tensorflow.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'hyper_train'
        settings["simulation_timeout"] = 60
        
        settings["perform_multiagent_training"] = 1
        settings['state_bounds'] = [[-1] * 11, [1] * 11]
        agent = DoNothingActor(settings_=settings, experience=None)
        env = DummyEnv(exp=None, settings=settings)
        modelDummy = ModelDummy(model=None, n_in=11, n_out=len(settings['action_bounds'][0]),
                                state_bounds=settings['state_bounds'], action_bounds=settings['action_bounds'], 
                                reward_bound=settings['reward_bounds'], settings_=settings)
        
        for i in range(1, settings['max_epoch_length']):
            env.setMaxT(i)
            out = simEpoch(actor=agent, exp=env, model=modelDummy, discount_factor=0.9, settings=settings, anchors=None, action_space_continuous=True, 
                           print_data=False, p=0.0, validation=False, epoch=0, evaluation=False, _output_queue=None, bootstrapping=False, visualizeEvaluation=None,
                 sampling=False, epsilon=0.2,
                 worker_id=None)
        
            (tuples, tmp_discounted_sum, tmp_baselines_, evalData) = out
            (tmp_states, tmp_actions, tmp_res_states, tmp_rewards, tmp_falls, tmp_G_ts, tmp_advantage, tmp_exp_actions, datas) = tuples
        # print ('out: ', out)
            assert ( len(tmp_states) == i)
            assert ( len(tmp_actions) == i)
            assert ( len(tmp_res_states) == i)
            assert ( len(tmp_rewards) == i)
            assert ( len(tmp_falls) == i)
            assert ( len(tmp_G_ts) == i)
            assert ( len(tmp_advantage) == i)
            assert ( len(tmp_exp_actions) == i)
    
    # @pytest.mark.timeout(600)       
    def test_get_state_size_from_env(self):
        filename = "tests/settings/gapGame2D/PPO/SingleNet_FixedSTD_Tensorflow.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['state_bounds'] = "ask_env"
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 2
        settings['pretrain_critic'] = 0
        settings["simulation_timeout"] = 60
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert simData != None
        
    # @pytest.mark.timeout(600)       
    def test_get_action_size_from_env(self):
        filename = "tests/settings/gapGame2D/PPO/SingleNet_FixedSTD_Tensorflow.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['action_bounds'] = "ask_env"
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 2
        settings['pretrain_critic'] = 0
        settings["simulation_timeout"] = 60
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert simData != None   
    

if __name__ == '__main__':
    pytest.main([__file__])