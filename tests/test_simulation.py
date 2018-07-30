import pytest
from numpy.testing import assert_allclose
import numpy as np

import warnings
from model.ModelUtil import *
from simulation.simEpoch import simEpoch, simModelParrallel
from simulation.evalModel import evalModelParrallel, evalModel
import json
from actor.DoNothingActor import DoNothingActor
from sim.DummyEnv import DummyEnv
from algorithm.ModelDummy import ModelDummy

class TestSimulation(object):
    
    @pytest.mark.timeout(600)
    def test_collect_tuples(self):
        filename = "tests/settings/gapGame2D/PPO/SingleNet_FixedSTD.json"
        file = open(filename)
        settings_ = json.load(file)
        file.close()
        settings_['visualize_learning'] = False
        settings_['shouldRender'] = False
        settings['print_level'] = 'hyper_train'
        agent = DoNothingActor(settings_=settings_, experience=None)
        env = DummyEnv(exp=None, settings=settings_)
        modelDummy = ModelDummy(model=None, n_in=11, n_out=7, 
                                state_bounds=None, action_bounds=None, 
                                reward_bound=None, settings_=settings_)
        env.setMaxT(settings_['max_epoch_length'])
        out = simEpoch(actor=agent, exp=env, model=modelDummy, discount_factor=0.9, settings=settings_, anchors=None, action_space_continuous=True, 
                       print_data=False, p=0.0, validation=False, epoch=0, evaluation=False, _output_queue=None, bootstrapping=False, visualizeEvaluation=None,
             sampling=False, epsilon=0.2,
             worker_id=None)
        
        (tuples, tmp_discounted_sum, tmp_baselines_, evalData) = out
        (tmp_states, tmp_actions, tmp_res_states, tmp_rewards, tmp_falls, tmp_G_ts, tmp_advantage, tmp_exp_actions) = tuples
        # print ('out: ', out)
        assert ( len(tmp_states) == settings_['max_epoch_length'])
        assert ( len(tmp_actions) == settings_['max_epoch_length'])
        assert ( len(tmp_res_states) == settings_['max_epoch_length'])
        assert ( len(tmp_rewards) == settings_['max_epoch_length'])
        assert ( len(tmp_falls) == settings_['max_epoch_length'])
        assert ( len(tmp_G_ts) == settings_['max_epoch_length'])
        assert ( len(tmp_advantage) == settings_['max_epoch_length'])
        assert ( len(tmp_exp_actions) == settings_['max_epoch_length'])
        
    @pytest.mark.timeout(600)    
    def test_collect_tuples_discount_sum(self):
        filename = "tests/settings/gapGame2D/PPO/SingleNet_FixedSTD.json"
        file = open(filename)
        settings_ = json.load(file)
        file.close()
        settings_['visualize_learning'] = False
        settings_['shouldRender'] = False
        settings['print_level'] = 'hyper_train'
        agent = DoNothingActor(settings_=settings_, experience=None)
        env = DummyEnv(exp=None, settings=settings_)
        modelDummy = ModelDummy(model=None, n_in=11, n_out=7, 
                                state_bounds=None, action_bounds=None, 
                                reward_bound=None, settings_=settings_)
        env.setMaxT(settings_['max_epoch_length'])
        out = simEpoch(actor=agent, exp=env, model=modelDummy, discount_factor=0.9, settings=settings_, anchors=None, action_space_continuous=True, 
                       print_data=False, p=0.0, validation=False, epoch=0, evaluation=False, _output_queue=None, bootstrapping=False, visualizeEvaluation=None,
             sampling=False, epsilon=0.2,
             worker_id=None)
        
        (tuples, tmp_discounted_sum, tmp_baselines_, evalData) = out
        (tmp_states, tmp_actions, tmp_res_states, tmp_rewards, tmp_falls, tmp_G_ts, tmp_advantage, tmp_exp_actions) = tuples
        # print ('out: ', out)
        assert ( len(tmp_discounted_sum) == settings_['max_epoch_length'])
        assert ( len(tmp_baselines_) == settings_['max_epoch_length'])
        assert ( len(evalData) == 1)   
    
    @pytest.mark.timeout(600)   
    def test_collect_tuples_fidd_length_episodes(self):
        filename = "tests/settings/gapGame2D/PPO/SingleNet_FixedSTD.json"
        file = open(filename)
        settings_ = json.load(file)
        file.close()
        settings_['visualize_learning'] = False
        settings_['shouldRender'] = False
        settings['print_level'] = 'hyper_train'
        agent = DoNothingActor(settings_=settings_, experience=None)
        env = DummyEnv(exp=None, settings=settings_)
        modelDummy = ModelDummy(model=None, n_in=11, n_out=7, 
                                state_bounds=None, action_bounds=None, 
                                reward_bound=None, settings_=settings_)
        
        for i in range(1, settings_['max_epoch_length']):
            env.setMaxT(i)
            out = simEpoch(actor=agent, exp=env, model=modelDummy, discount_factor=0.9, settings=settings_, anchors=None, action_space_continuous=True, 
                           print_data=False, p=0.0, validation=False, epoch=0, evaluation=False, _output_queue=None, bootstrapping=False, visualizeEvaluation=None,
                 sampling=False, epsilon=0.2,
                 worker_id=None)
        
            (tuples, tmp_discounted_sum, tmp_baselines_, evalData) = out
            (tmp_states, tmp_actions, tmp_res_states, tmp_rewards, tmp_falls, tmp_G_ts, tmp_advantage, tmp_exp_actions) = tuples
        # print ('out: ', out)
            assert ( len(tmp_states) == i)
            assert ( len(tmp_actions) == i)
            assert ( len(tmp_res_states) == i)
            assert ( len(tmp_rewards) == i)
            assert ( len(tmp_falls) == i)
            assert ( len(tmp_G_ts) == i)
            assert ( len(tmp_advantage) == i)
            assert ( len(tmp_exp_actions) == i)
    
    @pytest.mark.timeout(600)       
    def test_get_state_size_from_env(self):
        filename = "tests/settings/gapGame2D/PPO/FixedSTD_Tensorflow-v2.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['state_bounds'] = "ask_env"
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 2
        settings['pretrain_critic'] = 0
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert simData != None   

if __name__ == '__main__':
    pytest.main([__file__])