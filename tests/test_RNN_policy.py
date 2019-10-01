# from nose.tools import timed
from numpy.testing import assert_allclose
import numpy as np
import pytest
import warnings
from trainModel import trainModelParallel
import json
import sys


class TestPPO(object):

   
    # @pytest.mark.timeout(600)
    def test_ppo_keras_tensorflow_particleNav_2D_lstm(self):
        """
        Test that PPO can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/particleSim/PPO/RNNPolicy_5D.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        # settings['rounds'] = 2
        settings['rollouts'] = 4
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -0.5
    
    def test_ppo_keras_tensorflow_particleNav_2D_lstm_critic(self):
        """u
        Test that PPO can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/particleSim/PPO/RNNCritic_5D.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        # settings['rounds'] = 2
        settings['rollouts'] = 4
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -0.5 

            

if __name__ == '__main__':
    pytest.main([__file__])