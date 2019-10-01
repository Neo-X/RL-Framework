# from nose.tools import timed
from numpy.testing import assert_allclose
import numpy as np
import pytest
import warnings
from trainModel import trainModelParallel
import json
import sys


class TestPPOMore(object):

    """
    # @pytest.mark.timeout(600)
    def test_ppo_particleNav_2D(self):
        ### Test that PPO can still learn a good policy on 2d particle sim
        filename = "tests/settings/particleSim/PPO/PPO.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        # settings['rounds'] = 2
        settings['rollouts'] = 4
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -0.5
    
    # @pytest.mark.timeout(600)   
    def test_ppo_keras_particleNav_2D(self):
        ### Test that PPO can still learn a good policy on 2d particle sim
        filename = "tests/settings/particleSim/PPO/PPO_KERAS.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        # settings['rounds'] = 2
        settings['rollouts'] = 4
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -0.5
    """ 
    
    # @pytest.mark.timeout(600)
    def test_ppo_keras_gapGame_2D(self):
        """
            Test that PPO can still learn a good policy on 2d gapgame sim
        """
        filename = "tests/settings/gapGame2D/PPO/SingleNet_FixedSTD.json"
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
        assert np.mean(simData['mean_reward'][-5:]) > 0.65
    
    # @pytest.mark.timeout(600)   
    def test_ppo_keras_tensorflow_gapGame_2D(self):
        """
            Test that PPO can still learn a good policy on 2d gapgame sim
        """
        filename = "tests/settings/gapGame2D/PPO/SingleNet_FixedSTD_Tensorflow.json"
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
        assert np.mean(simData['mean_reward'][-5:]) > 0.65
            

if __name__ == '__main__':
    pytest.main([__file__])