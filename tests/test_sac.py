from numpy.testing import assert_allclose
import numpy as np
import pytest
import warnings
from trainModel import trainModelParallel
import json
import sys


class TestDDPG(object):

    # @pytest.mark.timeout(600)
    def test_ddpg_keras_particleNav_10D(self):
        """
        Test that DDPG can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/particleSim/SAC/Off_Policy_Tensorflow.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings["simulation_timeout"] = 60
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        # settings['rounds'] = 1
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -2.0
    """
    Unsupported old network models
    # @pytest.mark.timeout(600)
    def test_ddpg_lasagne_particleNav_10D(self):
        ### Test that can still learn a good policy
        filename = "tests/settings/particleSim/DDPG/Normal_OUNoise.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        # settings['rounds'] = 10
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -1.5
    """
    
    # @pytest.mark.timeout(600)
    def test_ddpg_keras_gapGame_2D(self):
        """
        
        """
        filename = "tests/settings/gapGame2D/SAC/Off_Policy_Tensorflow.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings["simulation_timeout"] = 60
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        # settings['rounds'] = 1
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > 0.35
        
        
        # @pytest.mark.timeout(600)
    def test_sac_keras_clevrObjects_HRL_MARL(self):
        from trainModel import trainModelParallel as trainModelParallel
        """
        
        """
        filename = "tests/settings/clevrObjects/SAC/HRL_Tensorflow_HLP_Only_v1.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings["simulation_timeout"] = 60
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        # settings['rounds'] = 1
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > 0.35  
            

if __name__ == '__main__':
    pytest.main([__file__])