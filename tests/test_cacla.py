# from nose.tools import timed
from numpy.testing import assert_allclose
import numpy as np
import pytest
import sys

import warnings
from trainModel import trainModelParallel
import json


class TestCACLA(object):

    # @pytest.mark.timeout(600)
    def test_cacla_keras_particleNav_10D(self):
        """
        Test that CACLA can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/particleSim/CACLA/On_Policy.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings["simulation_timeout"] = 60
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['rounds'] = 1
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -1.5
        
    # @pytest.mark.timeout(600)
    def test_cacla_keras_gapGame_2D(self):
        """
        
        """
        filename = "tests/settings/gapGame2D/CACLA/Net_FixedSTD_Tensorflow.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings["simulation_timeout"] = 60
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['rounds'] = 1
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > 0.35
    
    """   
    ### No longer suported
    # @pytest.mark.timeout(600)         
    def test_cacla_keras_off_policy_gapGame_2D(self):
        filename = "tests/settings/gapGame2D/CACLA/CACLA_KERAS_DeepCNNKeras.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 10
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > 0.65
       """     

if __name__ == '__main__':
    pytest.main([__file__])