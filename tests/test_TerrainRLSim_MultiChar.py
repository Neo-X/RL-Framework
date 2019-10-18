# from nose.tools import timed
from numpy.testing import assert_allclose
import numpy as np
import pytest
import warnings
from trainModel import trainModelParallel
import json
import sys


class TestVizImitation(object):
    
    # @pytest.mark.timeout(600)
    def test_terrainRLSim_MultiCharRugby_reward(self):
        filename = "tests/settings/terrainRLMultiChar/HLC/PPO/ScenarioSpace_Rubgy_WithObs_SimpleReward_3_NEWLLC_Hetero_v1.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['visualize_learning'] = False
        # settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 2
        settings['num_threads'] = 2
        settings['rollouts'] = 4
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert not (simData is None)
    
    # @pytest.mark.timeout(600)
    def test_terrainRLSim_MultiCharRugby_MADDPG(self):
        filename = "tests/settings/terrainRLMultiChar/HLC/DDPG/ScenarioSpace_Chase_WithObs_SimpleReward_5_NEWLLC_Hetero_v1.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['visualize_learning'] = False
        # settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 2
        settings['num_threads'] = 2
        settings['rollouts'] = 4
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert not (simData is None)
    

if __name__ == '__main__':
    pytest.main([__file__])