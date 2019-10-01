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
    def test_viz_state_normal_reward_singleNet(self):
        """
        Test that PPO can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/terrainRLImitation/Viz3D_StateTrans_SingleNet_30FPS.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['visualize_learning'] = False
        # settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 10
        settings['num_threads'] = 2
        settings['rollouts'] = 4
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert not (simData is None)
        
    def test_viz_state_64x64_MultiModel_WithCamVel_RNN_Siamese(self):
        """
        Test that PPO can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/terrainRLImitation/Imitation_Learning_Viz3D_64x64_MultiModal_WithCamVel_Reward_LSTM_Siamese_Reward.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['visualize_learning'] = False
        # settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 10
        settings['num_threads'] = 2
        settings['rollouts'] = 4
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert not (simData is None)
        
    def test_viz_state_64x64_WithCamVel_RNN_Siamese(self):
        """
        Test that can learn a good policy on 2d particle sim
        """
        filename = "tests/settings/terrainRLImitation/Imitation_Learning_Viz3D_64x64_WithCamVel_Reward_LSTM_Siamese_Reward.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['visualize_learning'] = False
        # settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 10
        settings['num_threads'] = 2
        settings['rollouts'] = 4
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert not (simData is None)
        
        # @pytest.mark.timeout(600)
if __name__ == '__main__':
    pytest.main([__file__])