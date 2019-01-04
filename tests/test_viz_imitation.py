from nose.tools import timed
from numpy.testing import assert_allclose
import numpy as np
import pytest
import warnings
from trainModel import trainModelParallel
import json
import sys


class TestVizImitation(object):

    # @pytest.mark.timeout(600)
    def test_viz_state_normal_reward(self):
        """
        Test that CACLA can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/projectileGame/PPO/Viz_State_Normal_Reward.json"
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
    def test_viz_imitation(self):
        """
        Test that CACLA can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/projectileGame/PPO/Viz_Imitation.json"
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
    def test_viz_imitation_reward_dense_state(self):
        """
        Test that CACLA can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/projectileGame/PPO/Viz_Imitation_DualState.json"
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
        
        # @pytest.mark.timeout(600)
    
    def test_viz_state_32x32_LSTM_Siamese(self):
        """
        Test that CACLA can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/projectileGame/CACLA/Imitation_Learning_Viz_32x32_1Sub_LSTM_FD_Reward_Encode.json"
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
        
    def test_viz_state_32x32_withCamVel_RNN_Siamese(self):
        """
        Test that CACLA can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/projectileGame/CACLA/Imitation_Learning_VizWithCamVel_32x32_1Sub_LSTM_FD_Reward_Encode.json"
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
        
    def test_viz_state_32x32_multiModal_withCamVel_Siamese_lstm(self):
        """
        Test that CACLA can still learn a good policy on 2d particle sim
        """
        filename = "settings/projectileGame/CACLA/Imitation_Learning_Imitation_Dense_Reward_LSTM_EncoderDecoder.json"
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
        
    def test_viz_state_32x32_encoder_decoder_lstm(self):
        """
            Test that CACLA can still learn a good policy on 2d particle sim
        """
        filename = "settings/projectileGame/CACLA/Imitation_Learning_Imitation_Dense_Reward_LSTM_EncoderDecoder.json"
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

if __name__ == '__main__':
    pytest.main([__file__])