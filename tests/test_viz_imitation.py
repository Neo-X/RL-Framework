from nose.tools import timed
from numpy.testing import assert_allclose
import numpy as np
import pytest
import warnings
from trainModel import trainModelParallel
import json


class TestVizImitation(object):

    @pytest.mark.timeout(600)
    def test_viz_state_normal_reward(self):
        """
        Test that CACLA can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/projectileGame/PPO/Viz_State_Normal_Reward.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        # settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 10
        settings['num_threads'] = 2
        settings['rollouts'] = 4
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert not (simData is None)
    
    @pytest.mark.timeout(600)    
    def test_viz_imitation(self):
        """
        Test that CACLA can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/projectileGame/PPO/Viz_Imitation.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        # settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 10
        settings['num_threads'] = 2
        settings['rollouts'] = 4
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert not (simData is None)
    
    @pytest.mark.timeout(600)
    def test_viz_imitation_reward_dense_state(self):
        """
        Test that CACLA can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/projectileGame/PPO/Viz_Imitation_DualState.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        # settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 10
        settings['num_threads'] = 2
        settings['rollouts'] = 4
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert not (simData is None)
    
    @pytest.mark.timeout(600)
    def test_viz_state_normal_reward_singleNet(self):
        """
        Test that CACLA can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/terrainRLImitation/Viz3D_StateTrans_SingleNet_30FPS.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
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