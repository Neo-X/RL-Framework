from nose.tools import timed
from numpy.testing import assert_allclose
import numpy as np
import pytest
import warnings
from trainModel import trainModelParallel
import json


class TestLearning(object):
    """
        Test some of the more basic learning features that can be used be all algorithms.
    """

    # @pytest.mark.timeout(600)
    def test_pretrain_critic(self):
        filename = "tests/settings/particleSim/CACLA/On_Policy_Tensorflow.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 2
        settings['pretrain_critic'] = 2
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert simData != None
        
        
    # @pytest.mark.timeout(600)
    def test_pretrain_fd(self):
        filename = "tests/settings/particleSim/CACLA/On_Policy_Tensorflow_FD.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 2
        settings['pretrain_fd'] = 2
        settings['train_reward_predictor'] = False
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert simData != None
        
        # @pytest.mark.timeout(600)
    def test_pretrain_fd_with_reward(self):
        filename = "tests/settings/particleSim/CACLA/On_Policy_Tensorflow_FD.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 2
        settings['pretrain_fd'] = 2
        settings['train_reward_predictor'] = True
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert simData != None

if __name__ == '__main__':
    pytest.main([__file__])