import pytest
from numpy.testing import assert_allclose
import numpy as np

import warnings
from trainModel import trainModelParallel
import json


class TestDDPG(object):

    def test_ddpg_keras_particleNav_10D(self):
        """
        Test that CACLA can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/particleSim/DDPG/On_Policy.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -0.5
        
    def test_ddpg_lasagne_particleNav_10D(self):
        """
        Test that can still learn a good policy
        """
        filename = "tests/settings/particleSim/DDPG/Normal_OUNoise.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -1.5
        
    def test_ddpg_keras_gapGame_2D(self):
        """
        
        """
        filename = "tests/settings/gapGame2D/DDPG/Net_FixedSTD.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > 0.65
            

if __name__ == '__main__':
    pytest.main([__file__])