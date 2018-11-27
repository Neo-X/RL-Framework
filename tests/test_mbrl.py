from nose.tools import timed
from numpy.testing import assert_allclose
import numpy as np
import pytest
import warnings
from trainModel import trainModelParallel
import json


class TestMBRL(object):

    # @pytest.mark.timeout(600)
    def test_mbrl_particleNav_5D_Dropout(self):
        """
        Test that MBRL can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/navGame/MBRL/FixedSTD_Dropout_Tensorflow_5D.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 2
        # settings['rollouts'] = 4
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > 0.5
        
    def test_mbrl_particleNav_5D_Ensamble(self):
        """
        Test that MBRL can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/navGame/MBRL/FixedSTD_Ensamble_Tensorflow_5D.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 2
        # settings['rollouts'] = 4
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > 0.5

if __name__ == '__main__':
    pytest.main([__file__])