import pytest
from numpy.testing import assert_allclose
import numpy as np

import warnings
from trainModel import trainModelParallel
import json


class TestLearning(object):
    """
        Test some of the more basic learning features that can be used be all algorithms.
    """

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
            

if __name__ == '__main__':
    pytest.main([__file__])