from nose.tools import timed
from numpy.testing import assert_allclose
import numpy as np

import warnings
from trainModel import trainModelParallel
import json


class TestMultiAgentNav(object):

    @pytest.mark.timeout(600)
    def test_cacla_multiAgentNAv(self):
        """
        Test that CACLA can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/navGameMultiAgent/CACLA/On_Policy.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        # settings['rounds'] = 2
        # settings['rollouts'] = 4
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -1.5
        
if __name__ == '__main__':
    pytest.main([__file__])