# from nose.tools import timed
from numpy.testing import assert_allclose
import numpy as np

import warnings
from trainModel import trainModelParallel
import json

### Disable this test for now, it is way to slow...
class DontTestReacher(object):

    # @pytest.mark.timeout(600)
    def test_ppo_reacher_2D(self):
        """
        Test that PPO can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/openAIGym/Reacher/PPO.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > 0.025
        
if __name__ == '__main__':
    pytest.main([__file__])