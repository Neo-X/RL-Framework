from nose.tools import timed
from numpy.testing import assert_allclose
import numpy as np

import warnings
from trainModel import trainModelParallel
import json

### Disable this test for now, it is way to slow...
class DontTestTerrainRLImitate(object):

    @timed(600)
    def test_ppo_keras_walk_flat_2D(self):
        """
        Test that PPO can still learn a good policy
        """
        filename = "tests/settings/terrainRLImitation/PPO_Flat_SingleNet.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > 0.5
        
if __name__ == '__main__':
    pytest.main([__file__])