import pytest
from numpy.testing import assert_allclose
import numpy as np

import warnings
from trainModel import trainModelParallel
from ModelEvaluation import modelEvaluation
import json


class TestSaveAndLoad(object):

    def test_keras_gapGame_2D_save_and_load_singleNet(self):
        """
            Test that PPO can still learn a good policy on 2d gapgame sim
            If this does not crash, things are good..
        """
        filename = "tests/settings/gapGame2D/PPO/SingleNet_FixedSTD_Tensorflow-v2.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 10
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        settings['visualize_expected_value'] = False
        modelEvaluation(filename, runLastModel=False, settings=settings, render=False)
                    

if __name__ == '__main__':
    pytest.main([__file__])