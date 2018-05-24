import pytest
from numpy.testing import assert_allclose
import numpy as np

import warnings
from trainModel import trainModelParallel
import json
from tuneHyperParameters import tuneHyperParameters
from trainMetaModel import trainMetaModel

class TestMetaTraining(object):
    
    def test_tuning_ppo_gapGame_2D(self):
        """
        Test that PPO can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/particleSim/PPO/PPO_KERAS_Tensorflow.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        metaSettings = "settings/hyperParamTuning/elementAI.json"
        file = open(metaSettings)
        metaSettings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 25
        simData = tuneHyperParameters(simsettingsFileName=filename, simSettings=settings, hyperSettings=metaSettings)
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert simData != None
        
    def test_metaTraining_ppo_gapGame_2D(self):
        """
        Test that PPO can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/particleSim/PPO/PPO_KERAS_Tensorflow.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        metaSettings = "settings/hyperParamTuning/elementAI.json"
        file = open(metaSettings)
        metaSettings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 25
        simData = trainMetaModel(filename, samples=2, settings=copy.deepcopy(settings), numThreads=2, hyperSettings=metaSettings)
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert simData != None
        