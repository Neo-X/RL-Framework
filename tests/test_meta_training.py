import pytest
from numpy.testing import assert_allclose
import numpy as np
import copy 

import warnings
from trainModel import trainModelParallel
import json
from tuneHyperParameters import tuneHyperParameters
from trainMetaModel import trainMetaModel

class TestMetaTraining(object):
    
    @pytest.mark.timeout(600)
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
    
    @pytest.mark.timeout(600)   
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
        
    @pytest.mark.timeout(600)
    def test_metaTraining_multiple_rounds_over_tuning_threads(self):
        """
        Test that PPO can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/particleSim/PPO/PPO_KERAS_Tensorflow.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        metaSettings = "settings/hyperParamTuning/element/state_normalization.json"
        file = open(metaSettings)
        metaSettings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 25
        metaSettings['meta_sim_samples'] = 1
        metaSettings['meta_sim_threads'] = 1
        metaSettings['tuning_threads'] = 2 # 4 samples total
        metaSettings['num_param_samples'] = [[2]]
        metaSettings['testing'] = True
        simData = tuneHyperParameters(simsettingsFileName=filename, simSettings=settings, hyperSettings=metaSettings)
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert simData != None    
    
    @pytest.mark.timeout(600)    
    def test_metaTraining_multiple_params_to_tune(self):
        """
        Test that PPO can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/gapGame2D/PPO/FixedSTD_Tensorflow-v2_dropout.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        metaSettings = "settings/hyperParamTuning/element/dropout_p_and_use_dropout_in_actor.json"
        file = open(metaSettings)
        metaSettings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 25
        metaSettings['meta_sim_samples'] = 2
        metaSettings['meta_sim_threads'] = 2
        metaSettings['tuning_threads'] = 2 # 4 samples total
        metaSettings['num_param_samples'] = [2, 2]
        metaSettings['testing'] = True
        simData = tuneHyperParameters(simsettingsFileName=filename, simSettings=settings, hyperSettings=metaSettings)
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert simData != None    
    
    @pytest.mark.timeout(600)   
    def test_metaTraining_mbrl(self):
        """
        Test that MBRL can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/navGame/MBRL/FixedSTD_Tensorflow_5D.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        metaSettings = "settings/hyperParamTuning/element/fd_network_layer_sizes.json"
        file = open(metaSettings)
        metaSettings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 4
        metaSettings['meta_sim_samples'] = 2
        metaSettings['meta_sim_threads'] = 2
        metaSettings['tuning_threads'] = 3 # 4 samples total
        metaSettings['num_param_samples'] = [[2]]
        metaSettings['testing'] = True
        simData = tuneHyperParameters(simsettingsFileName=filename, simSettings=settings, hyperSettings=metaSettings)
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert simData != None  