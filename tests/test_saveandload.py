import pytest
from numpy.testing import assert_allclose
import numpy as np

import warnings
from trainModel import trainModelParallel
from ModelEvaluation import modelEvaluation
import json


class TestSaveAndLoad(object):

    def test_ppo_keras_gapGame_2D_save_and_load_singleNet(self):
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
        settings["bootstrap_samples"] = 1000
        settings['rounds'] = 4
        settings["epochs"] = 2
        settings['eval_epochs'] = 2
        settings["plotting_update_freq_num_rounds"] = 2
        settings["saving_update_freq_num_rounds"] =  2
        settings["state_normalization"] = "variance"
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        settings['visualize_expected_value'] = False
        evalData = modelEvaluation(filename, runLastModel=False, settings=settings, render=False)
        print ("state bounds: ", simData['masterAgent'].getStateBounds(), evalData['masterAgent'].getStateBounds())
               
        assert np.allclose(simData['masterAgent'].getRewardBounds(), evalData['masterAgent'].getRewardBounds())
        assert np.allclose(simData['masterAgent'].getActionBounds(), evalData['masterAgent'].getActionBounds())
        assert np.allclose(simData['masterAgent'].getStateBounds(), evalData['masterAgent'].getStateBounds())
        
    def test_cacla_tensorflow_gapGame_2D_save_and_load_singleNet(self):
        """
            Test that PPO can still learn a good policy on 2d gapgame sim
            If this does not crash, things are good..
        """
        filename = "tests/settings/gapGame2D/CACLA/Net_FixedSTD_Tensorflow.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings["bootstrap_samples"] = 1000
        settings['rounds'] = 4
        settings["epochs"] = 2
        settings['eval_epochs'] = 2
        settings["plotting_update_freq_num_rounds"] = 2
        settings["saving_update_freq_num_rounds"] =  2
        settings["state_normalization"] = "variance"
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        settings['visualize_expected_value'] = False
        evalData = modelEvaluation(filename, runLastModel=False, settings=settings, render=False)
        print ("state bounds: ", simData['masterAgent'].getStateBounds(), evalData['masterAgent'].getStateBounds())
               
        assert np.allclose(simData['masterAgent'].getRewardBounds(), evalData['masterAgent'].getRewardBounds())
        assert np.allclose(simData['masterAgent'].getActionBounds(), evalData['masterAgent'].getActionBounds())
        assert np.allclose(simData['masterAgent'].getStateBounds(), evalData['masterAgent'].getStateBounds())
        
    def test_cacla_keras_gapGame_2D_save_and_load_singleNet(self):
        """
            Test that PPO can still learn a good policy on 2d gapgame sim
            If this does not crash, things are good..
        """
        filename = "tests/settings/gapGame2D/CACLA/Net_FixedSTD.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings["bootstrap_samples"] = 1000
        settings['rounds'] = 6
        settings["epochs"] = 2
        settings['eval_epochs'] = 2
        settings["plotting_update_freq_num_rounds"] = 2
        settings["saving_update_freq_num_rounds"] =  2
        settings["state_normalization"] = "variance"
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        settings['visualize_expected_value'] = False
        evalData = modelEvaluation(filename, runLastModel=False, settings=settings, render=False)
        
        assert np.allclose(simData['masterAgent'].getRewardBounds(), evalData['masterAgent'].getRewardBounds())
        assert np.allclose(simData['masterAgent'].getActionBounds(), evalData['masterAgent'].getActionBounds())
        assert np.allclose(simData['masterAgent'].getStateBounds(), evalData['masterAgent'].getStateBounds())
                    

if __name__ == '__main__':
    pytest.main([__file__])