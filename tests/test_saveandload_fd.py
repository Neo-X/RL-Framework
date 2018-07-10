import pytest
from numpy.testing import assert_allclose
import numpy as np

import warnings
from trainModel import trainModelParallel
from ModelEvaluation import modelEvaluation
import json


class TestSaveAndLoadFD(object):

    def test_MBRL_keras_cannon_2D_save_and_load_FD_scaling_params(self):
        filename = "tests/settings/cannonGame/MBRL/FixedSTD_Tensorflow-v2.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings["bootstrap_samples"] = 1000
        settings['rounds'] = 1
        settings["epochs"] = 2
        settings['eval_epochs'] = 2
        settings["plotting_update_freq_num_rounds"] = 2
        settings["saving_update_freq_num_rounds"] =  2
        settings["state_normalization"] = "variance"
        settings["return_model"] = True
        settings["num_available_threads"] = 2
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        settings['visualize_expected_value'] = False
        evalData = modelEvaluation(filename, runLastModel=False, settings=settings, render=False)
        
        assert np.allclose(simData['masterAgent'].getForwardDynamics().getRewardBounds(), 
                           evalData['masterAgent'].getForwardDynamics().getRewardBounds())
        assert np.allclose(simData['masterAgent'].getForwardDynamics().getActionBounds(),
                            evalData['masterAgent'].getForwardDynamics().getActionBounds())
        assert np.allclose(simData['masterAgent'].getForwardDynamics().getStateBounds(),
                            evalData['masterAgent'].getForwardDynamics().getStateBounds())
        
    def test_MBRL_keras_cannon_2D_save_and_load_FD_net_params(self):
        filename = "tests/settings/cannonGame/MBRL/FixedSTD_Tensorflow-v2.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings["bootstrap_samples"] = 1000
        settings['rounds'] = 1
        settings["epochs"] = 2
        settings['eval_epochs'] = 2
        settings["plotting_update_freq_num_rounds"] = 2
        settings["saving_update_freq_num_rounds"] =  2
        settings["state_normalization"] = "variance"
        settings["return_model"] = True
        settings["num_available_threads"] = 2
        settings["learning_backend"] = "theano"
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        settings['visualize_expected_value'] = False
        evalData = modelEvaluation(filename, runLastModel=False, settings=settings, render=False)
        
        for i in range(len(simData['masterAgent'].getForwardDynamics().getNetworkParameters())): ## For each network
            for j in range(len(simData['masterAgent'].getForwardDynamics().getNetworkParameters()[i])): ## For each layer
                assert np.allclose(simData['masterAgent'].getForwardDynamics().getNetworkParameters()[i][j], 
                                   evalData['masterAgent'].getForwardDynamics().getNetworkParameters()[i][j],
                                   rtol=1.e-5, atol=1.e-8, equal_nan=True)

if __name__ == '__main__':
    pytest.main([__file__])