# from nose.tools import timed
from numpy.testing import assert_allclose
import numpy as np
import pytest
import warnings
from trainModel import trainModelParallel
from ModelEvaluation import modelEvaluation
import json
import sys


class TestSaveAndLoadFD(object):

    
    # @pytest.mark.timeout(600)
    def test_MBRL_keras_cannon_2D_save_and_load_FD_scaling_params(self):
        ###    Check to see that the same network scaling value match after training with a reloaded model from a file.

        filename = "tests/settings/cannonGame/MBRL/FixedSTD_Tensorflow-v2.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings["bootstrap_samples"] = 1000
        settings['rounds'] = 1
        settings["epochs"] = 2
        settings['eval_epochs'] = 2
        settings["plotting_update_freq_num_rounds"] = 1
        settings["saving_update_freq_num_rounds"] =  1
        settings["state_normalization"] = "variance"
        settings["return_model"] = True
        settings["num_available_threads"] = 2
        settings["pretrain_critic"] = 0
        settings["simulation_timeout"] = 60
        settings["learning_backend"] = "tensorflow"
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
    
    ### This test doesn't work right now
    # @pytest.mark.timeout(600)   
    def test_MBRL_keras_cannon_2D_save_and_load_FD_net_params(self):
        """
            Check to see that the same network parameters match after training with a reloaded model from a file.
        """
        filename = "tests/settings/cannonGame/MBRL/FixedSTD_Tensorflow-v2.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings["bootstrap_samples"] = 1000
        settings['rounds'] = 1
        settings["epochs"] = 2
        settings['eval_epochs'] = 2
        settings["plotting_update_freq_num_rounds"] = 1
        settings["saving_update_freq_num_rounds"] =  1
        settings["state_normalization"] = "variance"
        settings["return_model"] = True
        settings["num_available_threads"] = 2
        settings["learning_backend"] = "theano"
        settings["pretrain_critic"] = 0
        settings["simulation_timeout"] = 60
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        settings['visualize_expected_value'] = False
        evalData = modelEvaluation(filename, runLastModel=False, settings=settings, render=False)
        
        ### This test does not work with tensorflow because the session for the old simData will be gone.
        for i in range(len(simData['masterAgent'].getForwardDynamics().getNetworkParameters())): ## For each network
            for j in range(len(simData['masterAgent'].getForwardDynamics().getNetworkParameters()[i])): ## For each layer
                # print ("layer: ", j , np.sum(np.array(simData['masterAgent'].getForwardDynamics().getNetworkParameters()[i][j]) - 
                #                    np.array(evalData['masterAgent'].getForwardDynamics().getNetworkParameters()[i][j])))
                assert np.allclose(simData['masterAgent'].getForwardDynamics().getNetworkParameters()[i][j], 
                                   evalData['masterAgent'].getForwardDynamics().getNetworkParameters()[i][j],
                                   rtol=1.e-4, atol=1.e-6, equal_nan=True)

if __name__ == '__main__':
    pytest.main([__file__])