# from nose.tools import timed
from numpy.testing import assert_allclose
import numpy as np
import pytest
import warnings
from trainModel import trainModelParallel
from ModelEvaluation import modelEvaluation
import json
import sys


class TestSaveAndLoad(object):
    
    # @pytest.mark.timeout(600)
    def test_ppo_keras_gapGame_2D_save_and_load_singleNet(self):
        ###    Test that PPO can still learn a good policy on 2d gapgame sim
        ###    If this does not crash, things are good..

        filename = "tests/settings/gapGame2D/PPO/SingleNet_FixedSTD_Tensorflow-v2.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings["bootstrap_samples"] = 1000
        settings['rounds'] = 4
        settings["epochs"] = 2
        settings['eval_epochs'] = 2
        settings["plotting_update_freq_num_rounds"] = 1
        settings["saving_update_freq_num_rounds"] =  1
        settings["state_normalization"] = "variance"
        settings["return_model"] = True
        settings["simulation_timeout"] = 60
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        settings['visualize_expected_value'] = False
        evalData = modelEvaluation(filename, runLastModel=False, settings=settings, render=False)
        print ("state bounds: ", simData['masterAgent'].getStateBounds(), evalData['masterAgent'].getStateBounds())
               
        assert np.allclose(simData['masterAgent'].getRewardBounds(), evalData['masterAgent'].getRewardBounds())
        assert np.allclose(simData['masterAgent'].getActionBounds(), evalData['masterAgent'].getActionBounds())
        assert np.allclose(simData['masterAgent'].getStateBounds(), evalData['masterAgent'].getStateBounds())
        
        # @pytest.mark.timeout(600)
    def test_ppo_keras_gapGame_2D_save_and_load_network_singleNet(self):
        ###    Test that PPO can still learn a good policy on 2d gapgame sim
        ###    If this does not crash, things are good..

        filename = "tests/settings/gapGame2D/PPO/SingleNet_FixedSTD_Tensorflow-v2.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings["bootstrap_samples"] = 1000
        settings['rounds'] = 4
        settings["epochs"] = 2
        settings['eval_epochs'] = 2
        settings["plotting_update_freq_num_rounds"] = 1
        settings["saving_update_freq_num_rounds"] =  1
        settings["state_normalization"] = "variance"
        settings["return_model"] = True
        settings["simulation_timeout"] = 60
        settings["learning_backend"] = "theano"
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        settings['visualize_expected_value'] = False
        evalData = modelEvaluation(filename, runLastModel=False, settings=settings, render=False)
        print ("state bounds: ", simData['masterAgent'].getStateBounds(), evalData['masterAgent'].getStateBounds())
               
        ### This test does not work with tensorflow because the session for the old simData will be gone.
        for i in range(len(simData['masterAgent'].getPolicy().getNetworkParameters())): ## For each network
            for j in range(len(simData['masterAgent'].getPolicy().getNetworkParameters()[i])): ## For each layer
                # print ("layer: ", j , np.sum(np.array(simData['masterAgent'].getPolicy().getNetworkParameters()[i][j]) - 
                #                    np.array(evalData['masterAgent'].getPolicy().getNetworkParameters()[i][j])))
                assert np.allclose(simData['masterAgent'].getPolicy().getNetworkParameters()[i][j], 
                                   evalData['masterAgent'].getPolicy().getNetworkParameters()[i][j],
                                   rtol=1.e-4, atol=1.e-6, equal_nan=True)
    
    # @pytest.mark.timeout(600)   
    def test_cacla_tensorflow_gapGame_2D_save_and_load_singleNet(self):
        ###    Test that PPO can still learn a good policy on 2d gapgame sim
        ###    If this does not crash, things are good..

        filename = "tests/settings/gapGame2D/CACLA/Net_FixedSTD_Tensorflow.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings["bootstrap_samples"] = 1000
        settings['rounds'] = 4
        settings["epochs"] = 2
        settings['eval_epochs'] = 2
        settings["plotting_update_freq_num_rounds"] = 1
        settings["saving_update_freq_num_rounds"] =  1
        settings["state_normalization"] = "variance"
        settings["return_model"] = True
        settings["simulation_timeout"] = 60
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        settings['visualize_expected_value'] = False
        evalData = modelEvaluation(filename, runLastModel=False, settings=settings, render=False)
        print ("state bounds: ", simData['masterAgent'].getStateBounds(), evalData['masterAgent'].getStateBounds())
               
        assert np.allclose(simData['masterAgent'].getRewardBounds(), evalData['masterAgent'].getRewardBounds())
        assert np.allclose(simData['masterAgent'].getActionBounds(), evalData['masterAgent'].getActionBounds())
        assert np.allclose(simData['masterAgent'].getStateBounds(), evalData['masterAgent'].getStateBounds())
    
if __name__ == '__main__':
    pytest.main([__file__])