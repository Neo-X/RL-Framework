# from nose.tools import timed
from numpy.testing import assert_allclose
import numpy as np
import pytest
import warnings
from model.ModelUtil import *
from trainModel import trainModelParallel
from trainForwardDynamics import trainForwardDynamics
import sys

### These don't work anymore...
class TestFDModel(object):

    # @pytest.mark.timeout(600)
    def test_gan_keras(self):
        """
        Test that CACLA can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/particleSim/CACLA/GAN_10D.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['rounds'] = 2
        settings['save_experience_memory'] = 'continual'
        settings["plotting_update_freq_num_rounds"] = 1
        settings["saving_update_freq_num_rounds"] =  1
        ### This is to generate the data from the simulation to train on
        simData = trainModelParallel((filename, settings))
        ### Now train fd model
        settings['rounds'] = 200
        learnData = trainForwardDynamics(settings)
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        ### I am not sure what good values should be for these metrics...
        assert np.mean(learnData['mean_forward_dynamics_loss'][-5:]) < 1.0
        ## assert np.mean(learnData['mean_forward_dynamics_reward_loss'][-5:]) < 1.0
    
    
    # @pytest.mark.timeout(600)
    def test_dense_l2_keras(self):
        """
        Test that CACLA can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/particleSim/CACLA/On_Policy_Tensorflow_FD.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 2
        settings['save_experience_memory'] = 'continual'
        settings["plotting_update_freq_num_rounds"] = 1
        settings["saving_update_freq_num_rounds"] =  1
        ### This is to generate the data from the simulation to learn from
        simData = trainModelParallel((filename, settings))
        ### Now train fd model
        settings['rounds'] = 200
        learnData = trainForwardDynamics(settings)
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(learnData['mean_forward_dynamics_loss'][-5:]) < 0.25
        assert np.mean(learnData['mean_forward_dynamics_reward_loss'][-5:]) < 1.0
        
        # @pytest.mark.timeout(600)
    def test_dense_l2_ensamble_keras(self):
        """
        Test that CACLA can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/particleSim/CACLA/On_Policy_Tensorflow_FD.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['save_experience_memory'] = 'continual'
        settings["plotting_update_freq_num_rounds"] = 1
        settings["saving_update_freq_num_rounds"] =  1
        settings['rounds'] = 2
        ### This is to generate the data from the simulation to learn from
        simData = trainModelParallel((filename, settings))
        ### Now train fd model
        settings['rounds'] = 200
        learnData = trainForwardDynamics(settings)
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(learnData['mean_forward_dynamics_loss'][-5:]) < 0.25
        assert np.mean(learnData['mean_forward_dynamics_reward_loss'][-5:]) < 1.0
    
if __name__ == '__main__':
    pytest.main([__file__])