from nose.tools import timed
from numpy.testing import assert_allclose
import numpy as np
import pytest
import warnings
from model.ModelUtil import *
from trainModel import trainModelParallel
from trainForwardDynamics import trainForwardDynamics
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
        settings['rounds'] = 5
        ### This is to generate the data from the simulation to train on
        simData = trainModelParallel((filename, settings))
        ### Now train fd model
        settings['rounds'] = 200
        settings['rounds'] = 5
        learnData = trainForwardDynamics(settings)
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(learnData['mean_forward_dynamics_loss'][-5:]) < 0.01
        assert np.mean(learnData['mean_forward_dynamics_reward_loss'][-5:]) < 0.1
    
    
    # @pytest.mark.timeout(600)
    def test_dense_l2_keras(self):
        """
        Test that CACLA can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/particleSim/CACLA/On_Policy_Tensorflow_FD.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 5
        ### This is to generate the data from the simulation to learn from
        simData = trainModelParallel((filename, settings))
        ### Now train fd model
        settings['rounds'] = 200
        settings['rounds'] = 5
        learnData = trainForwardDynamics(settings)
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(learnData['mean_forward_dynamics_loss'][-5:]) < 0.01
        assert np.mean(learnData['mean_forward_dynamics_reward_loss'][-5:]) < 0.1
        
        # @pytest.mark.timeout(600)
    def test_dense_l2_ensamble_keras(self):
        """
        Test that CACLA can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/particleSim/CACLA/On_Policy_Tensorflow_FD.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 5
        ### This is to generate the data from the simulation to learn from
        simData = trainModelParallel((filename, settings))
        ### Now train fd model
        settings['rounds'] = 200
        settings['rounds'] = 5
        learnData = trainForwardDynamics(settings)
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(learnData['mean_forward_dynamics_loss'][-5:]) < 0.01
        assert np.mean(learnData['mean_forward_dynamics_reward_loss'][-5:]) < 0.1
    
if __name__ == '__main__':
    pytest.main([__file__])