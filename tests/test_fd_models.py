from nose.tools import timed
from numpy.testing import assert_allclose
import numpy as np

import warnings
from model.ModelUtil import *
from trainModel import trainModelParallel
from trainForwardDynamics import trainForwardDynamics
### These don't work anymore...
class TestFDModel(object):

    @timed(600)
    def test_gan_lasagne(self):
        """
        Test that CACLA can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/particleSim/CACLA/SMBAE_GAN_10D.json"
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
    
    @timed(600)   
    def test_gan_GPU_lasagne(self):
        """
        Test that CACLA can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/particleSim/CACLA/SMBAE_GAN_10D.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 5
        settings['training_processor_type'] = 'cpu'
        ### This is to generate the data from the simulation to train on
        simData = trainModelParallel((filename, settings))
        ### Now train fd model
        settings['rounds'] = 200
        settings['rounds'] = 5
        settings['training_processor_type'] = 'cuda'
        learnData = trainForwardDynamics(settings)
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(learnData['mean_forward_dynamics_loss'][-5:]) < 0.01
        assert np.mean(learnData['mean_forward_dynamics_reward_loss'][-5:]) < 0.1
    
    @timed(600)
    def test_dense_l2_lasagne(self):
        """
        Test that CACLA can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/particleSim/CACLA/MBAE_10D.json"
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
    
    @timed(600)    
    def test_dense_l2_GPU_lasagne(self):
        """
        Test that CACLA can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/particleSim/CACLA/MBAE_10D_GPU.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 5
        settings['training_processor_type'] = 'cpu'
        ### This is to generate the data from the simulation to learn from
        simData = trainModelParallel((filename, settings))
        ### Now train fd model
        settings['rounds'] = 200
        settings['rounds'] = 5
        settings['training_processor_type'] = 'cuda'
        learnData = trainForwardDynamics(settings)
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(learnData['mean_forward_dynamics_loss'][-5:]) < 0.01
        assert np.mean(learnData['mean_forward_dynamics_reward_loss'][-5:]) < 0.1
        
if __name__ == '__main__':
    pytest.main([__file__])