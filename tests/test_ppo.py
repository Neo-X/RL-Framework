from nose.tools import timed
from numpy.testing import assert_allclose
import numpy as np
import pytest
import warnings
from trainModel import trainModelParallel
import json


class TestPPO(object):

    """
    # @pytest.mark.timeout(600)
    def test_ppo_particleNav_2D(self):
        ### Test that PPO can still learn a good policy on 2d particle sim
        filename = "tests/settings/particleSim/PPO/PPO.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 2
        settings['rollouts'] = 4
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -0.5
    
    # @pytest.mark.timeout(600)   
    def test_ppo_keras_particleNav_2D(self):
        ### Test that PPO can still learn a good policy on 2d particle sim
        filename = "tests/settings/particleSim/PPO/PPO_KERAS.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 2
        settings['rollouts'] = 4
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -0.5
    """ 
    
    # @pytest.mark.timeout(600)
    def test_ppo_keras_tensorflow_particleNav_2D(self):
        """
        Test that PPO can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/particleSim/PPO/PPO_KERAS_Tensorflow.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 2
        settings['rollouts'] = 4
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -0.5
    
    # @pytest.mark.timeout(600)    
    def test_ppo_keras_tensorflow_particleNav_2D_stochastic_policy(self):
        """
        Test that PPO can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/particleSim/PPO/PPO_KERAS_Tensorflow_stochasticPoli.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 2
        settings['rollouts'] = 4
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -0.5
        
    # @pytest.mark.timeout(600)    
    def test_ppo_keras_tensorflow_SingleNet_particleNav_2D(self):
        """
        Test that PPO can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/particleSim/PPO/PPO_KERAS_SingleNet_Tensorflow.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 2
        settings['rollouts'] = 4
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -0.5
    
    # @pytest.mark.timeout(600)    
    def test_ppo_keras_tensorflow_SingleNet_STD_Policy_particleNav_2D(self):
        """
        Test that PPO can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/particleSim/PPO/PPO_KERAS_SingleNet_Tensorflow_stochastic_policy.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 2
        settings['rollouts'] = 4
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -0.5
    """
    # @pytest.mark.timeout(600)   
    def test_ppo_keras_tensorflow_particleNav_10D(self):
        
        ## Test that PPO can still learn a good policy on 2d particle sim
        
        filename = "tests/settings/particleSim/PPO/PPO_KERAS_Tensorflow_10D.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        simData = trainModelParallel((filename, settings))
        settings['rounds'] = 2
        settings['rollouts'] = 4
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -0.5
    """
    # @pytest.mark.timeout(600)   
    def test_ppo_keras_tensorflow_stochastic_policy_particleNav_10D(self):
        """
        Test that PPO can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/particleSim/PPO/PPO_KERAS_Tensorflow_10D_StochasticPolicy.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 2
        settings['rollouts'] = 4
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -0.5
    
    # @pytest.mark.timeout(600)    
    def test_ppo_keras_tensorflow_SingleNet_particleNav_10D(self):
        """
        Test that PPO can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/particleSim/PPO/PPO_KERAS_SingleNet_Tensorflow_10D.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 2
        settings['rollouts'] = 4
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -0.5
    
    # @pytest.mark.timeout(600)    
    def test_ppo_keras_tensorflow_SingleNet_STD_Policy_particleNav_10D(self):
        """
        Test that PPO can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/particleSim/PPO/PPO_KERAS_SingleNet_Tensorflow_10D_Stochastic_Policy.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        settings['rounds'] = 2
        settings['rollouts'] = 4
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -0.5    

            

if __name__ == '__main__':
    pytest.main([__file__])