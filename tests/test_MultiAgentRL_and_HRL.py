from numpy.testing import assert_allclose
import numpy as np
import pytest
import warnings
from trainModel import trainModelParallel
from trainModel import trainModelParallel as trainModelParallel  
import json
import sys


class TestMRLAndHRL(object):

    # @pytest.mark.timeout(600)
    def test_cacla_multiAgentNAv(self):
        """
        Test that CACLA can still learn a good policy on 2d particle sim
        """
        filename = "tests/settings/navGameMultiAgent/CACLA/On_Policy-v2.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        # settings['rounds'] = 2
        # settings['rollouts'] = 4
        settings['rounds'] = 50
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -1.5
    
    # @pytest.mark.timeout(600)
    def test_trpo_HRL_HLC_navgame2D(self):
        """
            Test that PPO can still learn a good policy on 2d gapgame sim
        """
        filename = "tests/settings/navgame2D/TRPO/HRL_Learning_32x32.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        # settings['rounds'] = 2
        settings['rounds'] = 50
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > 0.65
    
    # @pytest.mark.timeout(600)   
    def test_ppo_chaseGame2D(self):
        """
            Test that PPO can still learn a good policy on 2d gapgame sim
        """
        filename = "tests/settings/ChaseGame/PPO/On_Policy_Tensorflow-v2.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        # settings['rounds'] = 2
        settings['rounds'] = 100
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -1.0
        
    # @pytest.mark.timeout(600)   
    def test_multiagentRL_ddpg_chaseGame2D(self):
        """
            Test that PPO can still learn a good policy on 2d gapgame sim
        """
        filename = "tests/settings/ChaseGame/DDPG/Tensorflow-v2.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        # settings['rounds'] = 2
        # settings['rounds'] = 100
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -1.0
        
    # @pytest.mark.timeout(600)   
    def test_multiagentRL_maddpg_chaseGame2D(self):
        """
            Test that PPO can still learn a good policy on 2d gapgame sim
        """
        filename = "tests/settings/ChaseGame/MADDPG/Tensorflow-v2.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        # settings['rounds'] = 2
        # settings['rounds'] = 100
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -1.0
        
    # @pytest.mark.timeout(600)   
    def test_multiagentRL_madTRPO_navgame2D(self):
        """
            Test that PPO can still learn a good policy on 2d gapgame sim
        """
        filename = "tests/settings/navgame2D/MADDPG/HRL_TRPO_Tensorflow_NoViz_HLC-v4.json"
        file = open(filename)
        settings = json.load(file)
        file.close()
        this_function_name = sys._getframe().f_code.co_name
        settings['data_folder'] = settings['data_folder'] + '/' + this_function_name
        settings['visualize_learning'] = False
        settings['shouldRender'] = False
        settings['print_level'] = 'testing_sim'
        # settings['rounds'] = 2
        # settings['rounds'] = 100
        simData = trainModelParallel((filename, settings))
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(simData['mean_reward'][-5:]) > -1.0
            

if __name__ == '__main__':
    pytest.main([__file__])