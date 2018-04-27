import pytest
from numpy.testing import assert_allclose
import numpy as np

import warnings
from model.ModelUtil import *
from ModelEvaluation import simEpoch
import json

class TestSimulation(object):
    
    def test_collect_tuples(self):
        settings_ = {}
        settings_['batch_size'] = 32
        settings_['learning_rate'] = 0.001
        settings_['discount_factor'] = 0.95
        settings_['discount_factor'] = 0.95
        settings_['rho'] = 0.1
        settings_['rms_epsilon'] = 0.0002
        settings_['steps_until_target_network_update'] = 100
        settings_['regularization_weight'] = 0.0001
        settings_['discrete_actions'] = [[0, 1], [1, 2], [2, 3]]
        settings_['reward_bounds'] = [[0.0], [1.0]]
        settings_['action_bounds'] = [[0.0, 1.0], [1.0, 2.0]]
        settings_['omega'] = 0.0
        settings_['max_epoch_length'] = 256
        
        filename = "tests/settings/gapGame2D/PPO/SingleNet_FixedSTD.json"
        file = open(filename)
        settings_ = json.load(file)
        file.close()
        settings_['visualize_learning'] = False
        settings_['shouldRender'] = False
        
                
        from actor.DoNothingActor import DoNothingActor
        from sim.DummyEnv import DummyEnv
        from algorithm.ModelDummy import ModelDummy
        agent = DoNothingActor(settings_=settings_, experience=None)
        env = DummyEnv(exp=None, settings=settings_)
        modelDummy = ModelDummy(model=None, n_in=11, n_out=7, 
                                state_bounds=None, action_bounds=None, 
                                reward_bound=None, settings_=settings_)
        
        out = simEpoch(actor=agent, exp=env, model=modelDummy, discount_factor=0.9, settings=settings_, anchors=None, action_space_continuous=True, 
                       print_data=False, p=0.0, validation=False, epoch=0, evaluation=False, _output_queue=None, bootstrapping=False, visualizeEvaluation=None,
             sampling=False, epsilon=0.2,
             worker_id=None)
            

if __name__ == '__main__':
    pytest.main([__file__])