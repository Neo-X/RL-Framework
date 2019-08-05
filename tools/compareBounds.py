
"""
    This compares the layer sizes, parameter values and activations of networks.
    Example:
    python3 tools/compareBounds.py RLSimulations/algorithm.TRPO_KERAS.TRPO_KERAS/CLEVRObjectHRL_NoVision_HLC_v0/MAD_1/model.DeepNNKerasAdaptive.DeepNNKerasAdaptive/agent1_bounds.hdf5 RLSimulations/algorithm.TRPO_KERAS.TRPO_KERAS/CLEVRObjectHRL_NoVision_LLP_v0/MAD_7/model.DeepNNKerasAdaptive.DeepNNKerasAdaptive/agent0_bounds.h5
"""

import sys
sys.path.append('../')
import theano
from theano import tensor as T
import numpy as np
import lasagne
import os

def loadBounds(fileName):
    import h5py
    data = {}
    hf = h5py.File(fileName,'r')
    data['_state_bounds'] = np.array(hf.get('_state_bounds'))
    data['_reward_bounds'] = np.array(hf.get('_reward_bounds'))
    data['_action_bounds'] = np.array(hf.get('_action_bounds'))
    # self._result_state_bounds = np.array(hf.get('_result_state_bounds'))
    hf.close()
    return data

def compareBounds(agent1, agent2):
    """
        Prints out information related to the size, shape and parameter values of the networks
    """
    print ("Compare state boundaries")
    print ("agent1: ", agent1['_state_bounds'])
    print ("agent2: ", agent2['_state_bounds'])
    print ("diff: ", agent1['_state_bounds'] - agent2['_state_bounds'])
    
    
if __name__ == '__main__':
    
    
    data = loadBounds(sys.argv[1])
    data1 = loadBounds(sys.argv[2])
    compareBounds(data, data1)