
"""
    This compares the layer sizes, parameter values and activations of networks.

"""

import sys
sys.path.append('../')
import theano
from theano import tensor as T
import numpy as np
import lasagne

from util.SimulationUtil import loadNetwork 

def compareNetworks(agent1, agent2):
    
    print ("blah")
    net1 = agent1.getModel()
    net2 = agent2.getModel()    
    print ("Comparing layer sizes:")
    net1_layers = lasagne.layers.get_all_layers(net1.getActorNetwork()) 
    for i in range(len(net1_layers)):
        print("Actor network layer ", i ," : ", net1_layers[i])
    
    
    
if __name__ == '__main__':
    
    
    net1 = loadNetwork(sys.argv[1])
    net2 = loadNetwork(sys.argv[2])
    compareNetworks(net1, net2)