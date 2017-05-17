

"""
    This file is very similar to ModelUtil in that is contains a collection of misc helper
    functions. However, the methods in this file tend to need Theano as a dependency.

"""
import theano
from theano import tensor as T
from lasagne.layers import get_all_params
import numpy as np
import lasagne
import sys
import copy
sys.path.append('../')
from model.ModelUtil import *

def kl(mean0, std0, mean1, std1, d):
    return T.log(std1 / std0).sum(axis=1) + ((T.square(std0) + T.square(mean0 - mean1)) / (2.0 * T.square(std1))).sum(axis=1) - 0.5 * d

def change_penalty(network1, network2):
    """
    The networks should be the same shape and design
    return ||network1 - network2||_2
    """
    return sum(T.sum((x1-x2)**2) for x1,x2 in zip(get_all_params(network1), get_all_params(network2)))
