

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
# from model.ModelUtil import *

def kl(mean0, std0, mean1, std1, d):
    """
        The first districbution should be from a fixed distribution. 
        The second should be from the distribution that will change from the parameter update.
        Parameters
        ----------
        mean0: mean of fixed distribution
        std0: standard deviation of fixed distribution
        mean1: mean of moving distribution
        std1: standard deviation of moving distribution
        d: is the dimensionality of the action space
        
        Return(s)
        ----------
        Vector: Of kl_divergence for each sample/row in the input data
    """
    return T.log(std1 / std0).sum(axis=1) + ((T.square(std0) + T.square(mean0 - mean1)) / (2.0 * T.square(std1))).sum(axis=1) - 0.5 * d

def change_penalty(network1, network2):
    """
    The networks should be the same shape and design
    return ||network1 - network2||_2
    """
    return sum(T.sum((x1-x2)**2) for x1,x2 in zip(get_all_params(network1), get_all_params(network2)))

def get_params_flat(var_list):
    return [v.flatten() for v in var_list]

def flatgrad(loss, var_list):
    """
        Returns the gradient as a vector instead of alist of vectors
    """
    grads = T.grad(loss, var_list)
    return T.concatenate([g.flatten() for g in grads])

def setFromFlat(var_list, flat_grad):
    """
        Probably does not work...
    """
    theta = T.vector()
    start = 0
    updates = []
    for v in var_list:
        shape = v.shape
        size = T.prod(shape)
        updates.append((v, theta[start:start+size].reshape(shape)))
        start += size
    self.op = theano.function([theta],[], updates=updates,**FNOPTS)
    
def entropy(std):
    """
        Computes the entropy for a Guassian distribution given the std.
    """
    return 0.5 * T.mean(T.log(2 * np.pi * std ) + 1 )

def loglikelihood(a, mean0, std0, d):
    """
        d is the number of action dimensions
    """
    # exp[ -(a - mu)^2/(2*sigma^2) ] / sqrt(2*pi*sigma^2)
    return T.reshape(- 0.5 * (T.square(a - mean0) / std0).sum(axis=1) - 0.5 * T.log(2.0 * np.pi) * d - T.log(std0).sum(axis=1), newshape=(-1, 1))
    # return (- 0.5 * T.square((a - mean0) / std0).sum(axis=1) - 0.5 * T.log(2.0 * np.pi) * d - T.log(std0).sum(axis=1))


def likelihood(a, mean0, std0, d):
    return T.exp(loglikelihood(a, mean0, std0, d))

def zipsame(*seqs):
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])
    return zip(*seqs)