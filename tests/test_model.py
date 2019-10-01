from numpy.testing import assert_allclose
import numpy as np
# from nose.tools import timed
import warnings
from model.ModelUtil import *
import pytest
import sys

class TestModel(object):

    def test_action_normalization(self):
        d = 10
        a_ = np.random.uniform(size=d)
        bounds = np.array([[np.min(a_)] * d, [np.max(a_)] * d])
        

        a_n = scale_action(norm_action(a_, bounds), bounds)
        # print (a_n)
        assert np.allclose(a_n, a_)
        # assert a_n == a_

    def test_random_normal(self):
        d = 10
        samples_ = 50000
        means = np.random.rand(1,d,d)
        stds = np.random.uniform(size=(1,d,d))+0.01
        for i in range(means.shape[1]):
            rand_ = []
            for j in range(samples_):
                rand_.append(randomExporationSTD([means[0][i]], [stds[0][i]])[0])
                # print ("garbage output, ", i, " ", j)
            rand_ = np.array(rand_)
            assert rand_.shape == (samples_, d)
            assert np.mean(rand_, axis=0).shape == means[0][i].shape
            assert np.all(np.less_equal(np.abs(np.mean(rand_, axis=0) - means[0][i]), 0.015))
            assert np.all(np.less_equal(np.abs(np.std(rand_, axis=0) - stds[0][i]), 0.015))
    
    def test_random_normal2(self):
        d = 10
        samples_ = 50000
        means = np.random.rand(1,d,d)
        stds = np.random.uniform(size=(1,d,d))+0.01
        for i in range(means.shape[1]):
            rand_ = []
            for j in range(samples_):
                rand_.append(randomExporationSTD([means[0][i]], [stds[0][i]])[0])
                # print ("garbage output, ", i, " ", j)
            rand_ = np.array(rand_)
            assert rand_.shape == (samples_, d)
            assert np.mean(rand_, axis=0).shape == means[0][i].shape
            assert np.all(np.less_equal(np.abs(np.mean(rand_, axis=0) - means[0][i]), 0.015))
            assert np.all(np.less_equal(np.abs(np.std(rand_, axis=0) - stds[0][i]), 0.015))
            
if __name__ == '__main__':
    nose.main([__file__])