import pytest
from numpy.testing import assert_allclose
import numpy as np

import warnings
from model.ModelUtil import *

class TestModel(object):

    def test_action_normalization(self):
        d = 10
        a_ = np.random.uniform(size=d)
        bounds = np.array([[np.min(a_)] * d, [np.max(a_)] * d])
        

        a_n = scale_action(norm_action(a_, bounds), bounds)
        print (a_n)
        assert np.array_equal(a_n, a_)
        # assert a_n == a_


if __name__ == '__main__':
    pytest.main([__file__])