"""
    There are some issues getting tf model to save and load.
    This code helps fix those issues so that I can pickle the model
"""

import types
import tempfile
import keras.models
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten, Lambda


def make_keras_picklable():
    
    # setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
    # setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
    # setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)
    # setattr(Lambda, '__deepcopy__', lambda self, _: self)

    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__