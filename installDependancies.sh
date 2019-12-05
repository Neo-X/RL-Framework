#!/bin/bash
# do not run this with sudo 
# Maybe just run the first one with sudo

# apt-get -y install python3-dev python3-pip

pip3 install --user numpy
pip3 install --user theano==1.0.1
# pip install matplotlib
pip3 install --user matplotlib
pip3 install --user dill
pip3 install --user pathos
pip3 install --user pyOpenGL
# pip3 install --user keras==2.1.5
### I have a custom version of keras now...
pip3 install --user git+https://github.com/Neo-X/keras.git
pip3 install --user tensorflow
pip3 install --user -U --no-deps keras-layer-normalization
## Does not work in Python3
# pip3 install pyODE
pip3 install --user h5py
pip3 install --user imageio

pip3 install --user pytest
pip3 install --user dask
pip3 install --user toolz
### For rendering network models to svg files from keras
pip3 install --user pydot
pip3 install --user pillow

### For rendering interactive metplotlib figures during training
# apt-get -y install python3-tk
