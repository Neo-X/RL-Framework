#!/bin/bash
# do not run this with sudo 
# Maybe just run the first one with sudo

apt-get -y install python3-dev python3-pip

pip3 install --user numpy
pip3 install --user theano==1.0.1
# pip install matplotlib
pip3 install --user matplotlib
pip3 install --user --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
### One of these should work... Curse you Lasagne...
cp patches/pool.py ~/.local/lib/python3.5/site-packages/lasagne/layers/
cp patches/pool.py ~/.local/lib/python3.6/site-packages/lasagne/layers/
cp patches/pool.py ~/.local/lib/python2.7/site-packages/lasagne/layers/
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

apt-get -y install python3-tk
