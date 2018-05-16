#!/bin/bash
# do not run this with sudo 
# Maybe just run the first one with sudo

apt-get -y install python3-dev python3-pip
# sudo apt-get -y install python3-tk

pip3 install numpy
pip3 install theano==1.0.1
# pip install matplotlib
pip3 install matplotlib
pip3 install Lasagne==0.1
### One of these should work... Curse you Lasagne...
cp patches/pool.py ~/.local/lib/python3.5/site-packages/lasagne/layers/
cp patches/pool.py ~/.local/lib/python3.6/site-packages/lasagne/layers/
cp patches/pool.py ~/.local/lib/python2.7/site-packages/lasagne/layers/
pip3 install dill
pip3 install pathos
pip3 install pyOpenGL
# pip3 install --user keras==2.1.5
### I have a custom version of keras now...
pip3 install git+https://github.com/Neo-X/keras.git
pip3 install tensorflow
## Does not work in Python3
# pip3 install pyODE
pip3 install h5py
pip3 install python3-tk


