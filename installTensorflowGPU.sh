#!/bin/bash

### Sets up an environment for training tensoflow on the GPU
### Make sure you have cuba 8.0 and cudnn 6 installed already.

## install virtual env
sudo apt-get install python-virtualenv
## Create virtual Env
virtualenv --system-site-packages -p python3 ~/tensorflow
## Start env
source ~/tensorflow/bin/activate
## install tensorflow GPU for Nvidia cuda 8.0 support
pip3 install --upgrade tensorflow-gpu==1.4
pip3 install --upgrade nvidia-ml-py3
## leave virtual env
deactivate
