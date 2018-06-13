#!/bin/bash

### Sets up an environment for training tensoflow on the GPU

## install virtual env
sudo apt-get install python-virtualenv
## Create virtual Env
targetDirectory="~/tensorflow"
virtualenv --system-site-packages -p python3 $targetDirectory
## Start env
source ~/tensorflow/bin/activate
## install tensorflow GPU for Nvidia cuda 8.0 support
pip3 install --upgrade tensorflow-gpu==1.4
## leave virtual env
deactivate
