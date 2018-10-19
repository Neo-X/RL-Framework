#!/bin/bash

sudo add-apt-repository ppa:jonathonf/python-3.6  # (only for Ubuntu 16.04 LTS)

## Then, run the following (this works out-of-the-box on 16.10 and 17.04):

user=""
# user="--user"
sudo apt update
sudo apt install -y python3.6
sudo apt install -y python3.6-dev
# sudo apt install python3.6-venv
wget https://bootstrap.pypa.io/get-pip.py
sudo python3.6 get-pip.py

### python3.6 and pip3.6 should work now.


# sudo apt-get -y install python3-dev python3-pip

pip3.6 install $user numpy
pip3.6 install $user theano==1.0.1
# pip install matplotlib
pip3.6 install $user matplotlib
pip3.6 install $user --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
### One of these should work... Curse you Lasagne...
cp ../patches/pool.py ~/.local/lib/python3.6/site-packages/lasagne/layers/
pip3.6 install $user dill
pip3.6 install $user pathos
pip3.6 install $user pyOpenGL
# pip3 install $user keras==2.1.5
### I have a custom version of keras now...
pip3.6 install $user git+https://github.com/Neo-X/keras.git
pip3.6 install $user dask toolz
pip3.6 install $user pytest
pip3.6 install $user tensorflow
## Does not work in Python3
# pip3 install pyODE
pip3.6 install $user h5py

sudo apt-get -y install python3-tk

#!/bin/bash
# do not run this with sudo 
# Maybe just run the first one with sudo
#### Only if you want to replace python3.5
# sudo ln -s /usr/bin/python3.6 /usr/local/bin/python3
# sudo ln -s /usr/local/bin/pip3.6 /usr/local/bin/pip3

