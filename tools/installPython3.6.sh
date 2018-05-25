#!/bin/bash

sudo add-apt-repository ppa:jonathonf/python-3.6  # (only for 16.04 LTS)

## Then, run the following (this works out-of-the-box on 16.10 and 17.04):

sudo apt update
sudo apt install python3.6
sudo apt install python3.6-dev
# sudo apt install python3.6-venv
wget https://bootstrap.pypa.io/get-pip.py
sudo python3.6 get-pip.py

### python3.6 and pip3.6 should work now.

#### Only if you want to replace python3.5
# sudo ln -s /usr/bin/python3.6 /usr/local/bin/python3
# sudo ln -s /usr/local/bin/pip /usr/local/bin/pip3

