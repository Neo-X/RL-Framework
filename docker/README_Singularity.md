
# Intro

This folder contains some scripts and files to run simulations in a singularity image.

## Getting Started

```
### Get latest ubuntu image with cuda/nvidia drivers

sudo wget -O- http://neuro.debian.net/lists/xenial.us-ca.full | sudo tee /etc/apt/sources.list.d/neurodebian.sources.list
sudo apt-key adv --recv-keys --keyserver hkp://pool.sks-keyservers.net:80 0xA5D32F012649A5A9

sudo apt-get update

sudo apt-get install -y singularity-container

singularity pull shub://ucr-singularity/cuda-9.0-base:latest
or
sudo singularity build --writable ubuntu_learning.img shub://ucr-singularity/cuda-9.0-base:latest
```

You should now have a usable singularity image

Now lets do somethings with it.

```
sudo singularity shell --writable ubuntu.img
```

