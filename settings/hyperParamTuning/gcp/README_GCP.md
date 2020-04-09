
# Intro

This folder contains some scripts and files to run simulations in a singularity image.

## start instance

```
gcloud compute instances start --project "glen-rl-framework" --zone "us-west1-c" "instance-default"
```

## Login to Instance

```
gcloud compute ssh "instance-default"
```

### Setup property security certificates to allow for ssh access

```
gcloud compute instances add-metadata [INSTANCE_NAME] --metadata enable-oslogin=TRUE
```

### Mount data folder

  1. Create a filestore (https://cloud.google.com/filestore/)
  ```
  gcloud filestore instances create nfs-server --zone=us-west1-c --tier=STANDARD --file-share=name="vol1",capacity=1TB --network=name="default"
  ```
  2. Install nfs library on instance
  ```
    sudo apt-get -y update
    sudo apt-get -y install nfs-common
    sudo mkdir /mnt/test
  ```
  3. Find the mount location, which can be foud in the instance info for the filestore
  4. Mount the file store
  ```
    sudo mount 10.182.188.58:/vol1 /mnt/test
    sudo chmod go+rw /mnt/test
  ```
  
## Getting Started


Install newer version of singularity
```
sudo wget -O- http://neuro.debian.net/lists/xenial.us-ca.full | sudo tee /etc/apt/sources.list.d/neurodebian.sources.list
### Sometimes this takes a few tries
sudo apt-key adv --recv-keys --keyserver hkp://pool.sks-keyservers.net:80 0xA5D32F012649A5A9

sudo apt-get update

sudo apt-get install -y singularity-container
```

Having issues installing some of the libraries in Ubuntu. 
In particular ```apt-get install freeglut3-dev libgles2-mesa-dev libglew1.6-dev``` is causing issues where old libraries are not getting overwitten.
I needed to specifically run the above command to install the libraries needed even though the proper versions of the dependancies were not updated.
 
 
## Setup sim environments

export RLSIMENV_PATH=/opt/RLSimulationEnvironments
export TERRAINRL_PATH=/opt/TerrainRL

## Example commands

- Convert "sandbox" into fixed image
```
sudo singularity build new-squashfs deepcrowds
sudo singularity build --sandbox ubuntu_learning ubuntu_learning.simg
```

- Run interactive shell in writable container
```
sudo singularity shell --writable ubuntu_learning
sudo singularity shell --writable deepcrowds
```

## Docker

### Push docker update

docker commit d0fb7d3cea01 glen:latest; docker tag glen:latest us.gcr.io/glen-rl-framework/glen:latest; docker push us.gcr.io/glen-rl-framework/glen:latest

### Run example

docker run -e TERRAINRL_PATH=/opt/playground/TerrainRL/ -e RLSIMENV_PATH=/opt/RLSimulationEnvironments --mount type=bind,src=/nfs/kun1/users/gberseth/playground/RL-Framework,dst=/opt/RL-Framework --mount type=bind,src=/nfs/kun1/users/gberseth/playground/TerrainRLSim,dst=/opt/TerrainRLSim --mount type=bind,src=/nfs/kun1/users/gberseth/playground/RLSimulationEnvironments,dst=/opt/RLSimulationEnvironments --rm  -it us.gcr.io/glen-rl-framework/glen:latest bash -c "pushd /opt/RL-Framework; python3 trainModel.py --config=tests/settings/particleSim/TRPO/FixedSTD_Tensorflow_KERAS.json --plot=false --shouldRender=false"

docker pull us.gcr.io/glen-rl-framework/glen:latest; docker run -e TERRAINRL_PATH=/opt/playground/TerrainRL/ -e RLSIMENV_PATH=/opt/RLSimulationEnvironments --mount type=bind,src=/nfs/kun1/users/gberseth/playground/RL-Framework,dst=/opt/RL-Framework --mount type=bind,src=/nfs/kun1/users/gberseth/playground/TerrainRLSim,dst=/opt/TerrainRLSim --mount type=bind,src=/nfs/kun1/users/gberseth/playground/RLSimulationEnvironments,dst=/opt/RLSimulationEnvironments --rm  -it us.gcr.io/glen-rl-framework/glen:latest bash -c "pushd /opt/RL-Framework; python3 tuneHyperParameters.py --config=settings/navgame2D/MADDPG/HRL_TRPO_Tensorflow_NoViz_HLC-v4.json --plot=false --shouldRender=false --metaConfig=settings/hyperParamTuning/element/state_normalization.json --meta_sim_samples=3 --meta_sim_threads=3 --tuning_threads=3 -p 2 --bootstrap_samples=10000 --num_rounds=100"


### Example learning commands Singularity



- Start a training simulation for training a multichar model
```
SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRL SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments singularity exec --cleanenv --home /mnt/test/playground/RL-Framework/:/opt/RL-Framework /mnt/test/playground/SingularityBuilding/ubuntu_learning.img python3.6 trainModel.py --config=settings/terrainRLMultiChar/HLC/PPO/ScenarioSpace_WithObs_SimpleReward_NEWLLC_Tensorflow_v0.json -p 16 --bootstrap_samples=10000 --rollouts=16 --on_policy=fast --save_experience_memory=continual --num_rounds=2500 --continue_training=last --saving_update_freq_num_rounds=1 --plot=false --meta_sim_samples=2 --meta_sim_threads=2 --plotting_update_freq_num_rounds=1 --metaConfig=settings/hyperParamTuning/deepCrowds.json 
```
- Train LLC
```
SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRL SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments singularity exec --cleanenv --home /mnt/test/playground/RL-Framework/:/opt/RL-Framework /mnt/test/playground/SingularityBuilding/ubuntu_learning.img python3.6 trainModel.py --config=settings/terrainRLImitate3D/PPO/Humanoid_Flat_Tensorflow_MultiAgent_WithObs_LLC.json -p 16 --bootstrap_samples=10000 --rollouts=16 --on_policy=fast --save_experience_memory=continual --num_rounds=2500 --continue_training=last --saving_update_freq_num_rounds=1 --plot=false --meta_sim_samples=2 --meta_sim_threads=2 --plotting_update_freq_num_rounds=1 --metaConfig=settings/hyperParamTuning/deepCrowds.json

```

## Setting up a private cluster on GCP Kubernetes

https://www.bogotobogo.com/DevOps/Docker/Docker-setting-up-private-cluster-on-GCP-Kubernetes.php

#!/bin/bash


