
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


### Get latest ubuntu image with cuda/nvidia drivers
```
singularity pull shub://ucr-singularity/cuda-9.0-base:latest
```
or
```
sudo singularity build --writable ubuntu_learning.img shub://ucr-singularity/cuda-9.0-base:latest
```

You should now have a usable singularity image

Now lets do some things with it.

```
sudo singularity shell --writable ubuntu.img
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

docker run -e TERRAINRL_PATH=/opt/playground/TerrainRL/ -e RLSIMENV_PATH=/opt/RLSimulationEnvironments --mount type=bind,src=/home/gberseth/playground/RL-Framework,dst=/opt/RL-Framework --mount type=bind,src=/home/gberseth/playground/TerrainRLSim,dst=/opt/TerrainRLSim --mount type=bind,src=/home/gberseth/playground/RLSimulationEnvironments,dst=/opt/RLSimulationEnvironments --rm  -it us.gcr.io/glen-rl-framework/glen:latest bash -c "pushd /opt/RL-Framework; python3 trainModel.py --config=tests/settings/particleSim/TRPO/FixedSTD_Tensorflow_KERAS.json --plot=false --shouldRender=false"


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


gcloud compute zones list

### Set default compute zone
gcloud config set compute/zone us-west1-c

### Create the private cluster
gcloud beta container clusters create private-cluster \
    --enable-private-nodes \
    --master-ipv4-cidr 172.16.0.16/26 \
    --enable-ip-alias \
    --preemptible \
    --enable-private-nodes \
    --num-nodes 2 --machine-type n1-standard-2 \
    --enable-autoscaling --min-nodes=0 --max-nodes=3 \
    --create-subnetwork ""

### List availible networks
gcloud compute networks subnets list --network default

### Get more info about the cluster network to check network connnectivity
gcloud compute networks subnets describe [SUBNET_NAME] --region us-west1

### Create the HEAD node
gcloud compute instances create cluster-head --scopes 'https://www.googleapis.com/auth/cloud-platform'

### Get HEAD node network info
gcloud compute instances describe cluster-head | grep natIP

### Authorize the network on the cluster to allow the mast node connections
gcloud container clusters update private-cluster \
    --enable-master-authorized-networks \
    --master-authorized-networks [MY_EXTERNAL_RANGE]


### SSH into head node to install kubernetes
gcloud compute ssh cluster-head

sudo apt-get install kubectl

### Get credentials to access the cluster from the head node
gcloud container clusters get-credentials private-cluster --zone us-west1-c


### Verify the nodes can be accessed
kubectl get nodes --output yaml | grep -A4 addresses
or
kubectl get nodes --output wide

### Delete the cluster
gcloud container clusters delete private-cluster --zone us-west1-c
### Delete head node
gcloud container instances delete cluster-head --zone us-west1-c

### Pushing docker containers to registry

https://cloud.google.com/kubernetes-engine/docs/tutorials/hello-app


### Kubernetes Commands

Run a job
```
kubectl apply -f test.yaml
```
Where test.yaml is a manifest file that dictates most of the job details.

Describe the state of a current joc
```
kubectl describe job example-job2
```

List the current pods
```
kubectl get pods -a
```

Get the log from a "pod"
```
kubectl logs example-job2-45cs9
```

### Installing fuse in the docker container

https://github.com/GoogleCloudPlatform/gcsfuse/blob/master/docs/installing.md


### Mounting shared storage in VM
```
mkdir /Cluster_Bucket
sudo chmod a+w /Cluster_Bucket/
```
Need to get /etc/fuse.conf, uncomment user_allow_other
```
gcsfuse --dir-mode "777" -o allow_other rl-framework-cluster-bucket /Cluster_Bucket/
```

```
export GOOGLE_APPLICATION_CREDENTIALS="/home/gberseth/Glen-RL-Framework-19b1aadaca75.json" gcsfuse --dir-mode "777" -o allow_other rl-framework-cluster-bucket /mnt/data
```