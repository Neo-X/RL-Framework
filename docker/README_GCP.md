
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

- Start a training simulation for training a multichar model
```
SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRL SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments singularity exec --cleanenv --home /mnt/test/playground/RL-Framework/:/opt/RL-Framework /mnt/test/playground/SingularityBuilding/ubuntu_learning.img python3.6 trainModel.py --config=settings/terrainRLMultiChar/HLC/PPO/ScenarioSpace_WithObs_SimpleReward_NEWLLC_Tensorflow_v0.json -p 16 --bootstrap_samples=10000 --rollouts=16 --on_policy=fast --save_experience_memory=continual --num_rounds=10 --print_level=train --continue_training=last --saving_update_freq_num_rounds=1 --plot=false --meta_sim_samples=2 --meta_sim_threads=2 --plotting_update_freq_num_rounds=1 --metaConfig=settings/hyperParamTuning/deepCrowds.json --additional_on-poli_trianing_updates=2 
```

## Setting up a private cluster on GCP Kubernetes

https://www.bogotobogo.com/DevOps/Docker/Docker-setting-up-private-cluster-on-GCP-Kubernetes.php


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