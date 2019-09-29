
# Intro

This folder contains some scripts and files to run simulations in a singularity image.

## Getting Started

Install newer version of singularity
```
sudo wget -O- http://neuro.debian.net/lists/xenial.us-ca.full | sudo tee /etc/apt/sources.list.d/neurodebian.sources.list
### Sometimes this takes a few tries
sudo apt-key adv --recv-keys --keyserver hkp://pool.sks-keyservers.net:80 0xA5D32F012649A5A9

sudo apt-get update

sudo apt-get install -y singularity-container=2.4.2-4
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
- Bind a directory outside the container to one inside
```
singularity exec --bind ~/playground/RL-Framework/terrainRLSim/:/opt/RL-Framework/terrainRLSim deepcrowds ls /opt/RL-Framework/terrainRLSim
```
- Set environment variable
```
SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRL  singularity exec --cleanenv --bind ~/playground/RL-Framework/terrainRLSim/:/opt/RL-Framework/terrainRLSim deepcrowds ls /opt/RL-Framework/terrainRLSim
```

Run a simulation on the laptop

```
SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRLSim SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments singularity exec --cleanenv /media/gberseth/Cluster_Port/playground/SingularityBuilding/ubuntu_learning.img python3.6 trainModel.py --config=tests/settings/particleSim/PPO/PPO_KERAS_Tensorflow-v2_5D.json -p 2 --num_rounds=10 --continue_training=false --saving_update_freq_num_rounds=1 --plot=false --meta_sim_samples=3 --meta_sim_threads=3 --tuning_threads=2 --plotting_update_freq_num_rounds=5 --metaConfig=settings/hyperParamTuning/element/learning_rate_ddpg.json --email_log_data_periodically=true --shouldRender=false --bootstrap_samples=0
```
or

```
SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRLSim SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments singularity exec --cleanenv /media/gberseth/Cluster_Port/playground/SingularityBuilding/ubuntu_learning.img python3.6 trainMultiAgentModel.py --config=settings/navgame2D/MADDPG/HRL_TRPO_Tensorflow_NoViz_HLC-v4.json -p 2 --num_rounds=10 --continue_training=false --saving_update_freq_num_rounds=1 --plot=false --meta_sim_samples=3 --meta_sim_threads=3 --tuning_threads=2 --plotting_update_freq_num_rounds=5 --metaConfig=settings/hyperParamTuning/element/learning_rate_ddpg.json --email_log_data_periodically=true --shouldRender=false --bootstrap_samples=0
```

- Start a training simulation for training a multichar model
```
SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRL SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments singularity exec --cleanenv --home ~/playground/RL-Framework/:/opt/RL-Framework deepcrowds python3.6 trainModel.py --config=settings/terrainRLMultiChar/HLC/CACLA/LargeBlocks_Multi_Char_On_Policy.json -p 8 --bootstrap_samples=10000 --rollouts=16 --max_epoch_length=64 --on_policy=fast --save_experience_memory=continual --num_rounds=1000 --print_level=train --continue_training=false --saving_update_freq_num_rounds=1 --epochs=4 --eval_epochs=8 --plot=false --meta_sim_samples=2 --meta_sim_threads=2 --plotting_update_freq_num_rounds=1 --metaConfig=settings/hyperParamTuning/deepCrowds.json --additional_on_policy_training_updates=2 
```
- Version with velocity field input
```
SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRL SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments singularity exec --cleanenv --home ~/playground/RL-Framework/:/opt/RL-Framework deepcrowds python3.6 trainModel.py --config=settings/terrainRLMultiChar/HLC/CACLA/Dynamic_Obstacles_NEWLLC.json -p 8 --bootstrap_samples=10000 --rollouts=16 --max_epoch_length=128 --on_policy=fast --save_experience_memory=continual --num_rounds=1000 --print_level=train --continue_training=false --saving_update_freq_num_rounds=1 --epochs=4 --eval_epochs=8 --plot=false --meta_sim_samples=2 --meta_sim_threads=2 --plotting_update_freq_num_rounds=5 --metaConfig=settings/hyperParamTuning/deepCrowds.json --additional_on_policy_training_updates=2
```
-- training an LLC type controller (imitation)

```
SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRL SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments sbatch settings/hyperParamTuning/computeCanada/test_run.sh "singularity exec --cleanenv --home /home/gberseth/playground/RL-Framework/:/opt/RL-Framework /scratch/gberseth/playground/singularity/ubuntu_learning.simg python3.6 trainModel.py --config=settings/terrainRLImitate/PPO/Flat_Tensorflow_NoPhase.json -p 8 --bootstrap_samples=10000 --rollouts=16 --max_epoch_length=256 --on_policy=fast --save_experience_memory=continual --num_rounds=250 --print_level=train --continue_training=false --saving_update_freq_num_rounds=1 --epochs=4 --eval_epochs=32 --plot=false --meta_sim_samples=2 --meta_sim_threads=2 --plotting_update_freq_num_rounds=10 --metaConfig=settings/hyperParamTuning/deepCrowds.json --additional_on_policy_training_updates=1"
```

-- train MultiChar on deepcrowds

```
SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRL SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments singularity exec --cleanenv --home /home/gberseth/playground/RL-Framework/:/opt/RL-Framework /scratch/gberseth/playground/singularity/ubuntu_learning.simg python3.6 trainModel.py --config=settings/terrainRLMultiChar/HLC/CACLA/Dynamic_Obstacles_NEWLLC.json -p 8 --bootstrap_samples=1000 --rollouts=16 --max_epoch_length=64 --on_policy=fast --save_experience_memory=continual --num_rounds=10 --print_level=train --continue_training=last --saving_update_freq_num_rounds=1 --epochs=4 --eval_epochs=8 --plot=false --meta_sim_samples=2 --meta_sim_threads=2 --plotting_update_freq_num_rounds=5 --metaConfig=settings/hyperParamTuning/deepCrowds.json --additional_on_policy_training_updates=1 --email_log_data_periodically=true --shouldRender=false

SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRL SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments singularity exec --cleanenv --home /home/gberseth/playground/RL-Framework/:/opt/RL-Framework deepcrowds.img python3.6 trainModel.py --config=settings/terrainRLMultiChar/HLC/CACLA/Dynamic_Obstacles_NEWLLC.json -p 8 --bootstrap_samples=1000 --rollouts=16 --max_epoch_length=64 --on_policy=true --save_experience_memory=continual --num_rounds=10 --print_level=train --continue_training=last --saving_update_freq_num_rounds=1 --epochs=4 --eval_epochs=8 --plot=false --meta_sim_samples=2 --meta_sim_threads=2 --plotting_update_freq_num_rounds=5 --metaConfig=settings/hyperParamTuning/deepCrowds.json --additional_on_policy_training_updates=1 --email_log_data_periodically=true --shouldRender=false
```
on compute canada
```
"SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRL SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments sbatch --time=50:00:00 --mem=32768M --cpus-per-task=8 ./settings/hyperParamTuning/computeCanada/test_run.sh 'singularity exec --cleanenv --home /home/gberseth/playground/RL-Framework/:/opt/RL-Framework /scratch/gberseth/playground/singularity/ubuntu_learning.simg python3.6 trainModel.py --config="$simConfig" -p 8 --bootstrap_samples=1000 --rollouts=16 --max_epoch_length=64 --on_policy=fast --save_experience_memory=continual --num_rounds="$rounds" --print_level=train --continue_training=last --saving_update_freq_num_rounds=1 --epochs=4 --eval_epochs=8 --plot=false --meta_sim_samples=2 --meta_sim_threads=2 --plotting_update_freq_num_rounds=5 --metaConfig=settings/hyperParamTuning/deepCrowds.json --additional_on_policy_training_updates=1 --email_log_data_periodically=true --shouldRender=false'"
```


