
# Intro

This folder contains some scripts and files to run simulations in a singularity image.

## Getting Started

Install newer version of singularity
```
sudo wget -O- http://neuro.debian.net/lists/xenial.us-ca.full | sudo tee /etc/apt/sources.list.d/neurodebian.sources.list
### Sometimes this takes a few tries
sudo apt-key adv --recv-keys --keyserver hkp://pool.sks-keyservers.net:80 0xA5D32F012649A5A9

sudo apt-get update

sudo apt-get install -y singularity-container
```


### Examples

SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRL SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments singularity shell --cleanenv -B /global/home/users/gberseth/playground/RL-Framework/:/opt/RL-Framework2 /global/scratch/gberseth/SingularityBuilding/ubuntu_learning.img -c "cd /opt/RL-Framework2; python3.6 trainMultiAgentModel.py --config=settings/navgame2D/MADDPG/HRL_Tensorflow_NoViz-v4.json -p 2 --num_rounds=10 --continue_training=false --saving_update_freq_num_rounds=1 --plot=false --meta_sim_samples=3 --meta_sim_threads=3 --tuning_threads=2 --plotting_update_freq_num_rounds=5 --metaConfig=settings/hyperParamTuning/element/learning_rate_ddpg.json --email_log_data_periodically=true --shouldRender=false --bootstrap_samples=0"

SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRL SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments singularity shell --cleanenv -B /global/home/users/gberseth/playground/RL-Framework:/opt/RL-Framework2 /global/scratch/gberseth/SingularityBuilding/ubuntu_learning.img -c "cd /opt/RL-Framework2; python3.6 trainMultiAgentModel.py --config=settings/navgame2D/MADDPG/HRL_TRPO_Tensorflow_NoViz_HLC-v4.json -p 2 --num_rounds=10 --continue_training=false --saving_update_freq_num_rounds=1 --plot=false --meta_sim_samples=3 --meta_sim_threads=3 --tuning_threads=2 --plotting_update_freq_num_rounds=5 --metaConfig=settings/hyperParamTuning/element/learning_rate_ddpg.json --email_log_data_periodically=true --shouldRender=false --bootstrap_samples=0"
