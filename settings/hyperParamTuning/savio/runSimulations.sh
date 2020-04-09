#!/bin/bash
## This script is designed to make it easier to start a number of simulation
## example:
## ./settings/hyperParamTuning/element/runSimulations.sh settings/terrainRLImitate3D/CACLA/Humanoid1_Run_Tensorflow.json 250


## declare an array variable
declare -a metaExps=(
#	"settings/terrainRLMultiChar/HLC/CACLA/LargeBlocks_Multi_Char_On_Policy.json"
#	"settings/terrainRLMultiChar/HLC/CACLA/Dynamic_Obstacles.json"
#	"settings/terrainRLMultiChar/HLC/CACLA/Dynamic_Obstacles_NEWLLC.json"
#	"settings/terrainRLImitate3D/Path_Folowing/Dynamic_Obstacles.json"
#	"settings/terrainRLMultiChar/HLC/CACLA/Dynamic_Obstacles_Tensorflow_NEWLLC.json"
#	"settings/terrainRLMultiChar/HLC/PPO/Dynamic_Obstacles_Tensorflow_NEWLLC.json"
#	"settings/terrainRLMultiChar/HLC/PPO/Dynamic_Obstacles_Tensorflow_NEWLLC_v1.json"
#	"settings/terrainRLMultiChar/HLC/TRPO/Dynamic_Obstacles_Tensorflow_NEWLLC_v1.json"
	"settings/terrainRLMultiChar/HLC/CACLA/Dynamic_Obstacles_SimpleReward_Tensorflow_NEWLLC_v1.json"
	"settings/terrainRLMultiChar/HLC/PPO/Dynamic_Obstacles_SimpleReward_Tensorflow_NEWLLC_v1.json"
	"settings/terrainRLMultiChar/HLC/TRPO/Dynamic_Obstacles_SimpleReward_Tensorflow_NEWLLC_v1.json"
#	"settings/terrainRLMultiChar/HLC/TRPO/Dynamic_Obstacles_Tensorflow_NEWLLC.json"
#	"settings/terrainRLImitate3D/Path_Folowing/LargeBlocks_OnPolicy-OLDLLC.json"
#	"settings/terrainRLImitate3D/Path_Folowing/LargeBlocks_OnPolicy-NEWLLC.json"
#	"settings/terrainRLImitate3D/Path_Folowing/PPO/LargeBlocks_Tensorflow_NEWLLC.json"
#	"settings/terrainRLImitate3D/Path_Folowing/PPO/LargeBlocks_Tensorflow_NEWLLC2.json"
#	"settings/terrainRLImitate3D/Path_Folowing/PPO/Path_Following_NEWLLC_Tensorflow.json"
#	"settings/terrainRLImitate3D/Path_Folowing/PPO/Dynamic_Obstacles_NEWLLC.json"
#
#	"settings/terrainRLImitate3D/Path_Folowing/TRPO/LargeBlocks_Tensorflow_NEWLLC.json"
#	"settings/terrainRLImitate3D/Path_Folowing/TRPO/Path_Following_NEWLLC_Tensorflow.json"
#	"settings/terrainRLImitate3D/Path_Folowing/TRPO/Dynamic_Obstacles_NEWLLC.json"
#	"settings/terrainRLImitate3D/Path_Folowing/Path_Following.json"
#	"settings/terrainRLMultiChar/HLC/CACLA/Concentric_Circles.json"
#	"settings/terrainRLMultiChar/HLC/CACLA/Concentric_Circles_NEWLLC.json"
#	"settings/terrainRLMultiChar/HLC/CACLA/Concentric_Circles_NEWLLC_Tensorflow.json"
#	"settings/terrainRLMultiChar/HLC/PPO/Concentric_Circles_NEWLLC_Tensorflow.json"
#	"settings/terrainRLMultiChar/HLC/CACLA/Concentric_Circles_5_NEWLLC_Tensorflow.json"
#	"settings/terrainRLMultiChar/HLC/PPO/Concentric_Circles_5_NEWLLC_Tensorflow.json"
#	"settings/terrainRLMultiChar/HLC/CACLA/Concentric_Circles.json"
#	"settings/terrainRLMultiChar/HLC/CACLA/Concentric_Circles_NEWLLC.json"
#	"settings/terrainRLMultiChar/HLC/CACLA/Concentric_Circles_NEWLLC_Tensorflow_v1.json"
#	"settings/terrainRLMultiChar/HLC/CACLA/Concentric_Circles_5_NEWLLC_Tensorflow_v1.json"
#	"settings/terrainRLMultiChar/HLC/PPO/Concentric_Circles_NEWLLC_Tensorflow_v1.json"
#	"settings/terrainRLMultiChar/HLC/PPO/Concentric_Circles_5_NEWLLC_Tensorflow_v1.json"
#	"settings/terrainRLMultiChar/HLC/TRPO/Concentric_Circles_NEWLLC_Tensorflow_v1.json"
#	"settings/terrainRLMultiChar/HLC/TRPO/Concentric_Circles_5_NEWLLC_Tensorflow_v1.json"
### Different reawrd function
	"settings/terrainRLMultiChar/HLC/TRPO/Concentric_Circles_SimpleReward_NEWLLC_Tensorflow_v2.json"
	"settings/terrainRLMultiChar/HLC/TRPO/Concentric_Circles_SimpleReward_5_NEWLLC_Tensorflow_v2.json"
	"settings/terrainRLMultiChar/HLC/PPO/Concentric_Circles_SimpleReward_NEWLLC_Tensorflow_v2.json"
	"settings/terrainRLMultiChar/HLC/PPO/Concentric_Circles_SimpleReward_5_NEWLLC_Tensorflow_v2.json"
	"settings/terrainRLMultiChar/HLC/CACLA/Concentric_Circles_SimpleReward_NEWLLC_Tensorflow_v2.json"
	"settings/terrainRLMultiChar/HLC/CACLA/Concentric_Circles_SimpleReward_5_NEWLLC_Tensorflow_v2.json"
)

metaConfigFile=$1
rounds=$2
opts=$3
## now loop through the above array
for simConfig in "${metaExps[@]}"
do
	# echo "$metaConfig"
	# or do whatever with individual element of the array
	# echo "$simConfigFile"
	command="SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRL SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments sbatch --time=100:00:00 --mem=64768M --cpus-per-task=16 ./settings/hyperParamTuning/computeCanada/test_run.sh 'singularity exec --cleanenv --home /home/gberseth/projects/def-vandepan/gberseth/playground/RL-Framework:/opt/RL-Framework /scratch/gberseth/playground/singularity/ubuntu_learning.img python3.6 trainMetaModel.py --config="$simConfig" -p 8 --on_policy=fast --num_rounds="$rounds" --continue_training=false --saving_update_freq_num_rounds=1 --plot=false --meta_sim_samples=2 --meta_sim_threads=2 --plotting_update_freq_num_rounds=5 --metaConfig=settings/hyperParamTuning/deepCrowds.json --additional_on_policy_training_updates=1 --email_log_data_periodically=true --simulation_timeout=1800 --shouldRender=false --max_epoch_length=64 --rollouts=16 "$opts"'"
	echo $command
	eval $command
done

### Toy Sim
# SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRL SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments sbatch --time=50:00:00 ./settings/hyperParamTuning/computeCanada/test_run.sh "singularity exec --cleanenv --home /home/gberseth/playground/RL-Framework/:/opt/RL-Framework /scratch/gberseth/playground/singularity/ubuntu_learning.simg python3.6 trainModel.py --config=settings/terrainRLImitate3D/Path_Folowing/Path_Following.json -p 8 --bootstrap_samples=1000 --rollouts=16 --max_epoch_length=64 --on_policy=fast --save_experience_memory=continual --num_rounds=250 --print_level=train --continue_training=false --saving_update_freq_num_rounds=1 --epochs=4 --eval_epochs=8 --plot=false --meta_sim_samples=2 --meta_sim_threads=2 --plotting_update_freq_num_rounds=5 --metaConfig=settings/hyperParamTuning/deepCrowds.json --additional_on_policy_training_updates=1 --email_log_data_periodically=true --shouldRender=false"

