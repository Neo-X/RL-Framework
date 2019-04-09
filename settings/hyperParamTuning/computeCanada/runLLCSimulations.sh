#!/bin/bash
## This script is designed to make it easier to start a number of simulation
## example:
## ./settings/hyperParamTuning/element/runSimulations.sh settings/terrainRLImitate3D/CACLA/Humanoid1_Run_Tensorflow.json 250

## declare an array variable
declare -a metaExps=(
	"settings/hyperParamTuning/deepCrowds.json" 
)

## declare an array variable
declare -a simExps=(
#	"settings/terrainRLImitate3D/CACLA/Flat_Tensorflow.json"
#	"settings/terrainRLImitate3D/CACLA/Flat_Tensorflow_MultiAgent_LLC.json"
#	"settings/terrainRLImitate3D/PPO/Flat_Tensorflow.json"
#	"settings/terrainRLImitate3D/PPO/Flat_Tensorflow_MultiAgent_LLC.json"
	"settings/terrainRLImitate3D/PPO/Humanoid1_Flat_Tensorflow_LLC.json"
	"settings/terrainRLImitate3D/PPO/Humanoid1_Flat_Tensorflow_MultiAgent_LLC.json"
	"settings/terrainRLImitate3D/TRPO/Humanoid1_Flat_Tensorflow_LLC.json"
	"settings/terrainRLImitate3D/TRPO/Humanoid1_Flat_Tensorflow_MultiAgent_LLC.json"
)

rounds=$1
opts=$2

### For each sim sonfig
for metaExp in "${metaExps[@]}"
do
	
	## now loop through the above array
	for simConfig in "${simExps[@]}"
	do
		# echo "$metaConfig"
		# or do whatever with individual element of the array
		# echo "$simConfigFile"
		command="SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRL SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments sbatch --time=50:00:00 --mem=32768M --cpus-per-task=16 ./settings/hyperParamTuning/computeCanada/test_run.sh 'singularity exec --cleanenv --home /home/gberseth/playground/RL-Framework/:/opt/RL-Framework /scratch/gberseth/playground/singularity/ubuntu_learning.img python3.6 trainMetaModel.py --config="$simConfig" -p 8 --on_policy=fast --save_experience_memory=continual --num_rounds="$rounds" --continue_training=last --saving_update_freq_num_rounds=1 --plot=false --meta_sim_samples=2 --meta_sim_threads=2 --plotting_update_freq_num_rounds=5 --metaConfig="$metaExp" --email_log_data_periodically=true --shouldRender=false "$opts"'"
		echo $command
		eval $command
	done
done

### Toy Sim
# SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRL SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments sbatch --time=50:00:00 ./settings/hyperParamTuning/computeCanada/test_run.sh "singularity exec --cleanenv --home /home/gberseth/playground/RL-Framework/:/opt/RL-Framework /scratch/gberseth/playground/singularity/ubuntu_learning.simg python3.6 trainModel.py --config=settings/terrainRLImitate3D/Path_Folowing/Path_Following.json -p 8 --bootstrap_samples=1000 --rollouts=16 --max_epoch_length=64 --on_policy=fast --save_experience_memory=continual --num_rounds=250 --print_level=train --continue_training=false --saving_update_freq_num_rounds=1 --epochs=4 --eval_epochs=8 --plot=false --meta_sim_samples=2 --meta_sim_threads=2 --plotting_update_freq_num_rounds=5 --metaConfig=settings/hyperParamTuning/deepCrowds.json --additional_on-poli_trianing_updates=1 --email_log_data_periodically=true --shouldRender=false"

