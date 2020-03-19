#!/bin/bash
## This script is designed to make it easier to start a number of simulation
## example:
## ./settings/hyperParamTuning/computeCanada/runGRFSimulations.sh 500 settings/terrainRLImitate/PPO/Flat_GRF_Tensorflow_NoPhase_30FPS.json


## declare an array variable
declare -a metaExps=(
 				"settings/hyperParamTuning/element/anneal_exploration.json" 
				"settings/hyperParamTuning/element/anneal_policy_std.json" 	
 				"settings/hyperParamTuning/element/critic_learning_rate.json"
#				"settings/hyperParamTuning/element/value_function_batch_size.json"
)

rounds=$1
simConfig=$2
extraOpts=$3
## now loop through the above array
for metaConfig in "${metaExps[@]}"
do
	# echo "$metaConfig"
	# or do whatever with individual element of the array
	# echo "$simConfigFile"
	command="SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRL SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments sbatch --time=20:00:00 --mem=32768M --cpus-per-task=24 ./settings/hyperParamTuning/computeCanada/test_run.sh 'singularity exec --cleanenv --home /home/gberseth/projects/def-vandepan/gberseth/playground/RL-Framework:/opt/RL-Framework /scratch/gberseth/playground/singularity/ubuntu_learning.img python3.6 tuneHyperParameters.py --config="$simConfig" -p 6 --rollouts=12 --on_policy=fast --save_experience_memory=continual --num_rounds="$rounds" --continue_training=false --saving_update_freq_num_rounds=1 --plot=false --meta_sim_samples=2 --meta_sim_threads=2 --tuning_threads=2 --plotting_update_freq_num_rounds=5 --metaConfig="$metaConfig" --email_log_data_periodically=true --shouldRender=false "$extraOpts"'"
	echo $command
	eval $command
done

### Toy Sim
# SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRL SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments sbatch --time=50:00:00 ./settings/hyperParamTuning/computeCanada/test_run.sh "singularity exec --cleanenv --home /home/gberseth/playground/RL-Framework/:/opt/RL-Framework /scratch/gberseth/playground/singularity/ubuntu_learning.simg python3.6 trainModel.py --config=settings/terrainRLImitate3D/Path_Folowing/Path_Following.json -p 8 --bootstrap_samples=1000 --rollouts=16 --max_epoch_length=64 --on_policy=fast --save_experience_memory=continual --num_rounds=250 --print_level=train --continue_training=false --saving_update_freq_num_rounds=1 --epochs=4 --eval_epochs=8 --plot=false --meta_sim_samples=2 --meta_sim_threads=2 --plotting_update_freq_num_rounds=5 --metaConfig=settings/hyperParamTuning/deepCrowds.json --additional_on_policy_training_updates=1 --email_log_data_periodically=true --shouldRender=false"

