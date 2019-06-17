#!/bin/bash
## This script is designed to make it easier to start a number of simulation
## example:
## ./settings/hyperParamTuning/element/runSimulations.sh settings/terrainRLImitate3D/CACLA/Humanoid1_Run_Tensorflow.json 250

## declare an array variable
declare -a metaExps=(
# 				"settings/hyperParamTuning/element/activation_type.json" 
# 				"settings/hyperParamTuning/element/additional_on_policy_training_updates.json"
#				"settings/hyperParamTuning/element/advantage_scaling.json" 
# 				"settings/hyperParamTuning/element/anneal_exploration.json" 
#				"settings/hyperParamTuning/element/anneal_learning_rate.json" 	
# 				"settings/hyperParamTuning/element/anneal_policy_std.json" 	
#				"settings/hyperParamTuning/element/batch_size.json"
# 				"settings/hyperParamTuning/element/CACLA_use_advantage.json"
# 				"settings/hyperParamTuning/element/CACLA_use_advantage_action_weighting.json"
#  				"settings/hyperParamTuning/element/clamp_actions_to_stay_inside_bounds.json" 
#  				"settings/hyperParamTuning/element/critic_learning_rate.json"
#				"settings/hyperParamTuning/element/critic_network_layer_sizes.json"
# 				"settings/hyperParamTuning/element/critic_updates_per_actor_update.json"
#  				"settings/hyperParamTuning/element/dont_use_td_learning.json"
#  				"settings/hyperParamTuning/element/dropout_p.json" 
#				"settings/hyperParamTuning/element/exploration_method.json"
#				"settings/hyperParamTuning/element/exploration_rate.json" 
# 				"settings/hyperParamTuning/computeCanada/mrl_exploration_rate.json"
#				"settings/hyperParamTuning/element/GAE_lambda.json"
# 				"settings/hyperParamTuning/element/initial_temperature.json" 
# 				"settings/hyperParamTuning/element/kl_divergence_threshold.json" 
#  				"settings/hyperParamTuning/element/last_policy_layer_activation_type.json"
#  				"settings/hyperParamTuning/element/learning_rate.json" 
  				"settings/hyperParamTuning/element/learning_rate_ddpg.json" 
# 				"settings/hyperParamTuning/element/normalize_advantage.json" 
#  				"settings/hyperParamTuning/element/num_on_policy_rollouts.json" 
#  				"settings/hyperParamTuning/element/optimizer.json" 
# 				"settings/hyperParamTuning/element/policy_activation_type.json"
# 				"settings/hyperParamTuning/element/policy_network_layer_sizes.json"
#				"settings/hyperParamTuning/computeCanada/policy_network_layer_sizes.json"
#				"settings/hyperParamTuning/computeCanada/policy_network_layer_sizes2.json"	
#  				"settings/hyperParamTuning/element/ppo_et_factor.json"
# 				"settings/hyperParamTuning/element/pretrain_critic.json" 
# 				"settings/hyperParamTuning/element/reset_on_fall.json" 
#				"settings/hyperParamTuning/element/state_normalization.json"
				"settings/hyperParamTuning/element/target_net_interp_weight.json"		
#  				"settings/hyperParamTuning/element/use_single_network.json"
#  				"settings/hyperParamTuning/element/use_stocastic_policy.json"
#  				"settings/hyperParamTuning/element/use_target_net_for_critic.json"
#				"settings/hyperParamTuning/element/value_function_batch_size.json"
)

## declare an array variable
declare -a simExps=(
	"settings/navgame2D/DDPG/HRL_DDPG_Tensorflow_NoViz_HLC-v4.json"
 	"settings/navgame2D/MADDPG/HRL_TRPO_Tensorflow_NoViz_HLC-v4.json"
)

rounds=$1
opts=$2

sbatchFlags="--account=fc_rail --partition=savio2 --mem=16384M --time=24:00:00 --job-name=test --output=%x-%j.out --cpus-per-task=8 --mail-user=gberseth@gmail.com --mail-type=BEGIN,END,FAIL,REQUEUE,ALL"

### For each sim sonfig
for metaExp in "${metaExps[@]}"
do
	
	## now loop through the above array
	for simConfig in "${simExps[@]}"
	do
		# echo "$metaConfig"
		# or do whatever with individual element of the array
		# echo "$simConfigFile"
		arg="python3.6 tuneHyperParameters.py --config=${simConfig} -p 2 --num_rounds=${rounds} --continue_training=false --saving_update_freq_num_rounds=1 --plot=false --meta_sim_samples=3 --meta_sim_threads=3 --tuning_threads=2 --plotting_update_freq_num_rounds=5 --metaConfig=${metaExp} --email_log_data_periodically=true --shouldRender=false --bootstrap_samples=0 ${opts}"
		# command=(sbatch --time=25:00:00 --mem=8536M --cpus-per-task=12 ./settings/hyperParamTuning/savio/test_run.sh singularity shell --cleanenv -B /global/home/users/gberseth/playground/RL-Framework:/opt/RL-Framework2 /global/scratch/gberseth/SingularityBuilding/ubuntu_learning.img -c "$arg")
		command="SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRLSim SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments sbatch --time=25:00:00 --mem=16000M --cpus-per-task=13 ./settings/hyperParamTuning/savio/test_run.sh 'singularity exec --cleanenv -B /global/home/users/gberseth/playground/RL-Framework:/opt/RL-Framework2 /global/scratch/gberseth/SingularityBuilding/ubuntu_learning.img $arg'"
		# command="SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRLSim SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments ./settings/hyperParamTuning/savio/test_run.sh 'singularity exec --cleanenv -B /home/gberseth/playground/RL-Framework:/opt/RL-Framework2 /media/gberseth/Cluster_Port/playground/SingularityBuilding/ubuntu_learning.img -c "$arg"'"
		echo $command 
		# echo SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRLSim SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments "${command[@]}"
		eval $command
		# SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRLSim SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments "${command[@]}"
	done
done

### Toy Sim
# SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRL SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments sbatch --time=50:00:00 ./settings/hyperParamTuning/computeCanada/test_run.sh "singularity exec --cleanenv --home /home/gberseth/playground/RL-Framework/:/opt/RL-Framework /scratch/gberseth/playground/singularity/ubuntu_learning.simg python3.6 trainModel.py --config=settings/terrainRLImitate3D/Path_Folowing/Path_Following.json -p 8 --bootstrap_samples=1000 --rollouts=16 --max_epoch_length=64 --on_policy=fast --save_experience_memory=continual --num_rounds=250 --print_level=train --continue_training=false --saving_update_freq_num_rounds=1 --epochs=4 --eval_epochs=8 --plot=false --meta_sim_samples=2 --meta_sim_threads=2 --plotting_update_freq_num_rounds=5 --metaConfig=settings/hyperParamTuning/deepCrowds.json --additional_on_policy_training_updates=1 --email_log_data_periodically=true --shouldRender=false"

# SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRLSim SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments singularity exec --cleanenv --home ~/playground/RL-Framework:/opt/RL-Framework /global/scratch/gberseth/SingularityBuilding/ubuntu_learning.img python3.6 tuneHyperParameters.py --config=settings/navgame2D/MADDPG/HRL_Tensorflow_NoViz-v3.json -p 2 --num_rounds=10 --continue_training=false --saving_update_freq_num_rounds=1 --plot=false --meta_sim_samples=3 --meta_sim_threads=3 --tuning_threads=2 --plotting_update_freq_num_rounds=5 --metaConfig="$metaExp" --email_log_data_periodically=true --shouldRender=false --bootstrap_samples=0 "$opts"'"

