#!/bin/bash
## This script is designed to make it easier to start a number of simulation
## example:
## ./settings/hyperParamTuning/element/runSimulations.sh settings/terrainRLImitate3D/CACLA/Humanoid1_Run_Tensorflow.json 250


## declare an array variable
declare -a metaExps=(
# 				"settings/hyperParamTuning/element/activation_type.json" 
 				"settings/hyperParamTuning/element/additional_on-poli_trianing_updates.json" 
				"settings/hyperParamTuning/element/advantage_scaling.json" 
 				"settings/hyperParamTuning/element/anneal_exploration.json" 
				"settings/hyperParamTuning/element/anneal_learning_rate.json" 		
				"settings/hyperParamTuning/element/batch_size.json"
# 				"settings/hyperParamTuning/element/CACLA_use_advantage.json"
# 				"settings/hyperParamTuning/element/CACLA_use_advantage_action_weighting.json"
# 				"settings/hyperParamTuning/element/clamp_actions_to_stay_inside_bounds.json" 
 				"settings/hyperParamTuning/element/critic_learning_rate.json"
#				"settings/hyperParamTuning/element/critic_network_layer_sizes.json"
# 				"settings/hyperParamTuning/element/critic_updates_per_actor_update.json"
 				"settings/hyperParamTuning/element/dont_use_td_learning.json"
 				"settings/hyperParamTuning/element/dropout_p.json" 
				"settings/hyperParamTuning/element/exploration_rate.json" 
#				"settings/hyperParamTuning/element/GAE_lambda.json"
# 				"settings/hyperParamTuning/element/initial_temperature.json" 
 				"settings/hyperParamTuning/element/kl_divergence_threshold.json" 
 				"settings/hyperParamTuning/element/learning_rate.json" 
 				"settings/hyperParamTuning/element/normalize_advantage.json" 
  				"settings/hyperParamTuning/element/num_on_policy_rollouts.json" 
#  				"settings/hyperParamTuning/element/optimizer.json" 
# 				"settings/hyperParamTuning/element/policy_activation_type.json"
# 				"settings/hyperParamTuning/element/policy_network_layer_sizes.json"	
#  				"settings/hyperParamTuning/element/ppo_et_factor.json"
# 				"settings/hyperParamTuning/element/pretrain_critic.json" 
# 				"settings/hyperParamTuning/element/reset_on_fall.json" 
				"settings/hyperParamTuning/element/state_normalization.json"		
#  				"settings/hyperParamTuning/element/use_single_network.json"
  				"settings/hyperParamTuning/element/use_stocastic_policy.json"
#  				"settings/hyperParamTuning/element/use_target_net_for_critic.json"
#				"settings/hyperParamTuning/element/value_function_batch_size.json"
)

rounds=$2
simConfigFile=$1
## now loop through the above array
for metaConfig in "${metaExps[@]}"
do
	# echo "$metaConfig"
	# or do whatever with individual element of the array
	# echo "$simConfigFile"
	
	command="borgy submit --restartable --cpu=24 --mem=64 -w /home/${USER} -v /mnt/home/$USER:/home/$USER --image=images.borgy.elementai.lan/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -e RLSIMENV_PATH=/home/glen/playground/RLSimulationEnvironments -e HOME=/home/$USER -e LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/nvidia-390:~/nvidia-390 -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3 tuneHyperParameters.py --config="$simConfigFile" --metaConfig="$metaConfig" --meta_sim_samples=3 --meta_sim_threads=3 --tuning_threads=2 --num_rounds="$rounds" --simulation_timeout=1200 --email_log_data_periodically=true --visualize_expected_value=false --plot=false --Multi_GPU=true --on_policy=true | tee -a $BORGY_JOB_ID.out'"
	echo $command
	eval $command
done

### TerrainRL sim
# borgy submit --restartable --req-cores=32 --req-ram-gbytes=32 -w /home/${USER} --image=images.borgy.elementai.lan/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3 tuneHyperParameters.py --config=settings/terrainRLImitate3D/PPO/Flat_Tensorflow.json --metaConfig=settings/hyperParamTuning/element/use_single_network.json --meta_sim_samples=4 --meta_sim_threads=4 --tuning_threads=1 --num_rounds=250 | tee -a $BORGY_JOB_ID.out'

### Toy Sim
# borgy submit --restartable --req-cores=32 --req-ram-gbytes=32 -w /home/${USER} --image=images.borgy.elementai.lan/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3 tuneHyperParameters.py --config=tests/settings/gapGame2D/PPO/FixedSTD_Tensorflow-v2_dropout.json --metaConfig=settings/hyperParamTuning/element/dropout_p_and_use_dropout_in_actor.json --meta_sim_samples=5 --meta_sim_threads=5 --tuning_threads=2 --num_rounds=250 | tee -a $BORGY_JOB_ID.out'

