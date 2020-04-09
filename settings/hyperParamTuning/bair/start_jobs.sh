#!/bin/bash
## This script is designed to make it easier to start a number of simulation
## example:
## ./settings/hyperParamTuning/element/runSimulations.sh settings/terrainRLImitate3D/CACLA/Humanoid1_Run_Tensorflow.json 250


### Example ray command
# ray exec settings/hyperParamTuning/bair/RL-Framework_Cluster.yaml 'docker run -e TERRAINRL_PATH=/opt/TerrainRLSim/ -e RLSIMENV_PATH=/opt/RLSimulationEnvironments --rm -it us.gcr.io/glen-rl-framework/glen:latest2 bash -c "pushd /opt/RL-Framework; ./update_and_compile.sh; backup_data_continuously.sh & python3 tuneHyperParameters.py --config=tests/settings/particleSim/TRPO/FixedSTD_Tensorflow_KERAS.json -p 2 --bootstrap_samples=100 --pretrain_fd=0 --on_policy=fast --rollouts=4 --max_epoch_length=32 --saving_update_freq_num_rounds=1 --plotting_update_freq_num_rounds=1 --epochs=1 eval_epochs=4 --shouldRender=false --metaConfig=settings/hyperParamTuning/element/additional_on-poli_trianing_updates.json --num_rounds=5 --meta_sim_sample=2 --meta_sim_threads=2 --tuning_threads=2"'

## Meta sim files for evaluating hyper parameter settings
declare -a metaExps=(
# 				"settings/hyperParamTuning/element/activation_type.json" 
#				"settings/hyperParamTuning/element/add_label_noise.json" 
#  				"settings/hyperParamTuning/element/additional_on-poli_trianing_updates.json"
# 				"settings/hyperParamTuning/element/advantage_scaling.json" 
#  				"settings/hyperParamTuning/element/anneal_exploration.json"
#  				"settings/hyperParamTuning/element/anneal_policy_std.json" 
# 				"settings/hyperParamTuning/element/anneal_learning_rate.json" 		
# 				"settings/hyperParamTuning/element/batch_size.json"
# 				"settings/hyperParamTuning/element/clamp_actions_to_stay_inside_bounds.json" 
#  				"settings/hyperParamTuning/element/critic_learning_rate.json"
#				"settings/hyperParamTuning/element/critic_network_layer_sizes.json"
# 				"settings/hyperParamTuning/element/critic_updates_per_actor_update.json"
# 				"settings/hyperParamTuning/element/critic_updates_per_actor_update_trpo.json"
#				"settings/hyperParamTuning/element/dont_use_td_learning.json"
				"settings/hyperParamTuning/element/discount_factor.json"
 				"settings/hyperParamTuning/bair/exploration_rate_MARL.json"
#				"settings/hyperParamTuning/element/GAE_lambda.json"
# 				"settings/hyperParamTuning/element/initial_temperature.json" 
#  				"settings/hyperParamTuning/element/kl_divergence_threshold.json"
# 				"settings/hyperParamTuning/element/normalize_advantage.json" 
#  				"settings/hyperParamTuning/element/num_on_policy_rollouts.json" 
#  				"settings/hyperParamTuning/element/optimizer.json" 
#  				"settings/hyperParamTuning/element/ppo_et_factor.json"
# 				"settings/hyperParamTuning/element/reset_on_fall.json" 
#				"settings/hyperParamTuning/element/state_normalization.json"
#  				"settings/hyperParamTuning/element/use_single_network.json"
#  				"settings/hyperParamTuning/element/use_stocastic_policy.json"
#  				"settings/hyperParamTuning/element/use_target_net_for_critic.json"
#				"settings/hyperParamTuning/element/value_function_batch_size.json"
)

### sim config files to try meta configs on
declare -a simConfigs=(
#	"settings/navgame2D/MADDPG/HRL_Tensorflow_NoViz_HLC-v5.json"
# 	"settings/navgame2D/MADDPG/HRL_Ant_PoseGoal_TRPO_Tensorflow_NoViz_HLC-v5.json"
#	"settings/navgame2D/MADDPG/HRL_Ant_TRPO_Tensorflow_NoViz_HLC-v4.json"
#	"settings/navgame2D/MADDPG/HRL_TRPO_Tensorflow_NoViz_HLC-v4.json"
# 	"settings/navgame2D/MADDPG/HRL_Centralized_Tensorflow_NoViz_HLC-v5.json"
#	"settings/navgame2D/MADDPG/HRL_Ant_Tensorflow_NoViz_HLC-v5.json"
#	"settings/navgame2D/DDPG/HRL_Ant_Tensorflow_NoViz_HLC-v5.json"
#	"settings/navgame2D/TRPO/HRL_Ant_Tensorflow_NoViz_HLC-v5.json"
	
	"settings/navgame2D/DDPG/HRL_Ant_PoseGoal_Tensorflow_NoViz_HLP-v5.json"
	"settings/navgame2D/TRPO/HRL_Ant_PoseGoal_Tensorflow_NoViz_HLP-v5.json"

)

rounds=$1
opts=$2
### For each sim sonfig
for simConfigFile in "${simConfigs[@]}"
do
	## now loop through the above array
	for metaConfig in "${metaExps[@]}"
	do
		# echo "$metaConfig"
		# or do whatever with individual element of the array
		# echo "$simConfigFile"
		output=' | tee -a $BORGY_JOB_ID.out'
	 	arg="docker run -e TERRAINRL_PATH=/opt/TerrainRLSim/ -e RLSIMENV_PATH=/opt/RLSimulationEnvironments --rm -it us.gcr.io/glen-rl-framework/glen:latest2 bash -c 'pushd /opt/RL-Framework; ./update_and_compile.sh; backup_data_continuously.sh & python3 tuneHyperParameters.py --config=${simConfigFile} --shouldRender=false --metaConfig=${metaConfig} --meta_sim_sample=3 --meta_sim_threads=3 --tuning_threads=2 --email_log_data_periodically=true --email_logging_time=28800 --plot=false ${opts}'"
	### GPU training
# 	 	arg="source ~/tensorflow/bin/activate; pushd /home/glen/playground/RL-Framework; python3 tuneHyperParameters.py --config=${simConfigFile} --metaConfig=${metaConfig} --meta_sim_samples=3 --meta_sim_threads=3 --tuning_threads=2 --num_rounds=${rounds} --plot=false --on_policy=fast --shouldRender=false --save_experience_memory=continual --continue_training=last --saving_update_freq_num_rounds=1 -p 4 --rollouts=16 --simulation_timeout=1200 --email_log_data_periodically=true --save_video_to_file=eval_movie2.mp4 --visualize_expected_value=false --force_sim_net_to_cpu=true ${opts}"
		arg=$arg$output
		command=(ray exec settings/hyperParamTuning/bair/RL-Framework_Cluster.yaml "$arg" --tmux)
		echo "${command[@]}"
		# eval $command
		# "${command[@]}"
	done
done

### TerrainRL sim
# borgy submit --restartable --req-cores=32 --req-ram-gbytes=32 -w /home/${USER} --image=images.borgy.elementai.net/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3 tuneHyperParameters.py --config=settings/terrainRLImitate3D/PPO/Flat_Tensorflow.json --metaConfig=settings/hyperParamTuning/element/use_single_network.json --meta_sim_samples=4 --meta_sim_threads=4 --tuning_threads=1 --num_rounds=250 | tee -a $BORGY_JOB_ID.out'

### Toy Sim
# borgy submit --restartable --req-cores=32 --req-ram-gbytes=32 -w /home/${USER} --image=images.borgy.elementai.net/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3 tuneHyperParameters.py --config=tests/settings/gapGame2D/PPO/FixedSTD_Tensorflow-v2_dropout.json --metaConfig=settings/hyperParamTuning/element/dropout_p_and_use_dropout_in_actor.json --meta_sim_samples=5 --meta_sim_threads=5 --tuning_threads=2 --num_rounds=250 | tee -a $BORGY_JOB_ID.out'

