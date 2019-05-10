#!/bin/bash
## This script is designed to make it easier to start a number of simulation
## example:
## ./settings/hyperParamTuning/element/runSimulations.sh settings/terrainRLImitate3D/CACLA/Humanoid1_Run_Tensorflow.json 250


## declare an array variable
declare -a metaExps=(
 				"settings/hyperParamTuning/element/add_label_noise.json" 
 				"settings/hyperParamTuning/element/image_noise_scale.json" 
 				"settings/hyperParamTuning/element/imperfect_compare_offset.json" 
 				"settings/hyperParamTuning/element/lstm_batch_size.json" 
 				"settings/hyperParamTuning/element/fd_learning_rate.json" 
				"settings/hyperParamTuning/element/fd_network_layer_sizes_json.json"
				"settings/hyperParamTuning/element/fd_reward_network_layer_sizes_json.json"
# 				"settings/hyperParamTuning/element/state_normalization.json"
)

## declare an array variable
declare -a simConfigs=(
 				"settings/terrainRLImitate/CACLA/Imitation_Learning_Walk_64x64_1Sub_WithCamVel_Walk_30FPS_LSTM_FD_Reward_Encode_MetaFD2.json"
 				"settings/terrainRLImitate3D/TRPO/MultiTask2_Imitation_Learning_ZombieWalk_64x64_1Sub_MultiModal_WithCamVel_Walk_30FPS_LSTM_FD_Reward_Encode.json" 
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
	#  	arg="pushd /home/glen/playground/RL-Framework; python3 tuneHyperParameters.py --config=${simConfigFile} --metaConfig=${metaConfig} --meta_sim_samples=3 --meta_sim_threads=3 --tuning_threads=2 --num_rounds=${rounds} --plot=false --Multi_GPU=true --on_policy=fast --save_experience_memory=continual --continue_training=last --saving_update_freq_num_rounds=1 -p 6 --rollouts=12 --simulation_timeout=1200 --email_log_data_periodically=true --visualize_expected_value=false ${opts}"
	### GPU training
		arg="source ~/tensorflow/bin/activate; pushd /home/glen/playground/RL-Framework; python3 tuneHyperParameters.py --config=${simConfigFile} --metaConfig=${metaConfig} --meta_sim_samples=4 --meta_sim_threads=2 --tuning_threads=2 --num_rounds=${rounds} --plot=false --Multi_GPU=true --on_policy=true --save_experience_memory=continual --continue_training=false --saving_update_freq_num_rounds=1 -p 8 --rollouts=16 --simulation_timeout=1200 --train_critic=false --train_actor=false --eval_epochs=0 --skip_rollouts=true --email_log_data_periodically=true --visualize_expected_value=false --force_sim_net_to_cpu=true --gpus=2 ${opts}"
		arg=$arg$output
		command=(borgy submit --restartable --gpu=2 --cpu=32 --mem=128 --max-run-time-secs=28800 -w /home/"$USER" -v /mnt/home/"$USER":/home/"$USER" --image=images.borgy.elementai.net/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -e RLSIMENV_PATH=/home/glen/playground/RLSimulationEnvironments -e HOME=/home/"$USER" -e LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/nvidia-390:~/nvidia-390 -- /bin/bash -c "$arg")
		echo "${command[@]}"
		# eval $command
		"${command[@]}"
	done
done

### TerrainRL sim
# borgy submit --restartable --req-cores=32 --req-ram-gbytes=32 -w /home/${USER} --image=images.borgy.elementai.net/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3 tuneHyperParameters.py --config=settings/terrainRLImitate3D/PPO/Flat_Tensorflow.json --metaConfig=settings/hyperParamTuning/element/use_single_network.json --meta_sim_samples=4 --meta_sim_threads=4 --tuning_threads=1 --num_rounds=250 | tee -a $BORGY_JOB_ID.out'

### Toy Sim
# borgy submit --restartable --req-cores=32 --req-ram-gbytes=32 -w /home/${USER} --image=images.borgy.elementai.net/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3 tuneHyperParameters.py --config=tests/settings/gapGame2D/PPO/FixedSTD_Tensorflow-v2_dropout.json --metaConfig=settings/hyperParamTuning/element/dropout_p_and_use_dropout_in_actor.json --meta_sim_samples=5 --meta_sim_threads=5 --tuning_threads=2 --num_rounds=250 | tee -a $BORGY_JOB_ID.out'

