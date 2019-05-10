#!/bin/bash
## This script is designed to make it easier to start a number of simulation
## example:
## ./settings/hyperParamTuning/element/runSimulations.sh settings/terrainRLImitate3D/CACLA/Humanoid1_Run_Tensorflow.json 250


## declare an array variable
declare -a simExps=(
# sims with lstm critic
# 	"settings/terrainRLImitate3D/CACLA/MultiTask_Imitation_Learning_Viz3D_Walk_128x128_MultiModal_WithCamVel_LSTM_Critic_Reward_LSTM_Siamese_Reward.json"
# 	"settings/terrainRLImitate3D/CACLA/MultiTask_Imitation_Learning_Viz3D_Run_128x128_MultiModal_WithCamVel_LSTM_Critic_Reward_LSTM_Siamese_Reward.json"
# 	"settings/terrainRLImitate3D/TRPO/MultiTask_Imitation_Learning_Viz3D_Walk_128x128_MultiModal_WithCamVel_LSTM_Critic_Reward_LSTM_Siamese_Reward.json"
# 	"settings/terrainRLImitate3D/CACLA/MultiTask_Imitation_Learning_Viz3D_Run_128x128_MultiModal_WithCamVel_LSTM_Critic_Reward_LSTM_Siamese_Reward.json"
# sims with multi head fd comparator
# 	"settings/terrainRLImitate3D/TRPO/MultiTask_Imitation_Learning_Viz3D_Walk_128x128_MultiModal_WithCamVel_LSTM_Critic_Reward_LSTM_Siamese_EncodeDecode_Reward.json"
# 	"settings/terrainRLImitate3D/TRPO/MultiTask_Imitation_Learning_Viz3D_Run_128x128_MultiModal_WithCamVel_LSTM_Critic_Reward_LSTM_Siamese_EncodeDecode_Reward.json"
# 	"settings/terrainRLImitate3D/CACLA/MultiTask_Imitation_Learning_Viz3D_Walk_128x128_MultiModal_WithCamVel_LSTM_Critic_Reward_LSTM_Siamese_EncodeDecode_Reward.json"
# 	"settings/terrainRLImitate3D/CACLA/MultiTask_Imitation_Learning_Viz3D_Run_128x128_MultiModal_WithCamVel_LSTM_Critic_Reward_LSTM_Siamese_EncodeDecode_Reward.json"
# sims with cam velocity
# 	"settings/terrainRLImitate3D/CACLA/MultiTask_Imitation_Learning_Viz3D_Walk_128x128_MultiModal_WithCamVel_Reward_LSTM_Siamese_Reward.json"
# 	"settings/terrainRLImitate3D/CACLA/MultiTask_Imitation_Learning_Viz3D_Run_128x128_MultiModal_WithCamVel_Reward_LSTM_Siamese_Reward.json"
# 	"settings/terrainRLImitate3D/DDPG/MultiTask_Imitation_Learning_Viz3D_Walk_128x128_MultiModal_WithCamVel_Reward_LSTM_Siamese_Reward.json"
# 	"settings/terrainRLImitate3D/DDPG/MultiTask_Imitation_Learning_Viz3D_Run_128x128_MultiModal_WithCamVel_Reward_LSTM_Siamese_Reward.json"
# 	"settings/terrainRLImitate3D/PPO/MultiTask_Imitation_Learning_Viz3D_Walk_128x128_MultiModal_WithCamVel_Reward_LSTM_Siamese_Reward.json"
# 	"settings/terrainRLImitate3D/PPO/MultiTask_Imitation_Learning_Viz3D_Run_128x128_MultiModal_WithCamVel_Reward_LSTM_Siamese_Reward.json"
# 	"settings/terrainRLImitate3D/TRPO/MultiTask_Imitation_Learning_Viz3D_Walk_128x128_MultiModal_WithCamVel_Reward_LSTM_Siamese_Reward.json"
# 	"settings/terrainRLImitate3D/TRPO/MultiTask_Imitation_Learning_Viz3D_Run_128x128_MultiModal_WithCamVel_Reward_LSTM_Siamese_Reward.json"
### Singe modal siamese networks
	"settings/terrainRLImitate3D/CACLA/MultiTask2_Imitation_Learning_All8_128x128_1Sub_WithCamVel_Walk_30FPS_LSTM_FD_Reward_Encode.json"
#	"settings/terrainRLImitate3D/CACLA/MultiTask2_Imitation_Learning_DanceA_128x128_1Sub_WithCamVel_Walk_30FPS_LSTM_FD_Reward_Encode.json"
	"settings/terrainRLImitate3D/CACLA/MultiTask2_Imitation_Learning_Jump_128x128_1Sub_WithCamVel_Walk_30FPS_LSTM_FD_Reward_Encode.json"
	"settings/terrainRLImitate3D/CACLA/MultiTask2_Imitation_Learning_ZombieWalk_128x128_1Sub_WithCamVel_Walk_30FPS_LSTM_FD_Reward_Encode.json"

	"settings/terrainRLImitate3D/TRPO/MultiTask2_Imitation_Learning_All8_128x128_1Sub_WithCamVel_Walk_30FPS_LSTM_FD_Reward_Encode.json"
#	"settings/terrainRLImitate3D/TRPO/MultiTask2_Imitation_Learning_DanceA_128x128_1Sub_WithCamVel_Walk_30FPS_LSTM_FD_Reward_Encode.json"
	"settings/terrainRLImitate3D/TRPO/MultiTask2_Imitation_Learning_Jump_128x128_1Sub_WithCamVel_Walk_30FPS_LSTM_FD_Reward_Encode.json"
	"settings/terrainRLImitate3D/TRPO/MultiTask2_Imitation_Learning_ZombieWalk_128x128_1Sub_WithCamVel_Walk_30FPS_LSTM_FD_Reward_Encode.json"
### Single mode with RNN Critic
	"settings/terrainRLImitate3D/CACLA/MultiTask2_Imitation_Learning_All8_128x128_1Sub_WithCamVel_Walk_30FPS_RNN_Critic_LSTM_FD_Reward_Encode.json"
#	"settings/terrainRLImitate3D/CACLA/MultiTask2_Imitation_Learning_DanceA_128x128_1Sub_WithCamVel_Walk_30FPS_RNN_Critic_LSTM_FD_Reward_Encode.json"
	"settings/terrainRLImitate3D/CACLA/MultiTask2_Imitation_Learning_Jump_128x128_1Sub_WithCamVel_Walk_30FPS_RNN_Critic_LSTM_FD_Reward_Encode.json"
	"settings/terrainRLImitate3D/CACLA/MultiTask2_Imitation_Learning_ZombieWalk_128x128_1Sub_WithCamVel_Walk_30FPS_RNN_Critic_LSTM_FD_Reward_Encode.json"

	"settings/terrainRLImitate3D/TRPO/MultiTask2_Imitation_Learning_All8_128x128_1Sub_WithCamVel_Walk_30FPS_RNN_Critic_LSTM_FD_Reward_Encode.json"
#	"settings/terrainRLImitate3D/TRPO/MultiTask2_Imitation_Learning_DanceA_128x128_1Sub_WithCamVel_Walk_30FPS_RNN_Critic_LSTM_FD_Reward_Encode.json"
	"settings/terrainRLImitate3D/TRPO/MultiTask2_Imitation_Learning_Jump_128x128_1Sub_WithCamVel_Walk_30FPS_RNN_Critic_LSTM_FD_Reward_Encode.json"
	"settings/terrainRLImitate3D/TRPO/MultiTask2_Imitation_Learning_ZombieWalk_128x128_1Sub_WithCamVel_Walk_30FPS_RNN_Critic_LSTM_FD_Reward_Encode.json"
)

rounds=$1
opts=$2
## now loop through the above array
for simExp in "${simExps[@]}"
do
	# echo "$metaConfig"
	# or do whatever with individual element of the array
	# echo "$simConfigFile"
	output=' | tee -a $BORGY_JOB_ID.out'
# 	arg="pushd /home/glen/playground/RL-Framework; python3 trainMetaModel.py --config=${simExp} --metaConfig=settings/hyperParamTuning/element/learned_reward_smoother.json --meta_sim_samples=4 --meta_sim_threads=4 --tuning_threads=2 --num_rounds=${rounds} --plot=false --Multi_GPU=true --on_policy=true --save_experience_memory=continual --continue_training=last --saving_update_freq_num_rounds=1 -p 16 --rollouts=32 --simulation_timeout=1200 --email_log_data_periodically=true --save_video_to_file=eval_movie2.mp4 --visualize_expected_value=false --max_epoch_length=64 ${opts}"
### GPU training
	arg="source ~/tensorflow/bin/activate; pushd /home/glen/playground/RL-Framework; python3 trainMetaModel.py --config=${simExp} --metaConfig=settings/hyperParamTuning/element/learned_reward_smoother.json --meta_sim_samples=4 --meta_sim_threads=4 --tuning_threads=2 --num_rounds=${rounds} --plot=false --Multi_GPU=true --on_policy=true --save_experience_memory=continual --continue_training=last --saving_update_freq_num_rounds=1 -p 16 --rollouts=32 --simulation_timeout=1200 --email_log_data_periodically=true --save_video_to_file=eval_movie2.mp4 --visualize_expected_value=false --max_epoch_length=64 --force_sim_net_to_cpu=true ${opts}"
	arg=$arg$output
	command=(borgy submit --restartable --gpu=4 --cpu=64 --mem=384 -w /home/"$USER" -v /mnt/home/"$USER":/home/"$USER" --image=images.borgy.elementai.net/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -e RLSIMENV_PATH=/home/glen/playground/RLSimulationEnvironments -e HOME=/home/"$USER" -e LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/nvidia-390:~/nvidia-390 -- /bin/bash -c "$arg")
	echo "${command[@]}"
	# eval $command
	"${command[@]}"
done

### TerrainRL sim
# borgy submit --restartable --req-cores=32 --req-ram-gbytes=32 -w /home/${USER} --image=images.borgy.elementai.net/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3 tuneHyperParameters.py --config=settings/terrainRLImitate3D/PPO/Flat_Tensorflow.json --metaConfig=settings/hyperParamTuning/element/use_single_network.json --meta_sim_samples=4 --meta_sim_threads=4 --tuning_threads=1 --num_rounds=250 | tee -a $BORGY_JOB_ID.out'

### Toy Sim
# borgy submit --restartable --req-cores=32 --req-ram-gbytes=32 -w /home/${USER} --image=images.borgy.elementai.net/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3 tuneHyperParameters.py --config=tests/settings/gapGame2D/PPO/FixedSTD_Tensorflow-v2_dropout.json --metaConfig=settings/hyperParamTuning/element/dropout_p_and_use_dropout_in_actor.json --meta_sim_samples=5 --meta_sim_threads=5 --tuning_threads=2 --num_rounds=250 | tee -a $BORGY_JOB_ID.out'

