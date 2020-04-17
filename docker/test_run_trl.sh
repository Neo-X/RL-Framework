docker build -f Dockerfile_trl -t rlframe_trl:latest .

# docker build --default-runtime=nvidia -f Dockerfile_trl -t rlframe_trl:latest .
# docker run --rm --runtime=nvidia -v /opt/OpenGL/:/usr/lib/nvidia -it rlframe_trl:latest /bin/bash -c "pushd /root/playground/RL-Framework; git pull origin master; python3 trainModel.py --config=settings/MiniGrid/TagEnv/PPO/Tag_SLAC_mini.json -p 4 --bootstrap_samples=2000 --max_epoch_length=16 --rollouts=4 --skip_rollouts=true --train_actor=false --train_critic=false --epochs=32 --fd_updates_per_actor_update=64 --on_policy=fast"

### Not using nvidia/cuda for now
docker build -f Dockerfile_trl -t rlframe_trl:latest .

echo "arg 1 $1"
# cmd='python3 trainModel.py --config=settings/MiniGrid/TagEnv/PPO/Tag_Dual_FullObserve_SLAC_mini.json  -p 2 --bootstrap_samples=10000 --max_epoch_length=32 --rollouts=32 --pretrain_fd=0 --plot=false --save_video_to_file=eval.mp4 --metaConfig=settings/hyperParamTuning/element/exploration_rate.json --experiment_logging="{\"use_comet\": true, \"project_name\": \"ic2\"}"'
cmd=$1
fullcmd="pushd /root/playground/TerrainRLSim; git pull origin master; popd; pushd /root/playground/RLSimulationEnvironments; git pull origin master; popd; pushd /root/playground/RL-Framework; git pull origin master; ${cmd}"
echo $fullcmd
command=(docker run --rm -it rlframe_trl:latest /bin/bash -c "$fullcmd" )
# arg="pushd /home/glen/playground/RL-Framework; python3 tuneHyperParameters.py --config=${simConfigFile} --metaConfig=${metaConfig} --meta_sim_samples=3 --meta_sim_threads=3 --tuning_threads=2 --num_rounds=${rounds} --plot=false --Multi_GPU=true --on_policy=fast --save_experience_memory=continual --continue_training=last --saving_update_freq_num_rounds=1 -p 6 --rollouts=12 --simulation_timeout=1200 --email_log_data_periodically=true --visualize_expected_value=false ${opts}"
### GPU training
# 	arg="source ~/tensorflow/bin/activate; pushd /home/glen/playground/RL-Framework; python3 tuneHyperParameters.py --config=${simConfigFile} --metaConfig=${metaConfig} --meta_sim_samples=2 --meta_sim_threads=2 --tuning_threads=2 --num_rounds=${rounds} --plot=false --Multi_GPU=true --on_policy=true --save_experience_memory=continual --continue_training=last --saving_update_freq_num_rounds=1 -p 16 --rollouts=16 --simulation_timeout=1200 --email_log_data_periodically=true --save_video_to_file=eval_movie2.mp4 --visualize_expected_value=false --max_epoch_length=64 --force_sim_net_to_cpu=true ${opts}"
# arg=$arg$output
# command=(borgy submit --restartable --gpu=0 --cpu=36 --mem=64 -w /home/"$USER" -v /mnt/home/"$USER":/home/"$USER" --image=images.borgy.elementai.net/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -e RLSIMENV_PATH=/home/glen/playground/RLSimulationEnvironments -e HOME=/home/"$USER" -e LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/nvidia-390:~/nvidia-390 -- /bin/bash -c "$arg")
echo "${command[@]}"
# eval $command
"${command[@]}"
