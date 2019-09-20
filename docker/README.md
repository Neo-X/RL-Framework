
# Intro

This folder contains some scripts and files to run simulations in a docker image.

## Docker Examples

Login to the container and mount a folder inside the docker image/container

```
 docker pull us.gcr.io/glen-rl-framework/glen:latest2; docker run -v ~/shared/playground/RL-Framework:/opt/RL-Framework -v ~/.mujoco/:/root/.mujoco/ -e TERRAINRL_PATH=/opt/TerrainRLSim/ -e RLSIMENV_PATH=/opt/RLSimulationEnvironments -e LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin -e LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-390 -it us.gcr.io/glen-rl-framework/glen:latest2 bash
```
Login to a running container

```
docker exec -ti {container id} bash
```

To login while using a GPU [you need to reserve a GPU first](https://elementai.atlassian.net/wiki/spaces/DEV/pages/6698906/Using+GPU+nodes)

```
NV_GPU=0 nvidia-docker run -v /mnt/home/${USER}:/home/glen -v ~/nvidia-390:/usr/lib/nvidia-390 -it images.borgy.elementai.net/glen:latest2 bash
```


To run something from the RLSimulationEnvironments

```
sudo docker run glen:latest /bin/bash -c "python3 /root/playground/RLSimulationEnvironments/EnvTester.py"
```

Run terrainRL

```
sudo docker run -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ glen:latest /bin/bash -c "python3 /root/playground/TerrainRL/simAdapter/terrainRLSimTest.py"
```

To run a learning simulation

```
sudo docker run -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ glen:latest /bin/bash -c "pushd /root/playground/RL-Framework/; python3 trainModel.py --config=tests/settings/gapGame2D/PPO/SingleNet_FixedSTD_Tensorflow-v2.json --plot=false --save_trainData=true --num_rounds=20"
```

Run a simulation and email the results

```
docker run -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ glen:latest /bin/bash -c "pushd /root/playground/RL-Framework/; python3 trainModel.py --config=tests/settings/gapGame2D/PPO/SingleNet_FixedSTD_Tensorflow-v2.json --plot=false --save_trainData=false --num_rounds=10 --metaConfig=settings/hyperParamTuning/elementAI.json"
```
## Submitting things to borgy

Submit to send an email.  

```
borgy submit --restartable -w /home/${USER} -i images.borgy.elementai.net/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework/; python3 sendEmail.py settings/hyperParamTuning/elementAI.json True | tee -a $BORGY_JOB_ID.out'
```

Simulate TerrainRLSIM
Submit to send an email.  

```
borgy submit --restartable -w /home/${USER} --image=images.borgy.elementai.net/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -e LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/nvidia-390:~/nvidia-390 -- /bin/bash -c 'pushd /home/glen/playground/TerrainRL/simAdapter; python3 terrainRLSimTest.py | tee -a $BORGY_JOB_ID.out'
```

Run a simulation  

```
borgy submit --restartable --req-cores=32 --req-ram-gbytes=32 -w /home/${USER} --image=images.borgy.elementai.net/glen:latest -e LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/nvidia-390:~/nvidia-390 -e RLSIMENV_PATH=/home/glen/playground/RLSimulationEnvironments -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3.6 tuneHyperParameters.py --config=settings/cannonGame/MBRL/FixedSTD_Tensorflow-v2.json --metaConfig=settings/hyperParamTuning/element/fd_policy_activation_type.json --meta_sim_samples=4 --meta_sim_threads=4 --tuning_threads=1 --num_rounds=20 | tee -a $BORGY_JOB_ID.out'
```
or

```
borgy submit --restartable --req-cores=16 --req-ram-gbytes=16 -w /home/${USER} --image=images.borgy.elementai.net/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -e RLSIMENV_PATH=/home/glen/playground/RLSimulationEnvironments -e LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/nvidia-390:~/nvidia-390 -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3.6 trainModel.py --config=settings/terrainRLImitate3D/PPO/Humanoid1_Walk_Tensorflow.json --num_rounds=1000 --metaConfig=settings/hyperParamTuning/deepCrowds.json --meta_sim_samples=2 --meta_sim_threads=2 --tuning_threads=2 --plot=false -p 16 --rollouts=16'
```
or for headless 2d biped terrain rl

```
borgy submit --restartable --req-gpus=1 --req-cores=16 --req-ram-gbytes=16 -w /home/${USER} -v /usr/lib/nvidia-390:/usr/lib/nvidia-390 --image=images.borgy.elementai.net/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -e RLSIMENV_PATH=/home/glen/playground/RLSimulationEnvironments -e LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/nvidia-390:~/nvidia-390 -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3.6 trainModel.py --config=settings/terrainRLImitate/PPO/Viz3D_30FPS.json -p 16 --bootstrap_samples=100 --rollouts=16 --num_rounds=250 --metaConfig=settings/hyperParamTuning/elementAI.json --plot=false '
```

or for multi char sim

```
borgy submit --restartable --req-cores=16 --req-ram-gbytes=16 -e LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/nvidia-390:~/nvidia-390 -w /home/${USER} --image=images.borgy.elementai.net/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3 trainModel.py --config=settings/terrainRLMultiChar/HLC/CACLA/LargeBlocks_MultiChar_WithVelocity_OnPolicy-OLDLLC.json --plot=false --save_trainData=true --num_rounds=250 --metaConfig=settings/hyperParamTuning/elementAI.json --print_level=testing_sim | tee -a $BORGY_JOB_ID.out'
```
Using a GPU

```
borgy submit --restartable --req-gpus=1 --req-cores=6 --req-ram-gbytes=6 -w /home/${USER} -v /usr/lib/nvidia-390:/usr/lib/nvidia-390 --image=images.borgy.elementai.net/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -e RLSIMENV_PATH=/home/glen/playground/RLSimulationEnvironments -e LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/nvidia-390:~/nvidia-390 -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3 trainModel.py --config=settings/projectileGame/PPO/Viz_Imitation.json -p 4 --rollouts=4 --bootstrap_samples=100 --plot=false --save_trainData=true --num_rounds=10 --metaConfig=settings/hyperParamTuning/elementAI.json --print_level=testing_sim | tee -a $BORGY_JOB_ID.out'
```
or 

```
borgy submit --restartable --req-gpus=4 --req-cores=64 --req-ram-gbytes=256 -w /home/${USER} -v /usr/lib/nvidia-390:/usr/lib/nvidia-390 -v /mnt/home/$USER:/mnt/home/$USER --image=images.borgy.elementai.net/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -e RLSIMENV_PATH=/home/glen/playground/RLSimulationEnvironments -e LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/nvidia-390:~/nvidia-390 -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3.6 -O tuneHyperParameters.py --config=settings/terrainRLImitate/TRPO/Imitation_Learning_Viz3D_128x128_1Sub_30FPS_LSTM_FD_Encode_Reward.json --num_rounds=250 --metaConfig=settings/hyperParamTuning/element/include_agent_imitator_pairs.json --meta_sim_samples=2 --meta_sim_threads=2 --tuning_threads=2 --plot=false --Multi_GPU=true --on_policy=fast --save_experience_memory=continual --continue_training=last --saving_update_freq_num_rounds=1 -p 16 --rollouts=16 --simulation_timeout=1200 --email_log_data_periodically=true | tee -a $BORGY_JOB_ID.out'
```
With GPU training on tensorflow

```
borgy submit --restartable --req-gpus=1 --req-cores=8 --req-ram-gbytes=8 -w /home/${USER} -v /usr/lib/nvidia-390:/usr/lib/nvidia-390 --image=images.borgy.elementai.net/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -e RLSIMENV_PATH=/home/glen/playground/RLSimulationEnvironments -e LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/nvidia-390:~/nvidia-390 -- /bin/bash -c 'source ~/tensorflow/bin/activate; pushd /home/glen/playground/RL-Framework; python3.6 trainModel.py --config=settings/projectileGame/PPO/Viz_Imitation.json -p 4 --rollouts=4 --bootstrap_samples=100 --save_trainData=true --num_rounds=1 --plot=false --save_trainData=true --metaConfig=settings/hyperParamTuning/elementAI.json --print_level=testing_sim | tee -a $BORGY_JOB_ID.out; deactivate'
```

Run a META simulation  

```
borgy submit --restartable --req-gpus=0 --req-cores=16 --req-ram-gbytes=32 -w /home/${USER} -v /mnt/home/$USER:/home/$USER --image=images.borgy.elementai.net/glen:latest -e HOME=/home/$USER -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -e RLSIMENV_PATH=/home/glen/playground/RLSimulationEnvironments -e LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/nvidia-390:~/nvidia-390 -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3.6 -O trainMetaModel.py --config=settings/terrainRLImitate/PPO/Flat_Tensorflow_NoPhase_Torque.json --num_rounds=1000 --metaConfig=settings/hyperParamTuning/element/use_fall_reward_shaping2.json --meta_sim_samples=2 --meta_sim_threads=2 --tuning_threads=2 --plot=false --Multi_GPU=true --on_policy=fast --save_experience_memory=continual --continue_training=last --saving_update_freq_num_rounds=1 -p 8 --rollouts=16 --simulation_timeout=1200 --email_log_data_periodically=true --visualize_expected_value=false --max_epoch_length=256 | tee -a $BORGY_JOB_ID.out'
```

Run a Tuning simulation  

```
borgy submit --restartable --req-cores=10 --req-ram-gbytes=10 -w /home/${USER} --image=images.borgy.elementai.net/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -e LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/nvidia-390:~/nvidia-390 -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3 tuneHyperParameters.py --config=tests/settings/gapGame2D/PPO/SingleNet_FixedSTD_Tensorflow-v2-test.json --metaConfig=settings/hyperParamTuning/element/use_single_network.json --meta_sim_samples=5 --meta_sim_threads=5 --tuning_threads=2 | tee -a $BORGY_JOB_ID.out'
```
```
borgy submit --restartable --req-gpus=0 --req-cores=32 --req-ram-gbytes=64 -w /home/${USER} -v /usr/lib/nvidia-390:/usr/lib/nvidia-390 -v /mnt/home/$USER:/home/$USER --image=images.borgy.elementai.net/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -e RLSIMENV_PATH=/home/glen/playground/RLSimulationEnvironments -e LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/nvidia-390:~/nvidia-390 -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3.6 -O tuneHyperParameters.py --config=settings/terrainRLImitate/PPO/Flat_Tensorflow_NoPhase.json --num_rounds=500 --metaConfig=settings/hyperParamTuning/element/exploration_rate.json --meta_sim_samples=2 --meta_sim_threads=2 --tuning_threads=2 --plot=false --Multi_GPU=true --on_policy=fast --save_experience_memory=continual --continue_training=last --saving_update_freq_num_rounds=1 -p 16 --rollouts=16 --simulation_timeout=1200 --email_log_data_periodically=true | tee -a $BORGY_JOB_ID.out'
```

Tuning simulation with GPU

```
 borgy submit --restartable --req-gpus=1 --req-cores=32 --req-ram-gbytes=32 -w /home/${USER} -v /usr/lib/nvidia-390:/usr/lib/nvidia-390 --image=images.borgy.elementai.net/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -e RLSIMENV_PATH=/home/glen/playground/RLSimulationEnvironments -e LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/nvidia-390:~/nvidia-390 -- /bin/bash -c 'source ~/tensorflow/bin/activate; pushd /home/glen/playground/RL-Framework; python3.6 tuneHyperParameters.py --config=settings/projectileGame/PPO/Viz_Imitation.json --metaConfig=settings/hyperParamTuning/element/use_single_network.json -p 4 --rollouts=4 --bootstrap_samples=100 --save_trainData=true --plot=false --save_trainData=true --meta_sim_samples=4 --meta_sim_threads=4 --tuning_threads=1 --num_rounds=10 | tee -a $BORGY_JOB_ID.out; deactivate'
```
or periodically send email updates with videos

```
borgy submit --restartable --req-gpus=4 --req-cores=64 --req-ram-gbytes=256 -w /home/${USER} -v /usr/lib/nvidia-390:/usr/lib/nvidia-390 -v /mnt/home/$USER:/home/$USER --image=images.borgy.elementai.net/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -e RLSIMENV_PATH=/home/glen/playground/RLSimulationEnvironments -e LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/nvidia-390:~/nvidia-390 -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3.6 -O tuneHyperParameters.py --config=settings/terrainRLImitate/CACLA/Imitation_Learning_Viz3D_128x128_1Sub_30FPS_LSTM_FD_Dual_Encode_Reward.json --num_rounds=250 --metaConfig=settings/hyperParamTuning/element/use_learned_reward_function.json --meta_sim_samples=2 --meta_sim_threads=2 --tuning_threads=2 --plot=false --Multi_GPU=true --on_policy=fast --save_experience_memory=continual --continue_training=last --saving_update_freq_num_rounds=1 -p 16 --rollouts=16 --simulation_timeout=1200 --email_log_data_periodically=true --visualize_expected_value=false --save_video_to_file=eval_movie.mp4 | tee -a $BORGY_JOB_ID.out'
```

Run tests

```
borgy submit --restartable --req-gpus=1 --req-cores=32 --req-ram-gbytes=64  -w /home/${USER} --image=images.borgy.elementai.net/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -e LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/nvidia-390:~/nvidia-390  -e RLSIMENV_PATH=/home/glen/playground/RLSimulationEnvironments -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3.6 run_tests.py settings/hyperParamTuning/run_tests.json | tee -a $BORGY_JOB_ID.out'
```

Run new sim

```
docker run -e TERRAINRL_PATH=/opt/TerrainRLSim/ -e RLSIMENV_PATH=/opt/RLSimulationEnvironments --rm -it us.gcr.io/glen-rl-framework/glen:latest2 bash -c "pushd /opt/RL-Framework; ./update_and_compile.sh; backup_data_continuously.sh & python3  tuneHyperParameters.py  --config=settings/navgame2D/MADDPG/HRL_Tensorflow_NoViz_HLC-v5.json --metaConfig=settings/hyperParamTuning/element/exploration_rate.json -p 2 --bootstrap_samples=1000 --pretrain_fd=0 --on_policy=true  --plot=false --meta_sim_samples=3 --meta_sim_threads=3 --tuning_threads=2"
```

### Info from man page

--cpu float32             Cores required (alias of --req-cores) (default 1)
  -e, --env stringArray         Environment variables
  -f, --file string             YAML description
      --gpu int                 GPUS required (alias of --req-gpus)
  -h, --help                    help for submit
  -i, --image string            Docker image (default "images.borgy.elementai.net/borsh:latest")
      --interactive             Interactive job. To be scheduled before non-interactive jobs
      --label stringArray       Labels
      --max-run-time-secs int   Max execution time in sec
      --mem int                 RAM required in Gb (alias of --req-ram-gbytes) (default 1)
      --name string             Name
      --req-cores float32       Cores required (default 1)
      --req-gpus int            GPUS required
      --req-ram-gbytes int      RAM required in Gb (default 1)
      --restart string          Restart policy on interrupted job. [no, on-interruption] (default "no")
      --restartable             Enable restart on interrupted job.
  -v, --volume stringArray      Volumes to mount in your container
  -w, --workdir string          Working directory inside the container
  -X, --x-sidecar               Start X11 VNC sidecar

## Helpful commands

Commit docker container

```
sudo docker commit d4aad5674841 glen:latest
sudo docker tag glen:latest images.borgy.elementai.net/glen:latest
sudo docker push images.borgy.elementai.net/glen:latest
```
As one command

```
docker commit f76d8e488c44 glen:latest2; docker tag glen:latest2 us.gcr.io/glen-rl-framework/glen:latest2; docker push us.gcr.io/glen-rl-framework/glen:latest2
```

revise commit tree

```
docker tag 2844 imagename # <-- that's the secret right there
```

open images as iteractive container

```
sudo docker run -it glen:latest /bin/bash
```

Name a docker container

```
docker run -d --name rl-framework-container us.gcr.io/glen-rl-framework/glen:latest2
```

Find children of docker image

```
for i in $(docker images -q)
do
    docker history $i | grep -q 8d5495222da7 && echo $i
done | sort -u
```

To remove an image that is no longer used

```
docker rmi <img_id>
```

Connect to a running borgy job

```
borgy exec {job id} bash
```

Run a command in a screen session and exits when the command is done.

```
screen -S sessionName bash -c 'for((i=1;i<=100;i+=1)); do echo "Welcome $i times"; sleep 1; done'
```

Kill all active borgy jobs

```
borgy ps --state alive|cut -d' ' -f1|tail -n +2|xargs borgy kill
```

/usr/local/nvidia/lib:/usr/local/nvidia/lib64