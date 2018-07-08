
# Intro

This folder contains some scripts and files to run simulations in a docker image.

## Docker Examples

Login to the container and mount a folder inside the docker image/container
```
docker run -v /mnt/home/${USER}:/home/glen -it images.borgy.elementai.lan/glen:latest bash
```
Login to a running container
```
docker exec -ti {container id} bash
```

To login while using a GPU [you need to reserve a GPU first](https://elementai.atlassian.net/wiki/spaces/DEV/pages/6698906/Using+GPU+nodes)
```
NV_GPU=0 nvidia-docker run -v /mnt/home/${USER}:/home/glen -v /usr/lib/nvidia-390:/usr/lib/nvidia-390 -it images.borgy.elementai.lan/glen:latest bash
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
sudo docker run -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ glen:latest /bin/bash -c "pushd /root/playground/RL-Framework/; python3 trainModel.py --config=tests/settings/gapGame2D/PPO/SingleNet_FixedSTD_Tensorflow-v2.json --plot=false --save_trainData=false --num_rounds=20"
```

Run a simulation and email the results
```
docker run -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ glen:latest /bin/bash -c "pushd /root/playground/RL-Framework/; python3 trainModel.py --config=tests/settings/gapGame2D/PPO/SingleNet_FixedSTD_Tensorflow-v2.json --plot=false --save_trainData=false --num_rounds=10 --metaConfig=settings/hyperParamTuning/elementAI.json"
```
## Submitting things to borgy

Submit to send an email.  
```
borgy submit -w /home/${USER} -i images.borgy.elementai.lan/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework/; python3 sendEmail.py settings/hyperParamTuning/elementAI.json True | tee -a $BORGY_JOB_ID.out'
```

Simulate TerrainRLSIM
Submit to send an email.  
```
borgy submit -w /home/${USER} --image=images.borgy.elementai.lan/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c 'pushd /home/glen/playground/TerrainRL/simAdapter; python3 terrainRLSimTest.py | tee -a $BORGY_JOB_ID.out'
```

Run a simulation  
```
borgy submit --req-cores=1 --req-ram-gbytes=1 -w /home/${USER} -v /usr/lib/nvidia-390:/usr/lib/nvidia-390 --image=images.borgy.elementai.lan/glen:latest -e LD_LIBRARY_PATH=/usr/lib/nvidia-390/:/usr/lib/x86_64-linux-gnu/mesa/ -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -e RLSIMENV_PATH=/home/glen/playground/RLSimulationEnvironments -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3 trainModel.py --config=tests/settings/gapGame2D/PPO/SingleNet_FixedSTD_Tensorflow-v2.json --plot=false --save_trainData=true --num_rounds=10 --metaConfig=settings/hyperParamTuning/elementAI.json --print_level=testing_sim | tee -a $BORGY_JOB_ID.out'
```
or
```
borgy submit --req-cores=16 --req-ram-gbytes=16 -v /usr/lib/nvidia-390/:/usr/lib/nvidia-390/ -w /home/${USER} --image=images.borgy.elementai.lan/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -e LD_LIBRARY_PATH=/usr/lib/nvidia-390/:/usr/lib/x86_64-linux-gnu/mesa/:/usr/lib/x86_64-linux-gnu/mesa-egl/  -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3.6 trainModel.py --config=settings/terrainRLImitate/PPO/Flat_Tensorflow_NoPhase.json --plot=false --save_trainData=true --num_rounds=10 --metaConfig=settings/hyperParamTuning/elementAI.json --print_level=testing_sim | tee -a $BORGY_JOB_ID.out'
```
or for multi char sim
```
borgy submit --req-cores=16 --req-ram-gbytes=16 -v /usr/lib/nvidia-390:/usr/lib/nvidia-390 -w /home/${USER} --image=images.borgy.elementai.lan/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3 trainModel.py --config=settings/terrainRLMultiChar/HLC/CACLA/LargeBlocks_MultiChar_WithVelocity_OnPolicy-v3.json --plot=false --save_trainData=false --num_rounds=10 --metaConfig=settings/hyperParamTuning/elementAI.json --print_level=testing_sim | tee -a $BORGY_JOB_ID.out'
```
Using a GPU
```
borgy submit --req-gpus=1 --req-cores=6 --req-ram-gbytes=6 -w /home/${USER} -v /usr/lib/nvidia-390:/usr/lib/nvidia-390 --image=images.borgy.elementai.lan/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -e RLSIMENV_PATH=/home/glen/playground/RLSimulationEnvironments -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3 trainModel.py --config=settings/projectileGame/PPO/Viz_Imitation.json -p 4 --rollouts=4 --bootstrap_samples=100 --plot=false --save_trainData=true --num_rounds=10 --metaConfig=settings/hyperParamTuning/elementAI.json --print_level=testing_sim | tee -a $BORGY_JOB_ID.out'
```
With GPU training on tensorflow
```
borgy submit --req-gpus=1 --req-cores=8 --req-ram-gbytes=8 -w /home/${USER} -v /usr/lib/nvidia-390:/usr/lib/nvidia-390 --image=images.borgy.elementai.lan/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -e RLSIMENV_PATH=/home/glen/playground/RLSimulationEnvironments -- /bin/bash -c 'source ~/tensorflow/bin/activate; pushd /home/glen/playground/RL-Framework; python3.6 trainModel.py --config=settings/projectileGame/PPO/Viz_Imitation.json -p 4 --rollouts=4 --bootstrap_samples=100 --save_trainData=true --num_rounds=1 --plot=false --save_trainData=true --metaConfig=settings/hyperParamTuning/elementAI.json --print_level=testing_sim | tee -a $BORGY_JOB_ID.out; deactivate'
```

Run a META simulation  
```
borgy submit --req-cores=24 --req-ram-gbytes=24 -w /home/${USER} --image=images.borgy.elementai.lan/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3 trainMetaModel.py --config=settings/terrainRLImitate/PPO/Flat_Tensorflow_NoPhase.json --metaConfig=settings/hyperParamTuning/elementAI.json --meta_sim_samples=3 --meta_sim_threads=3 | tee -a $BORGY_JOB_ID.out'
```

Run a Tuning simulation  
```
borgy submit --restartable --req-cores=10 --req-ram-gbytes=10 -w /home/${USER} --image=images.borgy.elementai.lan/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3 tuneHyperParameters.py --config=tests/settings/gapGame2D/PPO/SingleNet_FixedSTD_Tensorflow-v2-test.json --metaConfig=settings/hyperParamTuning/element/use_single_network.json --meta_sim_samples=5 --meta_sim_threads=5 --tuning_threads=2 | tee -a $BORGY_JOB_ID.out'
```

```
borgy submit --restartable --req-cores=24 --req-ram-gbytes=24 -w /home/${USER} --image=images.borgy.elementai.lan/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3 tuneHyperParameters.py --config=settings/terrainRLImitate/PPO/Flat_Tensorflow.json --metaConfig=settings/hyperParamTuning/element/use_single_network.json --meta_sim_samples=3 --meta_sim_threads=3 --tuning_threads=1 | tee -a $BORGY_JOB_ID.out'
```

Run tests
```
borgy submit --restartable --req-cores=32 --req-ram-gbytes=64 -w /home/${USER} --image=images.borgy.elementai.lan/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3 run_tests.py settings/hyperParamTuning/run_tests.json | tee -a $BORGY_JOB_ID.out'
```
### Info from man page

--cpu float32             Cores required (alias of --req-cores) (default 1)
  -e, --env stringArray         Environment variables
  -f, --file string             YAML description
      --gpu int                 GPUS required (alias of --req-gpus)
  -h, --help                    help for submit
  -i, --image string            Docker image (default "images.borgy.elementai.lan/borsh:latest")
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
sudo docker tag glen:latest images.borgy.elementai.lan/glen:latest
sudo docker push images.borgy.elementai.lan/glen:latest
```
As one command
```
docker commit d4aad5674841 glen:latest; docker tag glen:latest images.borgy.elementai.lan/glen:latest; docker push images.borgy.elementai.lan/glen:latest
```

open images as iteractive container
```
sudo docker run -it glen:latest /bin/bash
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
borgy ps --state alive|cut -d’ ' -f1|tail -n +2|xargs borgy kill
```