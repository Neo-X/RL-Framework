
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
borgy submit -w /home/${USER} -i images.borgy.elementai.lan/glen:new -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c "pushd /home/glen/playground/RL-Framework/; screen -S sessionName bash -c 'python3 sendEmail.py settings/hyperParamTuning/elementAI.json True'"
```

Simulate TerrainRLSIM
Submit to send an email.  
```
borgy submit -w /home/${USER} --image=images.borgy.elementai.lan/glen:new -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c "pushd /home/glen/playground/TerrainRL/simAdapter; python3 terrainRLSimTest.py"
```

Run a simulation  
```
borgy submit --req-cores=1 --req-ram-gbytes=1 -w /home/${USER} --image=images.borgy.elementai.lan/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c "pushd /home/glen/playground/RL-Framework; python3 trainModel.py --config=tests/settings/gapGame2D/PPO/SingleNet_FixedSTD_Tensorflow-v2.json --plot=false --save_trainData=true --num_rounds=10 --metaConfig=settings/hyperParamTuning/elementAI.json --print_level=testing_sim 2> $BORGY_JOB_ID.err > $BORGY_JOB_ID.out"
```

Run a META simulation  
```
borgy submit --req-cores=4 --req-ram-gbytes=4 -w /home/${USER} --image=images.borgy.elementai.lan/glen:latest -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c "pushd /home/glen/playground/RL-Framework; python3 trainMetaModel.py tests/settings/particleSim/PPO/PPO_KERAS.json settings/hyperParamTuning/elementAI.json 2 2"
```

Run a Tuning simulation  
```
borgy submit --restartable --req-cores=10 --req-ram-gbytes=10 -w /home/${USER} --image=images.borgy.elementai.lan/glen:new -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c "pushd /home/glen/playground/RL-Framework; python3 tuneHyperParameters.py tests/settings/gapGame2D/PPO/SingleNet_FixedSTD_Tensorflow-v2-test.json settings/hyperParamTuning/element/normalize_advantage.json"
```

```
borgy submit --restartable --req-cores=24 --req-ram-gbytes=24 -w /home/${USER} --image=images.borgy.elementai.lan/glen:new -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c "pushd /home/glen/playground/RL-Framework; python3 tuneHyperParameters.py settings/terrainRLImitate/PPO/Flat_Tensorflow.json settings/hyperParamTuning/element/normalize_advantage.json"
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

Connect to a running borgy job
```
borgy exec {job id} bash
```

Run a command in a screen session and exits when the command is done.
```
screen -S sessionName bash -c 'for((i=1;i<=100;i+=1)); do echo "Welcome $i times"; sleep 1; done'
```