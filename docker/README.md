
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
docker run -it us.gcr.io/glen-rl-framework/glen:latest2 bash
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

## Run new sim

```
docker run --rm -it us.gcr.io/glen-rl-framework/glen:latest2 bash -c "pushd /opt/RL-Framework; ./update_and_compile.sh; backup_data_continuously.sh & python3  tuneHyperParameters.py  --config=settings/navgame2D/MADDPG/HRL_Tensorflow_NoViz_HLC-v5.json --metaConfig=settings/hyperParamTuning/element/exploration_rate.json -p 2 --bootstrap_samples=1000 --pretrain_fd=0 --on_policy=true  --plot=false --meta_sim_samples=3 --meta_sim_threads=3 --tuning_threads=2"
```

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