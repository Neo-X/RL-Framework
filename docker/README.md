
# Intro

This folder contains some scripts and files to run simulations in a docker image.

### Examples

To run something from the RLSimulationEnvironments in the container

```
sudo docker run glen:latest /bin/bash -c "python3 /root/playground/RLSimulationEnvironments/EnvTester.py"
```

Run terrainRL
```
sudo docker run -e TERRAINRL_PATH=/root/playground/TerrainRL/ glen:latest /bin/bash -c "python3 /root/playground/TerrainRL/simAdapter/terrainRLSimTest.py"
```

To run a learning simulation

```
sudo docker run -e TERRAINRL_PATH=/root/playground/TerrainRL/ glen:latest /bin/bash -c "pushd /root/playground/RL-Framework/; python3 trainModel.py --config=tests/settings/gapGame2D/PPO/SingleNet_FixedSTD_Tensorflow-v2.json --plot=false --save_trainData=false --num_rounds=20"
```

Run a simulation and email the results
```
docker run -e TERRAINRL_PATH=/root/playground/TerrainRL/ glen:latest /bin/bash -c "pushd /root/playground/RL-Framework/; python3 trainModel.py --config=tests/settings/gapGame2D/PPO/SingleNet_FixedSTD_Tensorflow-v2.json --plot=false --save_trainData=false --num_rounds=10 --metaConfig=settings/hyperParamTuning/elementAI.json"
```
## Submitting things to borgy

Submit to send an email.  
```
borgy submit -i images.borgy.elementai.lan/glen:latest -e TERRAINRL_PATH=/root/playground/TerrainRL/ -- /bin/bash -c "pushd /root/playground/RL-Framework/; python3 sendEmail.py settings/hyperParamTuning/elementAI.json True"
```

Run a simulation  
```
borgy submit -i images.borgy.elementai.lan/glen:latest -e TERRAINRL_PATH=/root/playground/TerrainRL/ -- /bin/bash -c "pushd /root/playground/RL-Framework/; python3 trainModel.py --config=tests/settings/gapGame2D/PPO/SingleNet_FixedSTD_Tensorflow-v2.json --plot=false --save_trainData=true --num_rounds=10 --metaConfig=settings/hyperParamTuning/elementAI.json"
```

### Info from man page

--req-cores float32       Cores required (default 1)
--req-gpus int            GPUS required
--req-ram-gbytes int      RAM required in Gb (default 1)

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

