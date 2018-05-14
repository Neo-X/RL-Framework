
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
sudo docker run -e TERRAINRL_PATH=/root/playground/TerrainRL/ images.borgy.elementai.lan/glen:latest /bin/bash -c "pushd /root/playground/RL-Framework/; python3 trainModel.py --config=tests/settings/gapGame2D/PPO/SingleNet_FixedSTD_Tensorflow-v2.json --plot=false --save_trainData=false"
```
