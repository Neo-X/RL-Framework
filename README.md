# Learn

This package contains all of the python code used for learning. The code is based
on Keras which is based on Tensorflow.

##### Create conda environment and activate it

###### Anaconda 
If not done yet, install [anaconda](https://www.anaconda.com/) by following the instructions [here](https://www.anaconda.com/download/#linux).
Then reate a anaconda environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).

```
conda create -n <env-name> python=3.6
source activate <env-name>
```

##### B.3. Install the required python dependencies
```
pip install -r requirements.txt
```


## Using The system

Train a simple policy to navigate a particle

```
python3 trainModel.py --config=settings/particleSim/PPO/PPO.json
```

Train a model to imitate humanoid motion.

```
python3 doodad_trainModel.py --config=settings/terrainRLImitate/PPO/Flat_Tensorflow_NoPhase.json
```

Train a humanoid3d LLC for heirarchical training

```
python3 doodad_trainModel.py --config=settings/terrainRLImitate3D/PPO/Humanoid_Flat_Tensorflow_MultiAgent_WithObs_LLC_v3.json
```

Train a hierarchical model to navigate agents across many different scenarios

```
python3 doodad_trainModel.py --config=settings/terrainRLMultiChar/HLC/TD3/ScenarioMixture_WithObs_SimpleReward_Humanoid_1_Tensorflow_v4.json --log_comet=true --shouldRender=false --bootstrap_samples=1 --run_mode=local_docker --meta_sim_samples=4 --meta_sim_threads=4
```

### Running meta simulations

These simulations are designed to sample a few simulations in order to get a more reasonable average of the performance of a method.

```
python3 doodad_trainModel.py --config=tests/settings/particleSim/PPO/PPO_KERAS_Tensorflow.json --metaConfig=settings/hyperParamTuning/elementAI.json --meta_sim_samples=5 --meta_sim_threads=5 --tuning_threads=2
```

## Info about how the code works

1. This code uses doodad to help make running many simulations on different compute systems easy.
2. Hyperparameters can be sampled using the `--tuningConfig=` command on the command line.
3. One of the simulation platforms TerrainRLSim provides a large number of physics-based robot simulations. This library is coded in C++ and very fast. 

