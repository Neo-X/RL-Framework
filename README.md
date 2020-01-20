# Learn

This package contains all of the python code used for learning. The code is based
on Lasagne which is based on Theano.

##### Create either venv or conda environment and activate it

###### Virtualenv
```
pip install --upgrade virtualenv
virtualenv <venv-name>
source <venv-name>/bin/activate
```

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

## Install On Windows

  1. Install [Anaconda](https://www.continuum.io/downloads)
  1. Follow the [setup instruction for Theano](http://deeplearning.net/software/theano/install_windows.html#install-requirements-and-optional-packages)
   which are 
   ```
   	conda install numpy scipy mkl-service libpython m2w64-toolchain <nose> <nose-parameterized> <sphinx> <pydot-ng>
   ```



### For GPU training

 1. sudo apt-get install nvidia-cuda-toolkit nvidia-cuda-dev nvidia-modprobe  
	These libraries are needed to compile code for the GPU as well as to check what GPU devices are available

NOTE: Ran into this issue on Ubuntu 16.04 (https://github.com/Theano/Theano/issues/4425)
As a temporary workaround, I use the following hack:

    Add cmd.append('-D_FORCE_INLINES') just before p = subprocess.Popen( in the file nvcc_compiler.py


## Using The system

```
python3 trainModel.py --config=settings/particleSim/PPO/PPO.json
```

### Running meta simulations

These simulations are designed to sample a few simulations in order to get a more reasonable average of the performance of a method.

```
python3 tuneHyperParameters.py --config=tests/settings/particleSim/PPO/PPO_KERAS_Tensorflow.json --metaConfig=settings/hyperParamTuning/elementAI.json --meta_sim_samples=5 --meta_sim_threads=5 --tuning_threads=2
```


## References

 1. https://github.com/Newmu/Theano-Tutorials
 2. https://github.com/spragunr/deep_q_rl
