

## Monitorying jobs

### monitor by user:
squeue -u <username>
squeue -u <username> -t RUNNING
squeue -u <username> -t PENDING

### monitor specific job by <job_id>
scontrol show job -dd <job_id>

## Using singularity on Compute Canada

```
SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRL SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments ./settings/hyperParamTuning/computeCanada/test_run.sh "singularity exec --cleanenv --home /home/gberseth/playground/RL-Framework/:/opt/RL-Framework /Cluster/playground/singularity/deepcrowds.simg python3.6 trainModel.py --config=settings/terrainRLMultiChar/HLC/CACLA/LargeBlocks_Multi_Char_On_Policy.json -p 8 --bootstrap_samples=10000 --rollouts=16 --max_epoch_length=64 --on_policy=fast --save_experience_memory=continual --num_rounds=1000 --print_level=train --continue_training=false --saving_update_freq_num_rounds=1 --epochs=4 --eval_epochs=8 --plot=false --meta_sim_samples=2 --meta_sim_threads=2 --plotting_update_freq_num_rounds=1 --metaConfig=settings/hyperParamTuning/deepCrowds.json --additional_on-poli_trianing_updates=1"
```

Star HLC training sim
```
SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRL SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments ./settings/hyperParamTuning/computeCanada/test_run.sh "singularity exec --cleanenv --home ~/playground/RL-Framework/:/opt/RL-Framework ~/scratch/playground/singularity/ubuntu_learning.simg python3.6 trainModel.py --config=settings/terrainRLImitate3D/Path_Folowing/LargeBlocks_OnPolicy-NEWLLC.json -p 8 --bootstrap_samples=5000 --rollouts=8 --max_epoch_length=64 --on_policy=fast --save_experience_memory=continual --num_rounds=1000 --print_level=train --continue_training=false --saving_update_freq_num_rounds=1 --epochs=4 --eval_epochs=16 --plot=false --meta_sim_samples=2 --meta_sim_threads=2 --plotting_update_freq_num_rounds=10 --metaConfig=settings/hyperParamTuning/deepCrowds.json --additional_on-poli_trianing_updates=1 --email_log_data_periodically=true"
```

- Run Meta training simulation
```
SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRL SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments sbatch --time=50:00:00 ./settings/hyperParamTuning/computeCanada/test_run.sh "singularity exec --cleanenv --home /home/gberseth/playground/RL-Framework/:/opt/RL-Framework /scratch/gberseth/playground/singularity/ubuntu_learning.img python3.6 trainMetaModel.py -O --config=settings/terrainRLMultiChar/HLC/CACLA/LargeBlocks_Multi_Char_On_Policy.json -p 8 --on_policy=fast --save_experience_memory=continual --num_rounds=250 --print_level=train --continue_training=false --saving_update_freq_num_rounds=1 --plot=false --meta_sim_samples=2 --meta_sim_threads=2 --plotting_update_freq_num_rounds=5 --metaConfig=settings/hyperParamTuning/deepCrowds.json --additional_on-poli_trianing_updates=1 --email_log_data_periodically=true --shouldRender=false"
```

- Run tuning training simulation
```
SINGULARITYENV_TERRAINRL_PATH=/opt/TerrainRL SINGULARITYENV_RLSIMENV_PATH=/opt/RLSimulationEnvironments sbatch --time=50:00:00 --mem=65536M --cpus-per-task=32 ./settings/hyperParamTuning/computeCanada/test_run.sh "singularity exec --cleanenv --home /home/gberseth/playground/RL-Framework/:/opt/RL-Framework /scratch/gberseth/playground/singularity/ubuntu_learning.img python3.6 tuneHyperParameters.py --config=settings/terrainRLImitate3D/DDPG/Humanoid1_Run_Tensorflow.json -p 8 --bootstrap_samples=10000 --on_policy=fast --save_experience_memory=continual --num_rounds=500 --print_level=train --continue_training=last --saving_update_freq_num_rounds=1 --plot=false --meta_sim_samples=2 --meta_sim_threads=2 --plotting_update_freq_num_rounds=5 --metaConfig=settings/hyperParamTuning/element/action_bounds_humanoid3D.json --additional_on-poli_trianing_updates=1 --email_log_data_periodically=true --shouldRender=false --tuning_threads=2"
```
