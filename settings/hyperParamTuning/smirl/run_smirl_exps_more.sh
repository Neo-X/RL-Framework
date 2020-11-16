#!/bin/bash
## This script is designed to make it easier to start a number of simulation
## example:


## declare an array variable
declare -a metaExps=(
	"settings/hyperParamTuning/smirl/exploration_rate.json" 
)

## declare an array variable
declare -a simConfigs=(
	"settings/terrainRLImitate3D/TRPO/smirl/Pedestal_NoPhase_BayesianSurprise.json"
	"settings/terrainRLImitate3D/TRPO/smirl/Treadmil_NoPhase_BayesianSurprise.json"
	"settings/terrainRLImitate3D/TRPO/smirl/Cliff_NoPhase_BayesianSurprise.json"
#
)

opts='--print_level=hyper_train --num_rounds=1000 -p 8 --on_policy=fast --run_mode=ec2 --log_comet=true --bootstrap_sample=6 --meta_sim_sample=6 --random_seed=444 --meta_sim_threads=6 --experiment_logging="{\"use_comet\":true,\"project_name\":\"bayesiansurprise\"}"'
# opts='--print_level=hyper_train --num_rounds=500 -p 8 --on_policy=fast --run_mode=local_docker --log_comet=true --bootstrap_sample=1 --meta_sim_sample=1 --random_seed=555 --meta_sim_threads=1 --experiment_logging="{\"use_comet\":true,\"project_name\":\"bayesiansurprise\"}"'
### For each sim sonfig
for simConfigFile in "${simConfigs[@]}"
do
	## now loop through the above array
	for metaConfig in "${metaExps[@]}"
	do
		command=(python doodad_trainModel.py --config=$simConfigFile --metaConfig=$metaConfig  $opts)
		echo "${command[@]}"
 		"${command[@]}"
	done
done

