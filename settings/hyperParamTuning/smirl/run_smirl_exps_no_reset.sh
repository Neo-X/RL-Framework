#!/bin/bash
## This script is designed to make it easier to start a number of simulation
## example:


## declare an array variable
declare -a metaExps=(
	"settings/hyperParamTuning/smirl/exploration_rate.json" 
)

## declare an array variable
declare -a simConfigs=(
	"settings/terrainRLImitate/TRPO/Cliff_BayesianSurprise_IncludeStats.json"
#
	"settings/terrainRLImitate/TRPO/Precipice_BayesianSurprise_IncludeStats.json"
#
	"settings/terrainRLImitate/TRPO/Treadmill_NoPhase_BayesianSurprise.json"
#
	"settings/terrainRLImitate/TRPO/Flat_NoPhase_Forward_Reward_BayesianSurprise_Bonus.json"
	"settings/terrainRLImitate/TRPO/Flat_Walk_Forward_Reward_BayesianSurprise_Bonus2.json" ## This version if SMiRL without some initial imitation data.
	"settings/terrainRLImitate/TRPO/Falt_Walk_Forward_BayesianSurprise_IncludeStats_VAE.json" ## Using a VAE
	"settings/terrainRLImitate/TRPO/Flat_NoPhase_Forward_Reward_v0.json" ## regular reward
	
)

rounds=500
opts='--print_level=hyper_train --num_rounds=500 -p 8 --on_policy=fast --run_mode=ec2 --log_comet=true --bootstrap_sample=6 --meta_sim_sample=6 --random_seed=444 --meta_sim_threads=6 --reset_on_fall=false --experiment_logging="{\"use_comet\":true,\"project_name\":\"bayesiansurprise\"}"'
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

