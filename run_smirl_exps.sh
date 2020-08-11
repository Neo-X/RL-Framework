#!/bin/bash


## Cliff experiments
params='--print_level=hyper_train  -pde=local_docker --log_comet=true --bootstrap_sample=1 --print_level=hyper_train --experiment_logging="{\"use_comet\": true, \"project_name\": \"bayesiansurprise\"}"'

exps

python doodad_trainModel.py --config=settings/terrainRLImitate/TRPO/Cliff_BayesianSurprise_ICM.json $params
python doodad_trainModel.py --config=settings/terrainRLImitate/TRPO/Cliff_BayesianSurprise_RND.json -pde=local_docker --log_comet=true --bootstrap_sample=1 --experiment_logging="{\"use_comet\": true, \"project_name\": \"bayesiansurprise\"}"
python doodad_trainModel.py --config=settings/terrainRLImitate/TRPO/Cliff_BayesianSurprise_IncludeStats.json -pde=local_docker --log_comet=true --bootstrap_sample=1 --experiment_logging="{\"use_comet\": true, \"project_name\": \"bayesiansurprise\"}"
python doodad_trainModel.py --config=settings/terrainRLImitate/TRPO/Cliff_BayesianSurprise_IncludeStats_VAE.json -pde=local_docker --log_comet=true --bootstrap_sample=1 --experiment_logging="{\"use_comet\": true, \"project_name\": \"bayesiansurprise\"}"

python doodad_trainModel.py --config=settings/terrainRLImitate/TRPO/Cliff_BayesianSurprise_ICM.json -pde=local_docker --log_comet=true --bootstrap_sample=1 --experiment_logging="{\"use_comet\": true, \"project_name\": \"bayesiansurprise\"}"
python doodad_trainModel.py --config=settings/terrainRLImitate/TRPO/Cliff_BayesianSurprise_RND.json -pde=local_docker --log_comet=true --bootstrap_sample=1 --experiment_logging="{\"use_comet\": true, \"project_name\": \"bayesiansurprise\"}"
python doodad_trainModel.py --config=settings/terrainRLImitate/TRPO/Cliff_BayesianSurprise_IncludeStats.json -pde=local_docker --log_comet=true --bootstrap_sample=1 --experiment_logging="{\"use_comet\": true, \"project_name\": \"bayesiansurprise\"}"

python doodad_trainModel.py --config=settings/terrainRLImitate/TRPO/Cliff_BayesianSurprise_ICM.json -pde=local_docker --log_comet=true --bootstrap_sample=1 --experiment_logging="{\"use_comet\": true, \"project_name\": \"bayesiansurprise\"}"
python doodad_trainModel.py --config=settings/terrainRLImitate/TRPO/Cliff_BayesianSurprise_RND.json -pde=local_docker --log_comet=true --bootstrap_sample=1 --experiment_logging="{\"use_comet\": true, \"project_name\": \"bayesiansurprise\"}"
python doodad_trainModel.py --config=settings/terrainRLImitate/TRPO/Cliff_BayesianSurprise_IncludeStats.json -pde=local_docker --log_comet=true --bootstrap_sample=1 --experiment_logging="{\"use_comet\": true, \"project_name\": \"bayesiansurprise\"}"
