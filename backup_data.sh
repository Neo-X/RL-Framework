#!/bin/bash

echo "rsyncing data"
# gsutil -m rsync -r ./ gs://rl-framework-cluster-bucket/playground/RL-Framework/
gsutil -m rsync -r ./terrainRLSim/ gs://rl-framework-cluster-bucket/playground/RL-Framework/terrainRLSim/
gsutil -m rsync -r ./RLSimulations/ gs://rl-framework-cluster-bucket/playground/RL-Framework/RLSimulations/
gsutil -m rsync -r ./GymMultiChar/ gs://rl-framework-cluster-bucket/playground/RL-Framework/GymMultiChar/
