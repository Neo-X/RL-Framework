#!/bin/bash
while true
do
    # gsutil -m rsync -r ./ gs://rl-framework-cluster-bucket/playground/RL-Framework/
    gsutil -m rsync -r ./terrainRLSim/ gs://rl-framework-cluster-bucket/playground/RL-Framework/terrainRLSim/
    gsutil -m rsync -r ./RLSimulations/ gs://rl-framework-cluster-bucket/playground/RL-Framework/RLSimulations/
    sleep 3600 ### 10 minutes
done
