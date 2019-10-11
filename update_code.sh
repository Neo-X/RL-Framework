#!/bin/bash

pushd /opt/RL-Framework
    git pull origin master
popd
pushd /opt/TerrainRLSim
    git pull origin master
    git reset --hard origin/master
    popd; 
popd; 
pushd /opt/RLSimulationEnvironments
    git pull origin master
popd
echo "Done update"
