#!/bin/bash
###
### Print list of resouce links so you can quickly look through them
###

docker run -e TERRAINRL_PATH=/opt/TerrainRLSim/ -e RLSIMENV_PATH=/opt/RLSimulationEnvironments -it us.gcr.io/glen-rl-framework/glen:latest2 bash -c 'pushd /opt/RL-Framework; ./update_and_compile.sh;'

