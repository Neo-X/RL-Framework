#!/bin/bash
## This script is designed to make it easier to start a number of simulation

borgy submit --restartable --req-cores=32 --req-ram-gbytes=64 -w /home/${USER} --image=images.borgy.elementai.net/glen:new -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -- /bin/bash -c 'pushd /home/glen/playground/RL-Framework; python3 run_tests.py settings/hyperParamTuning/run_tests.json | tee -a $BORGY_JOB_ID.out'
