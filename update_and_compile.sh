#!/bin/bash

pushd /opt/efficient-hrl
    git pull origin connected_policies
popd
pushd /opt/RL-Framework
    git pull origin master
popd
pushd /opt/TerrainRLSim
    git pull origin master
    git reset --hard origin/master
    pushd simAdapter/
    ./gen_swig.sh
    popd
        # premake4 gmake
        ### Compile for off-screen rendering
        premake4 --file=premake4_openglES.lua gmake
        pushd gmake
    make config=release64 -j 8
    popd; 
popd; 
pushd /opt/RLSimulationEnvironments
    git pull origin master
    pushd rendering
        premake4 gmake
        make config=release64 -j 2
    popd
popd
echo "Done update and build"
