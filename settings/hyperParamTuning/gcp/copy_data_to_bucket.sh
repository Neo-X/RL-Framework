#!/bin/bash

### copy files to the bucket

gsutil -m rsync -r ~/playground/TerrainRLSim/  gs://rl-framework-cluster-bucket/playground/TerrainRLSim/