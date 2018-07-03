#!/bin/bash
#SBATCH --account=rrg-vandepan
#SBATCH --mem=1024M
#SBATCH --time=0-0:5
#SBATCH --cpus-per-task=2
# The arguments should always start with the keyword SBATCH
# The arguments must immediately follow the first line (i.e. `!/bin/bash`)
echo 'Hello, world!'  # use your command here
