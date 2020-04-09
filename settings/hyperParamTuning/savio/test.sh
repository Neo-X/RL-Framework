#!/bin/bash
#SBATCH --account=rrg-vandepan
#SBATCH --mem=1024M
#SBATCH --time=00:01:00
#SBATCH --job-name=test
#SBATCH --output=%x-%j.out
#SBATCH --cpus-per-task=2
# The arguments should always start with the keyword SBATCH
# The arguments must immediately follow the first line (i.e. `!/bin/bash`)
echo 'Hello, world!'  # use your command here
