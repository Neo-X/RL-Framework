#!/bin/bash
#SBATCH --account=rrg-vandepan
#SBATCH --mem=16384M
#SBATCH --time=12:00:00
#SBATCH --job-name=test
#SBATCH --output=%x-%j.out
#SBATCH --cpus-per-task=8
# The arguments should always start with the keyword SBATCH
# The arguments must immediately follow the first line (i.e. `!/bin/bash`)
$1  # use your command here
