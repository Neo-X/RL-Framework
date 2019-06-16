#!/bin/bash
#SBATCH --account=fc_rail
#SBATCH --mem=16384M
#SBATCH --time=24:00:00
#SBATCH --job-name=test
#SBATCH --output=%x-%j.out
#SBATCH --cpus-per-task=8
#SBATCH --partition=savio2

#SBATCH --mail-user=gberseth@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,ALL

# The arguments should always start with the keyword SBATCH
# The arguments must immediately follow the first line (i.e. `!/bin/bash`)
$1  # use your command here
