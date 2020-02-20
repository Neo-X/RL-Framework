#!/bin/bash
#SBATCH --account=co_rail
#SBATCH --partition=savio3_2080ti
#SBATCH --nodes=1
#SBATCH --constraints=8rtx
#SBATCH --mem=16384M
#SBATCH --time=24:00:00
#SBATCH --job-name=test
#SBATCH --output=%x-%j.out
#SBATCH --cpus-per-task=8

#SBATCH --mail-user=gberseth@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,ALL

# The arguments should always start with the keyword SBATCH
# The arguments must immediately follow the first line (i.e. `!/bin/bash`)
$1  # use your command here
