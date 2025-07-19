#!/bin/bash

#SBATCH --account=kmpardo_1034
#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=16G   # memory per CPU core
#SBATCH --output=../slurm/%A_%a.out
#SBATCH --array=3-4
#SBATCH --mail-user=yijunw@caltech.edu # kmpardo@usc.edu
#SBATCH --mail-type=ALL

export TMPDIR=/scratch1/kmpardo/

module load python

python3 get_lightcurves.py
