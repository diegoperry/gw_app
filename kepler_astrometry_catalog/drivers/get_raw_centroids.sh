#!/bin/bash
#SBATCH --account=kmpardo_1034
#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=16G   # memory per CPU core
#SBATCH --array=0-15
#SBATCH --output=../slurm/%A_%a.out
#SBATCH --mail-user=kmpardo@usc.edu
#SBATCH --mail-type=ALL

conda activate kac

python3 get_raw_centroids.py
