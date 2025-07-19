#!/bin/bash
#SBATCH --account=kmpardo_1034
#SBATCH --time=0:15:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=4G   # memory per CPU core
#SBATCH --output=../slurm/%A.out
#SBATCH --mail-user=yijunw@caltech.edu 
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load python

python3 plot_centroid_samples.py
