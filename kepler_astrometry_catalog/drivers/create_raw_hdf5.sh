#!/bin/bash
#SBATCH --account=kmpardo_1034
#SBATCH --partition=gpu
#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=16G   # memory per CPU core
#SBATCH --array=0-17
#SBATCH --output=../slurm/create_raw_hdf5.out
#SBATCH --error=../slurm/create_raw_hdf5.err
#SBATCH --mail-user=zhangben@usc.edu
#SBATCH --mail-type=END,FAIL

conda activate kac
python3 create_raw_hdf5.py all_207617 ../data/raw-hdf5/all_207617/all_207617_quarter_$SLURM_ARRAY_TASK_ID.hdf5 --quarters $SLURM_ARRAY_TASK_ID