#!/bin/bash

# SBATCH --account=kmpardo_1034
# SBATCH --time=48:00:00   # walltime
# SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
# SBATCH --nodes=1   # number of nodes
# SBATCH --mem-per-cpu=16G   # memory per CPU core
# SBATCH --output=../slurm/%A.out
# SBATCH --mail-user=kmpardo@usc.edu
# SBATCH --mail-type=ALL

# export TMPDIR=/scratch1/kmpardo/

# module load python

# python3 lomb_scargle_diagnostics.py 12


# Set the parameters
SAMPLEID="brightestnonsat10000_rot"
QUARTER="12"
FITSFN="../results/cleaned_centroids/brightestnonsat10000_rot_12.fits"
OUTPATH="../results/lomb_scargle/"
NSTARS=-1
EXTNAME="RESIDUALS"
DISTANCEAVE='False'

# Run the script with the parameters
python lomb_scargle_diagnostics.py \
    "${SAMPLEID}" \
    "${OUTPATH}" \
    "${FITSFN}" \
    --n_stars="${NSTARS}" \
    --extname="${EXTNAME}" \
    --distanceave="${DISTANCEAVE}" \
