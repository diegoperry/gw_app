#!/bin/bash

# Run make_clean.py with given inputs
# python make_clean.py "brightestnonsat100_rot" --fitsfn="../results/cleaned_centroids/brightestnonsat100_rot.fits" --read-fits=0 --write-cleaned-data=1 --max-nstars=100 --fake-signal --mc=5.e9 --dl=0.0001
python make_clean.py "brightestnonsat100_rot" --fitsfn="../results/cleaned_centroids/brightestnonsat100_rot.fits" --read-fits="0" --write-cleaned-data="0" --max-nstars="100" --pca="0"



# # Run create_thekla_input.py
# python create_thekla_input.py "brightestnonsat100_rot" --fake-signal

# # Copy files to thekla directory
# cp ../results/for_thekla/* ../../thekla/data/