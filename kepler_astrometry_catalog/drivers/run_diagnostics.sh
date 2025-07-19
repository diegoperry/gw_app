#!/bin/bash

for i in {1..18}
do
    python lomb_scargle_diagnostics.py $i
done