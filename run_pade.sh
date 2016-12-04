#!/bin/bash -l
#PBS -l nodes=2:ppn=24
#PBS -l walltime=01:00:00
#PBS -m abe
#PBS -q gpu
#PBS -d .

LAUNCHER='mpirun --bind-to none -np 2 -npernode 1' $REGENT_PATH/regent.py pade.rg -ll:cpu 1 -ll:csize 32768
