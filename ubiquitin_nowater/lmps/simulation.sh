#!/bin/bash

source /home/caohm/miniconda3/etc/profile.d/conda.sh

conda activate lammps

omp_nthreads=4
nprocessor=8

export OMP_NUM_THREADS=$omp_nthreads

# nohup mpirun -np $nprocessor lmp -in in.lammps &
nohup mpirun -np $nprocessor lmp -in dynamics.lammps &
