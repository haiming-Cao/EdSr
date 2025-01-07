#!/bin/bash

# activate your conda env
source ~/miniconda3/etc/profile.d/conda.sh
source activate lammps

# set omp thread
export OMP_NUM_THREADS=16

bash_pid=$$


# path of default arguments
jsonfile="params.json"

ntimestep=30

# benchmark timestep
basis=1.0 
# EdSr equation number of order
maxIter=500
# number of frames
ntrajs=20000 
# choose one in ['benchmark', 'control', 'EdSr', 'vv']
mode="EdSr" 
# condition, only support nve condition so far.
ensemble="nve" 
# before run MD or EdSr, you can set this value to run "benchmark" timestep.
prerun_step=0 
# positive integer. similar to the LAMMPS thermo command.
thermo=200 
# 0, ~0 mean False, True in python, respectively.
# Taking split argument is 100 and drop_last argument is 1 for example, if you run 105 step, the last 5 step will be dropped.
drop_last=0 
# number of frames saving to each npz file, non-positive number means the total trajectory will be save into a npz file
split=10000 
# if you do not want to write basical setting of your simulation in the core.py, you can provide path of env_set.lammps.
# Except you understand how the program run, don't write some commands in your env_set.lammps (details in README.md).
lmpfile="env_set.lammps" 

logpath="log"
# prefix=""

exec 2>&1>"${logpath}/${mode}_${ensemble}_basis${basis}_intv${ntimestep}_frames${ntrajs}_${bash_pid}.log"

# nohup python -u grid_loop.py --params $jsonfile --basis $basis    --ntimestep   $ntimestep   --maxiter $maxIter \
#                              --mode   $mode     --en    $ensemble --prerun_step $prerun_step --ntrajs  $ntrajs \
#                              --thermo $thermo   --split $split    --drop_last   $drop_last   --lmpfile $lmpfile &

nohup python -u grid_loop.py --params $jsonfile &

py_pid=$!
echo 
echo "Current Bash ID: ${bash_pid}"
echo "Python Process ID: ${py_pid}"
echo 