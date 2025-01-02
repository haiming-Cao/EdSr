#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
source activate lammps

export OMP_NUM_THREADS=16

bash_pid=$$

ntimestep=400
basis=0.01
maxIter=500 # attn taylor equation number of order
ntrajs=20000
mode="taylor" # ['benchmark', 'control', 'taylor', 'vv']
ensemble="nve"
scale=0 # 0, ~0 mean False, True in python, respectively
prerun_step=0
thermo=500
drop_last=0
split=10000
logpath="log"
prefix="beta"
debug=0

# exec 2>&1>"${mode}_${ensemble}_basis${basis}_scale_intv${ntimestep}_frames${ntrajs}_iter${maxIter}_${bash_pid}.log"
exec 2>&1>"${logpath}/${prefix}_${mode}_${ensemble}_basis${basis}_intv${ntimestep}_frames${ntrajs}_${bash_pid}.log"

nohup python -u grid_loop.py --ntrajs $ntrajs --en      $ensemble --basis $basis --ntimestep   $ntimestep \
                             --split  $split  --debug   $debug    --scale $scale --prerun_step $prerun_step \
                             --thermo $thermo --maxiter $maxIter  --mode  $mode  --drop_last   $drop_last \
                             --prefix $prefix &

py_pid=$!
echo 
echo "Current Bash ID: ${bash_pid}"
echo "Python Process ID: ${py_pid}"
echo 