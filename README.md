<div align="center">
    <h1>EdSr: A Novel End-to-End Approach for State-Space Sampling in Molecular Dynamics Simulation</h1>
     <p align="center"> 
        Hai-Ming Cao<sup>1</sup> · <a href="https://scholar.google.com/citations?hl=en&user=iOIRgyAAAAAJ">Bin Li</a><sup>1*</sup>
    </p>
    <p align="center"> 
        <b>School of Chemical Engineering and Technology, Sun Yat-Sen University, Zhuhai 519082, China</b>
    </p>
    <p align="center"> 
    <sup>*</sup> E-mail: libin76@mail.sysu.edu.cn
    </p>
    </p>  
    <a href="https://github.com/haiming-Cao/EdSr/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/haiming-Cao/EdSr?style=flat&label=License&color=blue">
    </a>
    <a href="https://arxiv.org/abs/2412.20978">
    <img alt="Static Badge" src="https://img.shields.io/badge/Arxiv-2412.20978-green">
    </a>
    <a href="https://arxiv.org/abs/2412.20978">
    <img alt="Static Badge" src="https://img.shields.io/badge/Supplementary-8A2BE2">
    </a>
</a>

</div>

## Overview

> The molecular dynamics (MD) simulation technique has been widely used in complex systems, but the time scale is limited due to the small timestep.  Here, we propose a novel method, named Exploratory dynamics Sampling with recursion (EdSr),  which an be used in MD simulation with flexible timestep, inspired by Langevin dynamics,  Stochastic Differential Equation and Taylor expansion formula.  By setting up four groups of experiments including simple function, ideal physical model, all-atom simulation and coarse-grained simulation,  we demonstrate that EdSr can dynamically and flexibly adjust the simulation timestep according to requirements during simulation period, and can work with larger timestep than the widely used velocity-Verlet integrator. Although this method can not perform perfectly at flexible timestep with all simulation systems, we believe that it will be a promising approach in the future.


## 🛠️ Requirements

> [!IMPORTANT] 
> PYTHON version >= 3.11

Third-Party Package|Version|
|:-:|:-:|
|numpy|>=1.26.0|
|scipy|>=1.11.3|
|einops|>=0.6.1|
|tqdm|>=4.65.0|
|matplotlib|>=3.7.2|
|lammps|==2024.4.17|
|mpi4py|>=3.1.6|

## 📖 Method

> [!TIP]
> If you set the value N smaller such as 3 and then expand the formula, you will find that this is a Taylor formula.

Assume that the initial state is ($X_t$, $X_t'$) and the next state ($X_{t + \Delta t}$, $X_{t + \Delta t}'$). The part of displacement of EdSr can be rewritten as the following form:

$$
X_{n-1} = X_N + \frac{1}{2n-1} \Big(X'_N\Delta t - \frac{1}{2n}\frac{\nabla_X U(X_n)}{M}(\Delta t)^{2}\Big), \quad n\ \rm{for}\ N\ to\ 1 
$$

where $X_0$, $X_N$, $X_N'$ denote $X_{t + \Delta t}$, $X_t$, $X'_t$ respectively. According to the definition of derivative, the part of velocity of EdSr can be expressed as:

$$
X_{n-1} =  X_N + \frac{1}{2n-2} \Big(X'_N\Delta t - \frac{1}{2n-1}\frac{\nabla_X U(X_n)}{M}(\Delta t)^{2}\Big), \quad n\ \rm{for}\ N\ to\ 2
$$

$$
X'_0 =  X'_N - \frac{\nabla_X U(X_1)}{M}\Delta t,  \quad  n = 1
$$

where $X_{0}'$ denotes $X_{t + \Delta t}'$. 

## 📈 Results

> [!NOTE] 
> In this section, we only show figures for each experiment. if you are interested in our work, you can get to know from our [paper](https://arxiv.org/abs/2412.20978) and [supplementary](https://arxiv.org/abs/2412.20978).

<!-- <div class="admonition note">
<p class="admonition-title">In this section, we only show figures for each experiment. if you are interested in our work, you can get to know from <a href="https://arxiv.org/abs/2412.20978">our paper</a> and <a href="https://arxiv.org/abs/2412.20978">supplementary</a></p>
</div> -->

### Equation

![sinx_1st](./Images/Equation/sinx.svg) | ![exp](./Images/Equation/exp.svg)
|-|-|

### ideal spring

![init_spr](./Images/Spring/init.svg)| ![cur_spr](./Images/Spring/cur_0.1_10.0.svg)
|-|-|

### ideal pendulum

![init_pen](./Images/Pendulum/init.svg) | ![cur_pen](./Images/Pendulum/cur_0.2_0.6.svg)
|-|-|

### two-body
    the first experiment
![init_twobody](./Images/TwoBody/init.svg)

    the second experiment
![cur_twobody](./Images/TwoBody/cur_0.5_1.0.svg)

### Indole
    1.0 fs (top) and 3.0 fs (bottom)
![coord_rmsd_1.0_3.0](./Images/indole/indole_coord_rmsd_1.0_3.0.svg)
![ke_epair_emol_1.0_3.0](./Images/indole/indole_ke_epair_emol_1.0_3.0.svg)
![rdf_dist_1.0_3.0](./Images/indole/indole_rdf_vdist_1.0_3.0.svg)

### ubiquitin
    10.0 fs (top) and 20.0 fs (bottom)
![coord_rmsd_rg](./Images/ubiquitin/protein_coord_rmsd_rg_10.0_20.0.svg)
![ke_epair_emol](./Images/ubiquitin/protein_ke_epair_emol_10.0_20.0.svg)
![rdf_dist](./Images/ubiquitin/protein_rdf_vdist_10.0_20.0.svg)


## 🚀 Usage

### Pre-learn

* **PYTHON With LAMMPS in** [tutorial for installation](https://docs.lammps.org/Python_install.html)

* **Gromacs** in [tutorial](https://www.gromacs.org/tutorial_webinar.html)

### Environment set up
We highly recommend Conda because all of our experiments are run under [Miniconda](https://docs.anaconda.com/miniconda/) environment

Before performing every experiment, run the following command to build your Conda environment:
```bash
conda create -n lammps python=3.11.5
```
and then run command `conda activate lammps` to test your environment.

### Examples
We provide two choices for each experiment, you can choose one of them to run experiment.

**For Equation**:
1. Use [jupyter notebook](https://jupyter.org/) to run .ipynb file directly. (*you can use conda to install the jupyter extension. Alternatively, you can install extension in VScode*)
2. Install jupyter extension and use command `jupyter nbconvert --to script *.ipynb` to tranform `*.ipynb` file to `*.py` file. Then you can use command `python *.py` to visualize your data.
3. Edit Equation/Equation to select function that you want. Run command `python Equation.py` (**Equation_thirdOrder.py** file is for $y = x^3$)

**For IdealSpring**:
1. Use [jupyter notebook](https://jupyter.org/) to run .ipynb file directly. (*you can use conda to install the jupyter extension. Alternatively, you can install extension in VScode*)
2. Install jupyter extension and use command `jupyter nbconvert --to script *.ipynb` to tranform `*.ipynb` file to `*.py` file. Then you can use command `python *.py` to visualize your data.
2. run command `python idealSpring.py`

**For IdealPendulum**:
1. Use [jupyter notebook](https://jupyter.org/) to run .ipynb file directly. (*you can use conda to install the jupyter extension. Alternatively, you can install extension in VScode*)
2. Install jupyter extension and use command `jupyter nbconvert --to script *.ipynb` to tranform `*.ipynb` file to `*.py` file. Then you can use command `python *.py` to visualize your data.
2. run command `python idealPendulum.py`

**For twoBody**
1. Use [jupyter notebook](https://jupyter.org/) to run .ipynb file directly. (*you can use conda to install the jupyter extension. Alternatively, you can install extension in VScode*)
2. Install jupyter extension and use command `jupyter nbconvert --to script *.ipynb` to tranform `*.ipynb` file to `*.py` file. Then you can use command `python *.py` to visualize your data.
2. if you want to test your data, you can edit `twoBodies.py` and then run command `python twoBodies.py`.

**For Indole**:

```bash
cd Indole/our
bash EdSr.sh
```
In the `EdSr.sh` bash file, the code is shown as follows:
```bash
#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
source activate lammps

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
prefix="beta"
debug=0 # ~0 denotes use default arguments of debugging

# exec 2>&1>"${mode}_${ensemble}_basis${basis}_scale_intv${ntimestep}_frames${ntrajs}_iter${maxIter}_${bash_pid}.log"
exec 2>&1>"${logpath}/${prefix}_${mode}_${ensemble}_basis${basis}_intv${ntimestep}_frames${ntrajs}_${bash_pid}.log"

# # the first choice to run the program
# nohup python -u grid_loop.py --ntrajs $ntrajs --en      $ensemble --basis  $basis  --ntimestep   $ntimestep \
#                              --split  $split  --lmpfile $lmpfile  --debug  $debug  --prerun_step $prerun_step \
#                              --thermo $thermo --maxiter $maxIter  --mode   $mode   --drop_last   $drop_last \
#                              --prefix $prefix &

# the second choice to run the program
nohup python -u grid_loop.py --params $jsonfile &

py_pid=$!
echo 
echo "Current Bash ID: ${bash_pid}"
echo "Python Process ID: ${py_pid}"
echo 
```
You can edit these variables according to your need. For example, if you want to change timestep, you can edit variables `basis` and `ntimestep` in `EdSr.sh` bash file. Benchmark timestep is `basis`. MD or EdSr timestep is `basis * ntimestep`.

> [!WARNING]
> Except you understand how the program run, don't write the following commands in your env_set.lammps:

|Command||
|:-:|:-:|
|atom_modify|the program has disabled the sort function using command `sort 0 0.0` in `core.py`.|
|fix nve/nvt|you can modify `create_simulation` function in `core.py`.|
|thermo_style |alternatively, you can edit `params.json`.|
|timestep |same as `thermo_style`|
|package / suffix| the program has used  `package omp num_threads neigh yes` and `suffix omp`|
|thermo_modify| we have used `thermo_modify lost/bond ignore` in the program, please clearly understand conflict or restrictions between them.|
|thermo| we have set thermo value to 1 in the program.

*After getting data, there are two choices for data visualization:*
1. Use [jupyter notebook](https://jupyter.org/) to run .ipynb file directly. (*you can use conda to install the jupyter extension. Alternatively, you can install extension in VScode*)
2. Install jupyter extension and use command `jupyter nbconvert --to script *.ipynb` to tranform `*.ipynb` file to `*.py` file. Then you can use command `python *.py` to visualize your data.

**For ubiquitin[_nowater]**
Before running experiment, you need to generate structure file sppuorted by LAMMPS, namely that you either directly data file supported by **LAMMPS** or generate file supported by **GROMACS** firstly, and then tranform **GROMACS** files to LAMMPS files. 
After that, you can run the following command:
```bash
cd ubiquitin[_nowater]/our
bash EdSr.sh
```

In the `EdSr.sh` bash file, the code is shown as follows:
```bash
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

# # the first choice to run the program
# nohup python -u grid_loop.py --params $jsonfile --basis $basis    --ntimestep   $ntimestep   --maxiter $maxIter \
#                              --mode   $mode     --en    $ensemble --prerun_step $prerun_step --ntrajs  $ntrajs \
#                              --thermo $thermo   --split $split    --drop_last   $drop_last   --lmpfile $lmpfile &

# the second choice to run the program
nohup python -u grid_loop.py --params $jsonfile &

py_pid=$!
echo 
echo "Current Bash ID: ${bash_pid}"
echo "Python Process ID: ${py_pid}"
echo 
```

*After getting data, there are two choices for data visualization:*
1. Use [jupyter notebook](https://jupyter.org/) to run .ipynb file directly. (*you can use conda to install the jupyter extension. Alternatively, you can install extension in VScode*)
2. Install jupyter extension and use command `jupyter nbconvert --to script *.ipynb` to tranform `*.ipynb` file to `*.py` file.

> [!TIP]
> if you want to learn more about this work, feel free to send e-mail to <a href="mailto:libin76@mail.sysu.edu.cn">libin76@mail.sysu.edu.cn</a> with your question. we are willing to answer questions about technical details or scientific questions.</p>


## ✨ Star
If you find this work very interesting, please don't hesitate to give us a star🌟. **Thank you so much**

## 🎓 Citation
If you find our work relevant to your research, please cite:
```
ArXiv:
@misc{cao2024edsrnovelendtoendapproach,
      title={EdSr: A Novel End-to-End Approach for State-Space Sampling in Molecular Dynamics Simulation}, 
      author={Hai-Ming Cao and Bin Li},
      year={2024},
      eprint={2412.20978},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/2412.20978}, 
}
```
