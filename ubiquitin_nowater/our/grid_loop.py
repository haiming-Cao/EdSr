# coding: utf-8
import sys,argparse,os
from typing import Callable, Dict, Tuple, Literal, Union

from einops import rearrange
import numpy as np
from lammps import IPyLammps, PyLammps, lammps

import logging

from tqdm import tqdm

from core import create_simulation, execute, VelocityVerlet
# from core_c import create_simulation, execute, VelocityVerlet

logging.basicConfig(
    level = logging.INFO,
    # format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s'
)

from mpi4py import MPI
comm = MPI.COMM_WORLD

np.set_printoptions(threshold = np.inf)


def print_state(Lammps: IPyLammps | PyLammps) -> None:
    '''
    print system state, function name: print_state
    '''
    llnp = Lammps.lmp.numpy

    x = llnp.extract_atom('x')
    mass = llnp.extract_atom('mass')
    v = llnp.extract_atom('v')
    id = llnp.extract_atom('id')
    force = llnp.extract_atom('f')
    print(f'{x = }')
    print(f'{mass = }')
    print(f'{v = }')
    print(f'{id = }')
    print(f'{force = }')
    print(f'{x.shape = }, {force.shape = }, {id.shape = }, {mass.shape = }')

def get_copy_current_state(Lammps: IPyLammps | PyLammps, properties_head: list[str]):
    llnp = Lammps.lmp.numpy
    a = ['x', 'v', 'id', 'mass', 'type']
    basis_ppties = [llnp.extract_atom(ppty.lower()).copy() for ppty in a]
    other_ppties = [Lammps.lmp.get_thermo(ppty.lower()) for ppty in properties_head]
    return *basis_ppties, other_ppties 

def get_state(Lammps: PyLammps, state: str):
    llnp = Lammps.lmp.numpy
    match state.lower():
        case 'x' | 'mass' | 'v' | 'id' | 'f' | 'type':
            data = llnp.extract_atom(state.lower())
        case 'pe' | 'ke':
            data = Lammps.lmp.get_thermo(state.lower())
    return data

def check_pbc(Lammps: PyLammps, coords: np.ndarray) -> np.ndarray:

    blo = np.array([Lammps.system.xlo, Lammps.system.ylo, Lammps.system.zlo])
    bhi = np.array([Lammps.system.xhi, Lammps.system.yhi, Lammps.system.zhi])
    blen = bhi - blo

    # [blo, bhi) is desrcibed in LAMMPS
    coords = np.where(coords < blo, coords + blen, coords)
    coords = np.where(coords < bhi, coords, coords - blen)
    return coords

def save_data(path: str, start: int, length: int, /, **kwargs) -> None: # [start, start + len)
    np.savez(f'{path}/frames{start}_{start + length}.npz', **kwargs)
    return

def get_simulation(
    basis_timestep: float, 
    ntimestep: int, 
    cmdargs: list, 
    ntrajs: int, 
    path: str,
    properties_head: list,
    /,
    split: int = 1000,
    *,
    drop_last = False,
    num_threads: int = 16,
    simulation: IPyLammps | PyLammps | None = None, 
    keep_state: bool = False,
    ensemble: str = 'nve',
    mode: Literal['control', 'taylor', 'benchmark', 'vv'] = 'benchmark', 
    maxIter: int = 400,
    disable_tqdm: bool = False,
    scale: bool = False,
    prerun_step: int = 2,
    thermo: int = 20,
) -> PyLammps | None:
    
    xtrajs, vtrajs, ppties = [], [], []
    thermo_values = []

    if simulation is None:
        step = basis_timestep if prerun_step > 0 or mode == 'benchmark' or mode == 'vv' else basis_timestep * ntimestep
        simulation = create_simulation(step, cmdargs = list(cmdargs), num_threads = num_threads, ensemble = ensemble)

    cmd_history = ['LAMMPS Setting up:'] + simulation._cmd_history
    logging.info('\n\n' + '\n- '.join(cmd_history) + '\n')

    boundary = np.array([
        [simulation.system.xlo, simulation.system.ylo, simulation.system.zlo],
        [simulation.system.xhi, simulation.system.yhi, simulation.system.zhi]
    ])

    logging.info('\n' + '\n - '.join(simulation.lmp_info('system')) + '\n')
    logging.info('\n' + '\n -  '.join(simulation.lmp_info('communication')) + '\n')

    # thermo head output with format
    thermo_style = list(simulation.lmp.last_thermo().keys())
    logging.info((' ' + f'''{"frame":>{len(str(ntrajs) * 2) + 1}}''' + '{:>15}' * len(thermo_style)).format(*thermo_style))

    # save global constant params, such as boundary
    np.savez(
        f'{path}/GlobalVariable.npz', 
        boundary = boundary, 
        mode = mode,
        ntimestep = ntimestep, 
        basis_timestep = basis_timestep, 
        maxIter = maxIter,
        prerun_step = prerun_step,
        ensemble = ensemble,
        cmdargs = cmdargs,
        properties_head = properties_head,
        thermo_style = thermo_style,
    )

    init_x, init_v, init_id, init_mass, init_type, ppty = get_copy_current_state(simulation, properties_head)
    xtrajs.append(init_x); vtrajs.append(init_v)
    ppties.append(ppty)

    thermo_output: dict = simulation.lmp.last_thermo()

    # if thermo_output.get('Step', None) is not None:
    #     thermo_output['Step'] = 0
        
    thermo_val = list(thermo_output.values())
    thermo_values.append(thermo_val)
    format_thermo_val = ''.join(map(lambda x: '{:>15}' if isinstance(x, int) else '{:>15.2f}', thermo_val)).format(*thermo_val)

    logging.info(f''' {f"{1}/{ntrajs}":>{len(str(ntrajs) * 2) + 1}}{format_thermo_val}''')

    start = 2
    try:
        with tqdm(total = ntrajs, desc = 'generating trajectories: ', position = start, disable = disable_tqdm) as run_bar:
            
            if prerun_step > 0 and mode != 'benchmark' and mode != 'vv':

                for idx in range(start, start + prerun_step + 1):
                    simulation.run(ntimestep, 'pre yes post no')
                    x, v, id, mass, atype, ppty = get_copy_current_state(simulation, properties_head)
                    xtrajs.append(x); vtrajs.append(v)
                    ppties.append(ppty)

                    run_bar.update()
                    if disable_tqdm and idx % thermo == 0:

                        thermo: dict = simulation.lmp.last_thermo()
                        # if thermo.get('Step', None) is not None:
                        #     thermo['Step'] = idx
                        
                        thermo_val = list(thermo.values())
                        thermo_values.append(thermo_val)
                        format_thermo_val = ''.join(map(lambda x: '{:>15}' if isinstance(x, int) else '{:>15.2f}', thermo.values())).format(*thermo_val)

                        logging.info(f''' {f"{idx}/{ntrajs}":>{len(str(ntrajs) * 2) + 1}}{format_thermo_val}''')

                    # save data
                    if len(xtrajs) == split and len(vtrajs) == split and len(ppties) == split:
                        logging.info(f'saving {path}/frames{idx - split}_{idx + 1}.npz......')
                        np.savez(
                            f'{path}/frames{idx - split}_{idx + 1}.npz', 
                            x = np.stack(xtrajs, axis = 0),
                            v = np.stack(vtrajs, axis = 0),
                            properties = ppties,
                            id = id,
                            mass = mass,
                            atom_type = atype,

                        )
                        xtrajs.clear(); vtrajs.clear(); ppties.clear(); thermo_values.clear()
                else:
                    start += prerun_step
                    simulation.timestep(basis_timestep * ntimestep)
                    logging.info(f'prerun has finished and already has run {prerun_step} step')

                    # clear up the cache
                    simulation.runs.clear()

            for idx in range(start, ntrajs + 1):
                if mode == 'benchmark':
                    simulation.run(ntimestep, 'pre yes post no')
                elif mode == 'control':
                    simulation.run(1, 'pre yes post no')
                elif mode == 'taylor':
                    execute(simulation, basis_timestep * ntimestep, maxIter, disable_tqdm = disable_tqdm, scale = scale)
                elif mode == 'vv':
                    VelocityVerlet(simulation, basis_timestep, ntimestep, disable_tqdm = disable_tqdm)
                
                x, v, id, mass, atype, ppty = get_copy_current_state(simulation, properties_head)

                xtrajs.append(x); vtrajs.append(v); ppties.append(ppty)

                assert (init_id == id).all() and (init_type == atype).all()
                
                if disable_tqdm and idx % thermo == 0:

                    thermo_output: dict = simulation.lmp.last_thermo()
                    # if thermo_output.get('Step', None) is not None:
                    #     thermo_output['Step'] = idx
                    
                    thermo_val = list(thermo_output.values())
                    thermo_values.append(thermo_val)
                    format_thermo_val = ''.join(map(lambda x: '{:>15}' if isinstance(x, int) else '{:>15.2f}', thermo_output.values())).format(*thermo_val)

                    logging.info(f''' {f"{idx}/{ntrajs}":>{len(str(ntrajs) * 2) + 1}}{format_thermo_val}''')

                # save data
                if len(xtrajs) == split and len(vtrajs) == split and len(ppties) == split:
                    logging.info(f'saving {path}/frames{idx - split + 1}_{idx}.npz......')
                    np.savez(
                        f'{path}/frames{idx - split + 1}_{idx}.npz', 
                        x = np.stack(xtrajs, axis = 0), 
                        v = np.stack(vtrajs, axis = 0), 
                        properties = ppties,
                        id = id,
                        mass = mass,
                        atom_type = atype,
                    )
                    xtrajs.clear(); vtrajs.clear(); ppties.clear()

                run_bar.update()

                # clear up the cache
                simulation.runs.clear()
            else:
                if idx % thermo != 0:
                    thermo_output: dict = simulation.lmp.last_thermo()
                    # if thermo_output.get('Step', None) is not None:
                    #     thermo_output['Step'] = ntrajs
                    thermo_val = list(thermo_output.values())
                    thermo_values.append(thermo_val)
                    format_thermo_val = ''.join(map(lambda x: '{:>15}' if isinstance(x, int) else '{:>15.2f}', thermo_output.values())).format(*thermo_val)
                    logging.info(f''' {f"{ntrajs}/{ntrajs}":>{len(str(ntrajs) * 2) + 1}}{format_thermo_val}''')

            if not drop_last and len(xtrajs) > 0 and len(vtrajs) > 0 and len(ppties) > 0:
                
                logging.info(f'saving {path}/frames{ntrajs - len(xtrajs)}_{ntrajs}.npz......')
                np.savez(
                    f'{path}/frames{ntrajs - len(xtrajs)}_{ntrajs}.npz', 
                    x = np.stack(xtrajs, axis = 0),
                    v = np.stack(vtrajs, axis = 0),
                    properties = ppties,
                    id = id,
                    mass = mass,
                    atom_type = atype,
                )
                xtrajs.clear(); vtrajs.clear(); ppties.clear()

    except Exception as e:
        if not drop_last and len(xtrajs) > 0 and len(vtrajs) > 0 and len(ppties) > 0:
            logging.error(f'raise Exception, saving {path}/frames{idx - len(xtrajs)}_{idx + 1}.npz......')
            np.savez(
                f'{path}/frames{idx - len(xtrajs)}_{idx + 1}.npz', 
                x = np.stack(xtrajs, axis = 0),
                v = np.stack(vtrajs, axis = 0),
                properties = ppties,
                id = id,
                mass = mass,
                atom_type = atype,
            )
            xtrajs.clear(); vtrajs.clear(); ppties.clear()

        logging.error("something wrong", exc_info = True)
        raise
        
    
    try:
        thermo_values = np.stack(thermo_values, axis = 0)
        np.save(f'{path}/thermo.npy', thermo_values)
        
    except Exception as e:
        logging.error((f"something wrong, thermo can not convert to ndarray and save to npy file"), exc_info = True)
        raise

    if not keep_state:
        simulation.close()
        return

    return simulation



parser = argparse.ArgumentParser()
parser.add_argument('--ntimestep', type = int, help = 'interval', default = 100)
parser.add_argument('--basis', type = float, help = 'timestep of bbenchmark', default = 0.01)
parser.add_argument('--maxiter', type = int, help = 'taylor iteration', default = None)
parser.add_argument('--ntrajs', type = int, help = 'frames', default = 100)
parser.add_argument('--mode', type = str, help = 'benchmark, control, taylor, vv', default = 'benchmark')
parser.add_argument('--en', type = str, help = 'ensemble: nvt, nve', default = 'nve')
parser.add_argument('--scale', type = int, help = 'scale: 0, ~0 mean False, True in python, respectively', default = 0)
parser.add_argument('--prerun_step', type = int, help = 'prerun_step >= 0, integer', default = 0)
parser.add_argument('--prefix', type = str, help = 'prefix of file name', default = None)
parser.add_argument('--thermo', type = int, help = 'positive integer. similar to the LAMMPS thermo command', default = 20)
parser.add_argument('--split', type = int, help = 'number of frames saving to each npz file, non-positive number means the total trajectory will be save into a npz file', default = 10000)
parser.add_argument('--drop_last', type = int, help = 'whether drop the last group of which frames will not be divided by split args or not. 0, ~0 denoted that False, True respectively.', default = 0)
# option
parser.add_argument('--savepath', type = str, help = 'path of saving file', default = None)
parser.add_argument('--fdname', type = str, help = 'saving folder name', default = None)
parser.add_argument('--debug', type = int, help = 'use a group of debugging params by default', default = 0)
args = parser.parse_args()

# mode
mode: Literal['benchmark', 'control', 'taylor', 'vv'] = args.mode.lower()

# need to get properties
properties_head = [
    'temp', # temperature
    'press', # pressure
    "pe", # total potential energy
    "ke", # kinetic energy
    "etotal", # total energy (pe + ke)
    "evdwl", # van der Waals pairwise energy (includes etail)
    "ecoul", # Coulombic pairwise energy
    "epair", # pairwise energy (evdwl + ecoul + elong)
    "ebond", # bond energy
    "eangle", # angle energy
    "edihed", # dihedral energy
    "eimp", # improper energy
    "emol", # molecular energy (ebond + eangle + edihed + eimp)
    "elong", # long-range kspace energy
    "etail", # van der Waals energy long-range tail correction
    "enthalpy", # enthalpy (etotal + press*vol)
    "ecouple", # cumulative energy change due to thermo/baro statting fixes
]

debug = True if args.debug != 0 else False

if not debug:

    # MD setting
    cmdargs        = ["-log", "none"] # args: https://docs.lammps.org/latest/Run_options.html 
    basis_timestep = args.basis
    ntimestep      = args.ntimestep
    ensemble       = args.en
    thermo         = 20 if args.thermo <= 0 else args.thermo

    # taylor params setting
    maxIter        = args.maxiter # attn taylor equation number of order
    ntrajs         = args.ntrajs
    Delta_t        = basis_timestep * ntimestep
    scale          = True if bool(args.scale) and mode == 'taylor' and ensemble == 'nvt' else False
    prerun_step    = args.prerun_step

    # saving data setting
    folder         = args.fdname
    savepath       = args.savepath
    prefix         = args.prefix
    split          = args.split
    drop_last      = args.drop_last

else:
    # mode
    logging.info('debugging......')
    mode = 'taylor'
    
    # MD setting
    cmdargs        = ["-log", "none"] # args: https://docs.lammps.org/latest/Run_options.html 
    basis_timestep = 0.01
    ntimestep      = 100
    ensemble       = 'nve'
    thermo         = 1

    # taylor params setting
    maxIter        = 500 # attn taylor equation number of order
    ntrajs         = 100
    Delta_t        = basis_timestep * ntimestep
    scale          = False
    prerun_step    = 0

    # saving data setting
    folder         = None
    savepath       = 'data/'
    prefix         = 'debug'
    split          = 10
    drop_last      = False



assert scale == False, 'Not Implemented Error'


# visualization setting
disable_tqdm = True




if savepath is None:
    savepath = 'data/'
    if not os.path.exists('./data/'):
        os.mkdir('./data')
    logging.info(f'because savepath is not given, the save path will use {savepath} by default')
else:
    assert os.path.exists(savepath), "savepath don't exists"

if folder is None:
    folder = f"{mode}_{ensemble}{'_prerun' if mode != 'benchmark' and prerun_step > 0 else ''}{'_scale' if (mode == 'taylor' or mode == 'vv') and scale == True and ensemble == 'nvt' else ''}_basis{basis_timestep}_intv{ntimestep}" + (f'_iter{maxIter}' if maxIter is not None and mode == 'taylor' else '')
    logging.info(f'fdname argument given is None, folder name will be initialized to {folder}')

if prefix not in ['', None]:
    logging.info(f'concat the prefix and folder name -> {prefix + folder}')
    folder = prefix + '_' + folder


if os.path.exists(savepath + folder):
    logging.info(f'{savepath}/{folder} has been existed')
    num = 1
    while True:
        if not os.path.exists(f"{savepath}/{folder}_{num}"):
            folder = folder + f'_{num}'
            break
        num += 1

logging.info(f'{savepath}/{folder} directory will be created to save data')
os.mkdir(f'{savepath}/{folder}')

common_dict_params = dict(
    split        = split,
    drop_last    = drop_last,
    mode         = mode,
    disable_tqdm = disable_tqdm,
    ensemble     = ensemble,
    prerun_step  = prerun_step,
    thermo       = thermo,
)

try:
    match mode:
        case 'benchmark' | 'control' | 'vv':
            ppties = get_simulation(basis_timestep, ntimestep, cmdargs, ntrajs, f'{savepath}/{folder}/', properties_head, **common_dict_params)

        case 'taylor':
            ppties = get_simulation(basis_timestep, ntimestep, cmdargs, ntrajs, f'{savepath}/{folder}/', properties_head, **common_dict_params, scale = scale)

        case _:
            logging.error(f"mode should be (benchmark, control, taylor), but got {mode}")
            raise

    logging.info('program has done')

except Exception as e:
    logging.error("something wrong", exc_info = True)
    raise Exception