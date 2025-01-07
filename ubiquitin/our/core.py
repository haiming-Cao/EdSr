# coding: utf-8
from typing import Callable, Tuple

from einops import rearrange
import numpy as np
from lammps import IPyLammps, PyLammps, lammps

from tqdm import tqdm

ftm2v_coeff = {
    'lj': 1.0,
    'real': 1.0 / 48.88821291 / 48.88821291,
    'metal': 1.0 / 1.0364269e-4,
    'si': 1.0,
    'cgs': 1.0,
    'electron': 0.937582899,
    'micro': 1.0,
    'nano': 1.0,
}

def env_preset(MDsimulation: PyLammps):

    MDsimulation.units('real')
    MDsimulation.atom_style('full')

    MDsimulation.pair_style('lj/cut/coul/cut 22.0')
    MDsimulation.bond_style('harmonic')
    MDsimulation.angle_style('cosine/squared')
    MDsimulation.dihedral_style('fourier')
    MDsimulation.improper_style('harmonic')

    MDsimulation.dielectric(15.0)
    MDsimulation.pair_modify('shift yes')
    MDsimulation.special_bonds('lj/coul 0.0 1.0 1.0')
    
    MDsimulation.read_data('../lmps/data/nve_protein.data')

    MDsimulation.group('protein id 1:163')

    MDsimulation.neighbor('10.0 bin')
    

def create_simulation(thermo_ouput: list, infile: str | None = None, timestep = 0.2, cmdargs = None, num_threads: int = 1, ensemble: str = 'nve') -> IPyLammps:

    lmp = lammps(cmdargs=cmdargs)
    
    MDsimulation = PyLammps(ptr = lmp)

    MDsimulation.enable_cmd_history = True
    if num_threads > 1:
        MDsimulation.package(f"omp {num_threads} neigh yes")
        MDsimulation.suffix('omp')

    if infile is None:
        env_preset(MDsimulation)
    else:
        MDsimulation.file(infile)

    if ensemble == 'nvt':
        raise NotImplementedError
    elif ensemble == 'nve':
        MDsimulation.fix('1 all nve')
    # MDsimulation.fix('eqfix all nvt temp 300.0 300.0 100.0')

    MDsimulation.atom_modify('sort 0 0.0') # turn off sort algorithm
        
    MDsimulation.thermo(1)

    MDsimulation.thermo_modify('lost/bond ignore')

    MDsimulation.thermo_style(' '.join(thermo_ouput))

    MDsimulation.timestep(timestep) # attn set timestep

    # initialize system state
    MDsimulation.run(0, 'pre yes post no')

    MDsimulation.enable_cmd_history = False

    return MDsimulation


def gradientFunction(Lammps: IPyLammps | PyLammps, Position: np.ndarray) -> np.ndarray:
    """
    define the gradient function
    """
    llnp = Lammps.lmp.numpy

    x: np.ndarray = llnp.extract_atom('x')
    x[:] = Position

    before_id = llnp.extract_atom('id').copy()

    Lammps.run(0, 'pre yes post no');

    after_id = llnp.extract_atom('id')

    assert (before_id == after_id).all(), 'array has been changed !!!!!'

    force: np.ndarray = llnp.extract_atom('f')

    return force.copy()



def compute_EdSr(
    Lammps: IPyLammps, 
    SystemState: Tuple[np.ndarray, np.ndarray], 
    Dt: float, 
    maxIter: int, 
    gradient_func: Callable,
    boundary: np.ndarray, 
    shielding_matrix: np.ndarray = None,
    disable_tqdm: bool = False,
) -> Tuple[np.ndarray, np.ndarray, ]:
    """
    core function
    """
    x, v, mass = SystemState

    matrix = np.ones(x.shape)
    
    if shielding_matrix is not None:
        matrix[shielding_matrix] = 0.
        v[shielding_matrix] = 0.

    ftm2v = ftm2v_coeff[Lammps.system.units]
    massinv = matrix * ftm2v / mass

    Dtsq = Dt * Dt

    massinv_Dtsq = massinv * Dtsq

    blo, bhi = boundary
    blen = bhi - blo
    
    xn = x.copy()
    vn = x.copy()
    with tqdm(total = maxIter, desc = 'EdSr Iteration: ', leave = False, position = 1, disable = disable_tqdm) as edsr_bar:
        for n in range(maxIter, 0, -1):
            xcoeff = 2.0 * n
            vcoeff = 2.0 * n

            # * compute displacement
            xn_grad = gradient_func(Lammps, xn)

            # dx = v * Dt + massinv * xn_grad * Dtsq / xcoeff
            # xn = x + dx / (xcoeff - 1)
            xn = x + v * Dt / (xcoeff - 1) + massinv_Dtsq * (1./(xcoeff - 1) - 1./xcoeff) * xn_grad

            # * compute velocity
            vn_grad = gradient_func(Lammps, vn)

            dv = massinv * vn_grad * Dt / (vcoeff - 1)
            vn = (x + (v + dv) * Dt / (vcoeff - 2)) if n > 1 else (v + dv)

            # attn periodical condition
            # * [blo, bhi) has been desrcibed in LAMMPS
            xn = np.where(xn < blo, xn + blen, xn)
            xn = np.where(xn < bhi, xn, xn - blen)
            if n > 1:
                vn = np.where(vn < blo, vn + blen, vn)
                vn = np.where(vn < bhi, vn, vn - blen)

            edsr_bar.update()
    
    return xn, vn



def execute(
    Lammps: PyLammps | IPyLammps, 
    Dt: float, 
    maxIter: int, 
    disable_tqdm: bool = False, 
) -> None:
    """
    execute a step of the whole EdSr algorithm
    """
    
    # get the initial state of system
    lmpX, lmpV = Lammps.lmp.numpy.extract_atom('x'), Lammps.lmp.numpy.extract_atom('v')
    lmpMass, atomtype = Lammps.lmp.numpy.extract_atom('mass'), Lammps.lmp.numpy.extract_atom('type')
    
    # set the shielding matrix, True means that gradient set to 0., False is the opposite, shielding matrix is 1-D tensor
    # * set gradient of atoms of id < 3
    # shielding_matrix = atomtype < 3
    shielding_matrix = None

    mass = rearrange(lmpMass[atomtype], 'l -> l 1')

    SystemState = (lmpX.copy(), lmpV.copy(), mass)
    boundary = np.array([
        [Lammps.system.xlo, Lammps.system.ylo, Lammps.system.zlo],
        [Lammps.system.xhi, Lammps.system.yhi, Lammps.system.zhi]
    ])
    
    newX, newV = compute_EdSr(Lammps, SystemState, Dt, maxIter, gradientFunction, boundary, shielding_matrix, disable_tqdm = disable_tqdm)

    lmpX[:], lmpV[:] = newX, newV

    return 

def VelocityVerlet(
    Lammps: PyLammps | IPyLammps,
    basis_timestep: float, 
    ntimestep: int, 
    disable_tqdm: bool = False
) -> None:

    def execute(Lammps: PyLammps | IPyLammps, Dt: float):
        # get the initial state of system
        lmpX, lmpV = Lammps.lmp.numpy.extract_atom('x'), Lammps.lmp.numpy.extract_atom('v')
        lmpMass, atomtype = Lammps.lmp.numpy.extract_atom('mass'), Lammps.lmp.numpy.extract_atom('type')

        # copy variables
        x_0, v_0 = Lammps.lmp.numpy.extract_atom('x').copy(), Lammps.lmp.numpy.extract_atom('v').copy()
        mass = rearrange(lmpMass[atomtype], "l -> l 1")
        
        # keep the atoms of boundary stationary
        matrix = np.ones_like(x_0)
        matrix[atomtype < 3] = 0.
        v_0[atomtype < 3] = 0.

        ftm2v = ftm2v_coeff[Lammps.system.units]
        massinv = matrix * ftm2v / mass
        Dt_half = Dt * 0.5

        # get gradient at t by using Lammps API and compute v(t + Dt/2)
        force = gradientFunction(Lammps, x_0)
        v_ht = v_0 + massinv * force * Dt_half

        # compute r(t + Dt)
        x_t = x_0 + v_ht * Dt

        # get gradient at t + Dt by using Lammps API and compute v(t + Dt)
        force = gradientFunction(Lammps, x_t)
        v_t = v_ht + massinv * force * Dt_half

        # put x_t, v_t into source address
        lmpX[:] = x_t; lmpV[:] = v_t

    with tqdm(total = ntimestep, desc = 'vv Iteration: ', leave = False, position = 1, disable = disable_tqdm) as vv_bar:
        for _ in range(ntimestep):
            execute(Lammps, basis_timestep)
            vv_bar.update()

    return



