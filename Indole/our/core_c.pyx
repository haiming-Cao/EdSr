# distutils: language=c++

# cython: language_level=3 

import numpy as np
cimport numpy as cnp

cimport cython
from cython.parallel import prange, parallel

from libc.string cimport memcpy

from ctypes import *

from lammps import PyLammps, lammps

from tqdm import tqdm

cnp.import_array()


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

# """
# thermo_style = [
#     # energy
#     "pe", # total potential energy
#     "ke", # kinetic energy
#     "etotal", # total energy (pe + ke)
#     "evdwl", # van der Waals pairwise energy (includes etail)
#     "ecoul", # Coulombic pairwise energy
#     "epair", # pairwise energy (evdwl + ecoul + elong)
#     "ebond", # bond energy
#     "eangle", # angle energy
#     "edihed", # dihedral energy
#     "eimp", # improper energy
#     "emol", # molecular energy (ebond + eangle + edihed + eimp)
#     "elong", # long-range kspace energy
#     "etail", # van der Waals energy long-range tail correction
#     "enthalpy", # enthalpy (etotal + press*vol)
#     "ecouple", # cumulative energy change due to thermo/baro statting fixes
#     "econserve", # pe + ke + ecouple = etotal + ecouple
#     # properties
#     "atoms", # number of atoms
#     "temp", # temperature
#     "press", # pressure
#     "vol", # volume
#     "density", # mass density of system
#     lx,ly,lz = box lengths in x,y,z
# ]
# """

thermo_style = [
    'custom', 'step', 'time', 'spcpu',
    'temp', 'press',
    'pe', 'ke',
    'enthalpy', 'evdwl', 'ecoul', 'epair',
    'ebond', 'eangle', 'edihed',
    'elong', 'etail', 'emol',
    'ecouple', 'econserve', 'etotal',
    'lx', 'ly', 'lz',
]

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double[:, :] gradientFunction(
    Lammps: PyLammps, 
    cnp.ndarray[cnp.double_t, ndim = 2] Position
):
    """
    define the gradient function
    """
    x = Lammps.lmp.numpy.extract_atom('x')

    cdef double[:,:] x_view = x
    cdef int natoms = x.shape[0], ndims = x.shape[1], i, j
    # x: np.ndarray = llnp.extract_atom('x')
    for i in range(natoms):
        for j in range(ndims):
            x_view[i, j] = Position[i, j]

    # before_id = llnp.extract_atom('id').copy()

    Lammps.run(0, "pre yes post no")

    # after_id = llnp.extract_atom('id')

    # assert (before_id == after_id).all(), 'array has been changed !!!!!'

    # cdef cnp.ndarray force = llnp.extract_atom('f').copy()
    cdef double[:, :] force = Lammps.lmp.numpy.extract_atom('f')

    return force.copy()

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef tuple compute_Taylor_notqdm(
    Lammps: PyLammps, 
    cnp.ndarray coord,
    cnp.ndarray velocity,
    cnp.ndarray mass, 
    double Dt, 
    int maxIter, 
    cnp.ndarray[cnp.double_t, ndim = 2] boundary, 
    cnp.ndarray[cnp.npy_bool, ndim = 1] matrix,
    bint disable_tqdm,
):
    """
    core function
    """
    cdef cnp.ndarray[cnp.double_t, ndim = 2] x_out = coord.copy(), v_out = coord.copy()

    cdef double[:, :] x_view = coord
    cdef double[:, :] v_view = velocity
    cdef double[:] mass_view = mass
    cdef double[:, :] x_out_view = x_out
    cdef double[:, :] v_out_view = v_out

    cdef double[:, :] x_in_lmp = Lammps.lmp.numpy.extract_atom('x')
    cdef double[:, :] f_in_lmp = Lammps.lmp.numpy.extract_atom('f')

    cdef double[3] blo = boundary[0]
    cdef double[3] bhi = boundary[1]
    cdef cnp.npy_bool[:] matrix_view = matrix

    cdef int natoms = coord.shape[0], ndims = coord.shape[1], i, j

    cdef double ftm2v = ftm2v_coeff[Lammps.system.units]
    # cdef double ftm2v = 1.0 / 48.88821291 / 48.88821291
    cdef double Dtsq = Dt * Dt

    cdef int n
    cdef double xcoeff, vcoeff, dv
    
    for n in range(maxIter, 0, -1):
        xcoeff = 2.0 * <double>(n)
        vcoeff = 2.0 * <double>(n)

        # * compute displacement
        # for i in prange(natoms, nogil = True, schedule = 'dynamic'):
        #     for j in range(ndims):
        #         x_in_lmp[i, j] = x_out_view[i, j]
        Lammps.run(0, "pre yes post no")

        with nogil, parallel():
            for i in prange(natoms):
                if matrix_view[i] == 0:
                    continue
                for j in range(ndims):
                    x_out_view[i, j] = x_view[i, j] + v_view[i, j] * Dt / (xcoeff - 1) + f_in_lmp[i, j] * ftm2v * Dtsq * (1./(xcoeff - 1) - 1./xcoeff) / mass_view[i]
                    
                    # attn periodical condition
                    if x_out_view[i, j] < blo[j]:
                        x_out_view[i, j] += bhi[j] - blo[j]
                    elif x_out_view[i, j] >= bhi[j]:
                        x_out_view[i, j] -= bhi[j] - blo[j]

                    x_in_lmp[i, j] = v_out_view[i, j]

        # * compute velocity
        # for i in prange(natoms, nogil = True, schedule = 'dynamic'):
        #     for j in range(ndims):
        #         x_in_lmp[i, j] = v_out_view[i, j]
        Lammps.run(0, "pre yes post no")

        with nogil, parallel():
            for i in prange(natoms):
                if matrix_view[i] == 0:
                    continue
                for j in range(ndims):
                    dv = matrix_view[i] * ftm2v / mass_view[i] * f_in_lmp[i, j] * Dt / (vcoeff - 1)
                    if n > 1:
                        v_out_view[i, j] = (x_view[i, j] + (v_view[i, j] + dv) * Dt / (vcoeff - 2))
                        # attn periodical condition
                        if v_out_view[i, j] < blo[j]:
                            v_out_view[i, j] += bhi[j] - blo[j]
                        elif v_out_view[i, j] >= bhi[j]:
                            v_out_view[i, j] -= bhi[j] - blo[j]
                    
                    else:
                        v_out_view[i, j] = v_view[i, j] + dv
                    
                    x_in_lmp[i, j] = x_out_view[i, j]


        # attn periodical condition
        # * [blo, bhi) has been desrcibed in LAMMPS
        # for i in prange(natoms, nogil = True):
        #     for j in range(ndims):
        #         lo = blo[j]
        #         hi = bhi[j]
        #         if x_out_view[i, j] < lo:
        #             x_out_view[i, j] += hi - lo
        #         elif x_out_view[i, j] >= hi:
        #             x_out_view[i, j] -= hi - lo

        #         if n > 1:
        #             if v_out_view[i, j] < lo:
        #                 v_out_view[i, j] += hi - lo
        #             elif v_out_view[i, j] >= hi:
        #                 v_out_view[i, j] -= hi - lo

    return x_out, v_out

# @cython.wraparound(False)
# @cython.boundscheck(False)
# @cython.cdivision(True)
# cdef Tuple[cnp.ndarray, cnp.ndarray] compute_Taylor(
#     Lammps: PyLammps, 
#     cnp.ndarray coord,
#     cnp.ndarray velocity,
#     cnp.ndarray mass, 
#     double Dt, 
#     int maxIter, 
#     cnp.ndarray[cnp.double_t, ndim = 2] boundary, 
#     cnp.ndarray[cnp.npy_bool, ndim = 1] matrix,
#     bint disable_tqdm = False,
# ):
#     """
#     core function
#     """
#     x_out, v_out = coord.copy(), coord.copy()

#     cdef double[:, :] x_view = coord
#     cdef double[:, :] v_view = velocity
#     cdef double[:] mass_view = mass
#     cdef double[:, :] x_out_view = x_out
#     cdef double[:, :] v_out_view = v_out

#     cdef double[3] blo = boundary[0]
#     cdef double[3] bhi = boundary[1]
#     cdef double[:] matrix_view = matrix

#     cdef int natoms = coord.shape[0], ndims = coord.shape[1], i, j

#     # cdef double ftm2v = ftm2v_coeff[Lammps.system.units]
#     cdef double ftm2v = 1.0 / 48.88821291 / 48.88821291
#     cdef double Dtsq = Dt * Dt

    
#     cdef double[:, :] xn_grad, vn_grad

#     cdef int n
#     cdef double xcoeff, vcoeff, dv, coe, lo, hi

#     with tqdm(total = maxIter, desc = 'taylor Iteration: ', leave = False, position = 1, disable = disable_tqdm) as taylor_bar:
#         for n in range(maxIter, 0, -1):
#             xcoeff = 2.0 * <double>(n)
#             vcoeff = 2.0 * <double>(n)

#             # * compute displacement
#             xn_grad = gradientFunction(Lammps, x_out_view)
#             # dx = v * Dt + massinv * xn_grad * Dtsq / xcoeff
#             # xn = x + dx / (xcoeff - 1)
#             # xn = x + v * Dt / (xcoeff - 1) + massinv_Dtsq * (1./(xcoeff - 1) - 1./xcoeff) * xn_grad

#             for i in range(natoms):
#                 if matrix_view[i] == 0:
#                     continue

#                 coe = ftm2v * Dtsq * (1./(xcoeff - 1) - 1./xcoeff) / mass_view[i]
#                 for j in range(ndims):

#                     x_out_view[i, j] = x_view[i, j] + v_view[i, j] * Dt / (xcoeff - 1) + xn_grad[i, j] * coe
#             # xn = x + v * Dt / (xcoeff - 1) + matrix * ftm2v / mass * Dtsq * xn_grad * (1./(xcoeff - 1) - 1./xcoeff)
            
#             # * compute velocity
#             vn_grad = gradientFunction(Lammps, v_out_view)

#             for i in range(natoms):
#                 if matrix_view[i] == 0:
#                     continue
#                 for j in range(ndims):

#                     dv = ftm2v / mass_view[i] * vn_grad[i, j] * Dt / (vcoeff - 1)
#                     if n > 1:
#                         v_out_view[i, j] = (x_view[i, j] + (v_view[i, j] + dv) * Dt / (vcoeff - 2))
#                     else:
#                         v_out_view[i, j] = v_view[i, j] + dv
                    
#             # dv = massinv * vn_grad * Dt / (vcoeff - 1)
#             # vn = (x + (v + dv) * Dt / (vcoeff - 2)) if n > 1 else (v + dv)

#             # attn periodical condition
#             # * [blo, bhi) has been desrcibed in LAMMPS
#             for j in range(ndims):
#                 lo = blo[j]
#                 hi = bhi[j]
#                 for i in range(natoms):
#                     if matrix_view[i] == 0:
#                         continue
#                     if x_out_view[i, j] < lo:
#                         x_out_view[i, j] += hi - lo
#                     elif x_out_view[i, j] >= hi:
#                         x_out_view[i, j] -= hi - lo

#                     if n > 1:
#                         if v_out_view[i, j] < lo:
#                             v_out_view[i, j] += hi - lo
#                         elif v_out_view[i, j] >= hi:
#                             v_out_view[i, j] -= hi - lo

#             taylor_bar.update()
    
#     return x_out, v_out



def execute(
    Lammps: PyLammps, 
    Dt: float, 
    maxIter: int,
    disable_tqdm: bool = False, 
    scale: bool = False
) -> None:
    """
    execute a step of the whole taylor algorithm
    """
    
    # get the initial state of system
    lmpX, lmpV = Lammps.lmp.numpy.extract_atom('x'), Lammps.lmp.numpy.extract_atom('v')
    lmpMass, atomtype = Lammps.lmp.numpy.extract_atom('mass'), Lammps.lmp.numpy.extract_atom('type')
    
    # set the shielding matrix, True means that gradient set to 0., False is the opposite, shielding matrix is 1-D tensor
    # * set gradient of atoms of id < 3
    shielding_matrix = np.ones((lmpX.shape[0],), dtype = np.bool_)
    shielding_matrix[atomtype < 3] = 0.

    # mass = rearrange(lmpMass[atomtype], 'l -> l 1')
    mass = lmpMass[atomtype]

    coord, velocity = lmpX.copy().astype(np.double), lmpV.copy().astype(np.double)
    boundary = np.array([
        [Lammps.system.xlo, Lammps.system.ylo, Lammps.system.zlo],
        [Lammps.system.xhi, Lammps.system.yhi, Lammps.system.zhi]
    ], dtype = np.double)

    # newX, newV = compute_Taylor(Lammps, coord, velocity, mass.astype(np.double), Dt, maxIter, boundary, shielding_matrix, disable_tqdm = disable_tqdm)
    newX, newV = compute_Taylor_notqdm(Lammps, coord, velocity, mass.astype(np.double), Dt, maxIter, boundary, shielding_matrix, disable_tqdm = disable_tqdm)

    lmpX[:], lmpV[:] = newX, newV
    
    if scale:
        Lammps.velocity('indole scale 700.0')

    return 

def VelocityVerlet(
    Lammps: PyLammps, 
    basis_timestep: float, 
    ntimestep: int, 
    disable_tqdm: bool = False
) -> None:

    from einops import rearrange

    def execute(Lammps: PyLammps, Dt: float):
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

def create_simulation(timestep = 0.2, cmdargs = None, num_threads: int = 1, ensemble: str = 'nve'):

    lmp = lammps(cmdargs=cmdargs)
    
    MDsimulation = PyLammps(ptr = lmp)

    MDsimulation.enable_cmd_history = True
    if num_threads > 1:
        MDsimulation.package(f"omp {num_threads} neigh yes")
        MDsimulation.suffix('omp')
    
    MDsimulation.units('real')
    MDsimulation.atom_style('full')

    MDsimulation.pair_style('lj/cut/coul/long 12.0')
    MDsimulation.bond_style('harmonic')
    MDsimulation.angle_style('harmonic')
    MDsimulation.dihedral_style('opls')
    MDsimulation.improper_style('cvff')

    MDsimulation.dielectric(1.0)
    MDsimulation.pair_modify('mix arithmetic')
    MDsimulation.special_bonds('lj/coul 0.0 0.0 1.0')
    
    MDsimulation.read_data('../lmps/data.lammps')

    MDsimulation.atom_modify('sort 0 0.0') # turn off sort algorithm

    MDsimulation.set('type 1 charge -0.55')
    MDsimulation.set('type 2 charge 1.1')

    MDsimulation.group('zeo type 1 2 ')
    MDsimulation.group('indole type 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18')
    
    MDsimulation.kspace_style('pppm', 1e-4)

    MDsimulation.neighbor('10.0 bin')
    MDsimulation.neigh_modify('every 1 delay 0 check yes exclude molecule/intra zeo')

    MDsimulation.delete_bonds('zeo multi')

    MDsimulation.velocity('indole create 700.0 902144 dist gaussian')
    MDsimulation.velocity('zeo set 0.0 0.0 0.0')
    MDsimulation.velocity('indole scale 700.0')

    # if ensemble == 'nvt':
    #     MDsimulation.fix(f'1 indole nvt temp 700.0 700.0 0.01')
    #     # MDsimulation.fix("1 indole temp/rescale 1 700.0 700.0 0.01 1.0")
    #     # MDsimulation.fix("2 indole nve")
    # elif ensemble == 'nve':
    #     MDsimulation.fix('1 indole nve')
    if ensemble == 'nvt':
        MDsimulation.fix('1 indole langevin 700.0 700.0 0.01 902144 tally no')

    MDsimulation.fix('2 indole nve')

    MDsimulation.compute('1 indole temp')

    MDsimulation.thermo_modify('lost/bond ignore')

    MDsimulation.thermo_style(' '.join(thermo_style))

    MDsimulation.timestep(timestep) # attn set timestep

    # initialize system state
    MDsimulation.run(0, 'pre yes post no')

    MDsimulation.enable_cmd_history = False

    return MDsimulation