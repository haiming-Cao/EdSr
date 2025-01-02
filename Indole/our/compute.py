import numpy as np
from numpy import ndarray

from einops import repeat, rearrange

def __remove_boundary(coords: ndarray, boundary: ndarray) -> ndarray:
    """remove the boundary for MSD conputation. do NOT use this function in other function.

    Args:
        coords (ndarray): trajectory of Molecule, shape: [ntrajs, nAtoms, nDims].
        boundary (ndarray): boundary of box. shape: [2, nDims]

    Returns:
        ndarray: return a new trajectory without boundary
    """

    ntrajs, natoms, ndims = coords.shape

    blo, bhi = np.split(boundary, [1], axis = 0)
    blen = bhi - blo

    distance = np.diff(coords, prepend = coords[0:1], axis = 0)
    check = np.zeros_like(distance)
    check = np.where(distance >= blen / 2, check - 1, check)
    check = np.where(distance <= -blen / 2, check + 1, check)
    check = np.cumsum(check, axis = 0)

    coords = coords + check * blen 

    return coords

def extraction(folder: str, data_file: str, global_file: str, filter: str = None):
    id_filter = None
    type_filter = None
    
    if filter is not None:
        ftype, *group = filter.split(" ")
        match ftype:
            case 'id':
                id_filter = group
            case 'type':
                type_filter = group
            case _:
                raise ValueError

    properties: dict = np.load(folder + '/' + global_file)
    data: dict = np.load(folder + '/' + data_file)

    mass, id, atom_type = data.get('mass', None), data.get('id', None), data.get('atom_type', None)
    
    delta_t = properties['basis_timestep'] * properties['ntimestep']
    heads = properties['properties_head']
    # print(heads)
    x, v, ppties = data['x'], data['v'], np.array(data['properties'], dtype = np.float32)

    init_state = (x[0], v[0], atom_type, id, mass[atom_type])
    last_state = (x[-1], v[-1], atom_type, id, mass[atom_type])

    if id_filter is not None and id is not None:
        st, en = id_filter
        mask = np.logical_and(id >= int(st), id <= int(en))

        x, v, atom_type, id = x[:, mask], v[:, mask], atom_type[mask], id[mask]

    
    if type_filter is not None and atom_type is not None:
        st, en = type_filter
        mask = np.logical_and(atom_type >= int(st), atom_type <= int(en))
        x, v, atom_type, id = x[:, mask], v[:, mask], atom_type[mask], id[mask]



    # attn check for pbc
    boundary = properties['boundary']

    blo, bhi = boundary
    blen = bhi - blo

    x = np.where(x < blo, x + blen, x)
    x = np.where(x < bhi, x, x - blen)

    return x, v, delta_t, mass[atom_type], atom_type, id, boundary, heads, ppties, init_state, last_state


def compute_COM(x: ndarray, mass: ndarray) -> ndarray:
    """compute center of mass of molecule

    Args:
        x (ndarray): coordinate of Molecule, shape: [nMols, nAtoms, nDims]
        mass (ndarray): mass related to each index, shape: [nMols, nAtoms]

    Returns:
        ndarray: center of mass for each molecule, shape: [nMols, nDims]
    """

    ntrajs, natoms, ndims = x.shape
    if len(mass.shape) == 1:
        mass = repeat(mass, "natoms -> ntrajs natoms", ntrajs = ntrajs)

    sum_of_invmass = 1. / np.sum(mass, axis = -1, keepdims = True)

    mass = repeat(mass, "ntrajs natoms -> ntrajs natoms ndims", ndims = ndims)

    com = np.sum(mass * x, axis = 1) * sum_of_invmass

    return com

def compute_RG(x: ndarray, mass: ndarray, boundary: ndarray | None = None) -> ndarray:
    """compute radius of gyration

    Args:
        x (ndarray): coordinate of Molecule, shape: [nMols, nAtoms, nDims]
        mass (ndarray): mass related to each index, shape: [nMols, nAtoms] or [nAtoms,]

    Returns:
        ndarray: radius of gyration for each molecule, shape: [nMols, nDims]
    """

    if boundary is not None:
        x = __remove_boundary(x, boundary = boundary)

    ntrajs, natoms, ndims = x.shape
    if len(mass.shape) == 1:
        mass = repeat(mass, "natoms -> ntrajs natoms", ntrajs = ntrajs)

    com = compute_COM(x, mass)

    com = rearrange(com, "ntrajs ndims -> ntrajs 1 ndims")

    sum_of_mass = np.sum(mass, axis = -1)

    rsq_m = np.sum((x - com) ** 2, axis = -1) * mass

    Rg_sq = np.sum(rsq_m, axis = -1) / sum_of_mass

    Rg = np.sqrt(Rg_sq)

    return Rg

def compute_MSD(trajs: ndarray, t2: int = 0, boundary: None | ndarray = None):
    """compute Mean Square Displacement

    Args:
        trajs (ndarray): trajectory of Molecule, shape: [ntrajs, nAtoms, nDims].
        t2 (int, optional): _description_. Defaults to 0.
        boundary (None | ndarray, optional): boundary of box. Defaults to None.

    Returns:
        ndarray: return a array of mean square displacement
    """
    
    ntrajs, natoms, ndims = trajs.shape

    if boundary is not None:
        trajs = __remove_boundary(trajs, boundary = boundary)
        
    rt2 = trajs[t2]

    diff = np.sum((trajs - rt2) ** 2, axis = -1)

    diff = np.mean(diff, axis = -1)

    return diff

def compute_RMSD(trajs: ndarray, mass: ndarray, t2: int = 0, boundary: ndarray | None = None) -> ndarray:
    """compute Root Mean Square Deviation\n
    $$
    RMSD(t_1,t_2) ~=~ \\left[\\frac{1}{M} \\sum_{i=1}^N m_i \\|{\\bf r}_i(t_1)-{\\bf r}_i(t_2)\\|^2 \\right]^{\\frac{1}{2}}
    $$

    Args:
        trajs (ndarray): coordinate of Molecule, shape: [ntrajs, nAtoms, nDims].
        mass (ndarray): mass related to each index, shape: [nAtoms,].
        t2 (int, optional): index of frame selected. Defaults to 0.

    Returns:
        ndarray: radius of gyration for each molecule, shape: [ntrajs,]
    """
    assert len(mass.shape) == 1 and len(trajs.shape) == 3

    if boundary is not None:
        trajs = __remove_boundary(trajs, boundary = boundary)

    ntrajs, natoms, ndims = trajs.shape
    
    rt2 = trajs[t2]

    sum_of_mass = np.sum(mass, axis = -1)

    mass = rearrange(mass, "natoms -> 1 natoms")

    diff = np.sum((trajs - rt2) ** 2, axis = -1) * mass / sum_of_mass

    diff = np.sum(diff, axis = -1) ** 0.5

    return diff

def traj_abs_diff(src : ndarray, src_boundary: ndarray, dst: ndarray, dst_boundary: ndarray):

    if src_boundary is not None:
        src = __remove_boundary(src, boundary = src_boundary)
    
    if dst_boundary is not None:
        dst = __remove_boundary(dst, boundary = dst_boundary)

    return np.fabs(src - dst)