# coding: utf-8
import argparse
import numpy as np

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--intv', type = int, help = 'interval', default = 30)
parser.add_argument('--basis', type = float, help = 'timestep of bbenchmark', default = 1.0)
parser.add_argument('--maxiter', type = int, help = 'taylor iteration', default = None)
parser.add_argument('--gfile', type = str, help = 'path of GlobalVariable.npz', default = 'GlobalVariable.npz')
parser.add_argument('--dfile', type = str, help = 'scale: 0, ~0 mean False, True in python, respectively', default = 0)
parser.add_argument('--format', type = str, help = 'xyz', default = 0)
args = parser.parse_args()


intv = args.intv
basis = args.basis

path = f'data//benchmark_nve_basis{basis}_intv{intv}//'
pathlist = [x for x in path.split('/') if x != '']
name = pathlist[-1]

global_file = args.gfile
data_file = args.dfile
out_format = args.format

properties: dict = np.load(path + global_file)
data: dict = np.load(path + '/' + data_file)


coord, velocity, atype, aid = data['x'], data['v'], data.get('atom_type', None), data.get('id', None)
boundary = properties['boundary']
blo, bhi = boundary

sort_id = np.argsort(aid)

coord, velocity, atype, aid = coord[:, sort_id], velocity[:, sort_id], atype[sort_id], aid[sort_id]

with open(f'{name}.{out_format}', 'w') as f:
    if out_format == 'xyz':
        for i in tqdm(range(coord.shape[0]), desc = 'generating file: '):
            f.write(f'ITEM: TIMESTEP\n{i}\n')
            f.write(f'ITEM: NUMBER OF ATOMS\n{coord.shape[1]}\n')
            f.write(f'ITEM: BOX BOUNDS xx yy zz pp pp pp\n{blo[0]:.8f} {bhi[0]}\n{blo[1]:.8f} {bhi[1]}\n{blo[2]:.8f} {bhi[2]}\n')
            f.write(f'ITEM: ATOMS id type x y z vx vy vz\n')
            for j in range(coord.shape[1]):
                f.write(f'{aid[j]:<5d}{atype[j]:3d}{coord[i,j,0]:11.6f}{coord[i,j,1]:11.6f}{coord[i,j,2]:11.6f}{velocity[i,j,0]:11.6f}{velocity[i,j,1]:11.6f}{velocity[i,j,2]:11.6f}\n')