import numpy as np
from matplotlib import pyplot as plt
from matplotlib import font_manager

from einops import rearrange

fontsize = font_manager.FontProperties(size = 15)

small_data = np.load('./data/benchmark_nvt_basis0.01_intv100_frames1000_iter500.npz')
big_data = np.load('./data/big_nvt_basis0.01_intv100_frames1000_iter500.npz')
taylor_data = np.load('./data/taylor_nvt_scale_basis0.01_intv1000_frames100_iter500.npz')

src_small_x, src_small_v, small_basis_timestep, small_ntimestep = small_data['x'], small_data['v'], small_data['basis_timestep'], small_data['ntimestep']
src_big_x, src_big_v, big_basis_timestep, big_ntimestep = big_data['x'], big_data['v'], big_data['basis_timestep'], big_data['ntimestep']
src_taylor_x, src_taylor_v, taylor_basis_timestep, taylor_ntimestep = taylor_data['x'], taylor_data['v'], taylor_data['basis_timestep'], taylor_data['ntimestep']

skip = 10

small_x, small_v = src_small_x[::skip], src_small_v[::skip]
big_x, big_v = src_big_x[::skip], src_big_v[::skip]
taylor_x, taylor_v = src_taylor_x, src_taylor_v

ntrajs = src_small_x.shape[0]

maxIter = int(small_data['maxIter'])
big_Delta_t = big_ntimestep * big_basis_timestep
taylor_Delta_t = taylor_ntimestep * taylor_basis_timestep

dx_small_big = np.abs((small_x - big_x) / (small_x + 1e-34))
dx_small_taylor = np.abs((small_x - taylor_x) / (small_x + 1e-34))

dx_small_big = rearrange(dx_small_big, "ntrajs natoms dim -> ntrajs (natoms dim)")
dx_small_taylor = rearrange(dx_small_taylor, "ntrajs natoms dim -> ntrajs (natoms dim)")

dx_small_big = np.mean(dx_small_big, axis = -1)
dx_small_taylor = np.mean(dx_small_taylor, axis = -1)

fig = plt.figure(figsize=[10,4], dpi=200)

plt.xlabel('$T_{time}$', fontproperties = fontsize) ; plt.ylabel(r'$Error = \frac{1}{3N} abs(\frac{predict - label}{label + \epsilon})$', fontproperties = fontsize)
plt.title(f'Displacement Error, interval = {taylor_basis_timestep * taylor_ntimestep} fs,  $t \in [0.0, {small_ntimestep * small_basis_timestep * ntrajs}]$, {maxIter = }', loc = 'center', fontproperties = fontsize)
plt.yscale('log')
plt.scatter(np.arange(ntrajs, step = skip), dx_small_big , s = 15)
plt.scatter(np.arange(ntrajs, step = skip), dx_small_taylor, s = 15)
plt.plot(np.arange(ntrajs, step = skip), dx_small_big, label = f'basis_{small_basis_timestep}fs vs taylor_{big_Delta_t}fs Error')
plt.plot(np.arange(ntrajs, step = skip), dx_small_taylor, label = f'basis_{small_basis_timestep}fs vs Ours_{taylor_Delta_t}fs Error')

plt.legend()
plt.show()
# plt.close()