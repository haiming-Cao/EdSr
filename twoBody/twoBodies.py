from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

import torch
import numpy as np

from einops import rearrange, repeat
from torch import Tensor, einsum
from tqdm import tqdm

from data import *



# G = 6.67e-2 # cm^3*g/s^2
G = 1.0

np.set_printoptions(precision = 11, linewidth = np.inf)




class TwoBody(object):

    def __init__(
        self,
        tStart    : float,
        tStep     : int,
        interval  : float,
        init_state: np.ndarray = None,
        **kwargs
    ) -> None:
        # * set the initial state

        if init_state is None:
            init_state = state_generator(
                nbodies = kwargs['mass'],
                mass = kwargs['mass'], 
                min_radius = kwargs['min_radius'], 
                max_radius = kwargs['max_radius'], 
                orbit_noise = kwargs['orbit_noise']
            )
        orbit, orbit_settings = get_orbit(init_state, t_start = tStart, t_step = tStep, interval = interval, rtol = 3e-14)

        self.nbodies = init_state.shape[0]
        self.init_state: np.ndarray = init_state
        self.orbit: np.ndarray = orbit
        self.orbit_setting: dict = orbit_settings
        self.times = orbit_settings['t_eval']
    def getState(self, idx):
        return self.orbit[idx]
    
    def gradient(self, xn, mass):
        
        m = torch.from_numpy(mass).unsqueeze(dim = -1)
        q = torch.tensor(xn, requires_grad = True)

        potential: Tensor = self.potential(q, m)
        grad = torch.autograd.grad(potential, q)[0]
        
        return -grad.detach().numpy()

    def potential(self, q: Tensor, mass: Tensor):
        natoms = q.shape[0]

        row, col = np.triu_indices(natoms, 1)

        r_ij = (q[row] - q[col]).square().sum(dim = -1).sqrt()
        Uenergy = (mass[col] * mass[row] * G / r_ij).sum()

        return -Uenergy

    # def gradient(self, xn, mass):
    #     return self.get_accelerations(xn, mass)
    
    def compute_classic(self, state, Dt):

        x, v = state[:, 1:3], state[:, 3:5]
        mass = state[:, 0:1]
        massinv = 1. / mass

        acc = self.gradient(x, mass) * massinv

        nextx = x + v * Dt + 0.5 * Dt * Dt * acc 

        new_acc = self.gradient(nextx, mass) * massinv
        nextv = v + (acc + new_acc) * Dt / 2 

        nextState = np.concatenate([mass, nextx, nextv], axis = -1)

        return nextState
        
    # torch 
    # def potential(self, q: Tensor, mass: Tensor):
    #     Uenergy = torch.tensor(0., requires_grad = True)
    #     for i in range(q.shape[0] - 1): # number of bodies
    #         m_i, m_j = mass[i], mass[i+1:] 
    #         q_i = q[i:i+1] 
    #         q_j = q[i+1:]  
    #         displacements = q_i - q_j  
    #         r_ij = displacements.square().sum(dim = -1).sqrt() 
    #         Uenergy = torch.add(Uenergy, (m_j / r_ij * m_i * G).sum(dim = -1))
    #     return -Uenergy
    
    def computeEdSr(self, state, Dt, maxIter, xi: None = None) -> np.ndarray:

        x, v = state[:, 1:3], state[:, 3:5]
        mass = state[:, 0:1]

        massinv = 1. / mass
        xxn = x.copy() if xi is None else xi[:, 1:3]
        xvn = x.copy() if xi is None else xi[:, 1:3]


        Dtsq = Dt * Dt


        for n in range(maxIter, 0, -1):
            vcoeff = 2.0 * n
            # * compute velocity
            vacc = self.gradient(xvn, mass) * massinv
            dxv = vacc * Dt / (vcoeff - 1)
            xvn = (x + (v + dxv) * Dt / (vcoeff - 2)) if n > 1 else (v + dxv)
            if n == 2:
                xxn = xvn

        for n in range(maxIter, 0, -1):
            xcoeff = 2.0 * n
            
            # * compute displacement
            # acc = np.einsum("ij,i->ij", self.gradient(xxn, mass), massinv)
            xacc = self.gradient(xxn, mass) * massinv
            dxx = v * Dt + xacc * Dtsq / xcoeff
            xxn = x + dxx / (xcoeff - 1)


        # mass = rearrange(mass, 'b -> b 1')
        
        nextState = np.concatenate([mass, xxn, xvn], axis = -1)

        return nextState
    
    def computeEdSr_parallel(self, state, Dt, maxIter, xi: None = None) -> np.ndarray:

        mass = state[:, 0:1]

        # attn parallel
        with ThreadPoolExecutor(max_workers = 2) as executor:
            xres = executor.submit(self._computeX, state, Dt, maxIter, xi)
            vres = executor.submit(self._computeV, state, Dt, maxIter, xi)

            xxn, xvn = xres.result(), vres.result()
        
        # attn single thread
        # for n in range(maxIter, 0, -1):

        #     xcoeff = 2.0 * n
        #     vcoeff = 2.0 * n
            
        #     # * compute displacement
        #     # acc = np.einsum("ij,i->ij", self.gradient(xxn, mass), massinv)
        #     acc = self.gradient(xxn, mass) * massinv
        #     dxx = v * Dt + acc * Dtsq / xcoeff
        #     xxn = x + dxx / (xcoeff - 1)

        #     # * compute velocity
        #     # dxv = v * Dt + self.gradient(xvn, mass) * Dtsq / (coeff - 1)
        #     # xvn = (x + dxv / (coeff - 2)) if n > 1 else (dxv / Dt)
        #     # acc = np.einsum("ij,i->ij", self.gradient(xvn, mass), massinv)
        #     acc = self.gradient(xvn, mass) * massinv
        #     dxv = acc * Dt / (vcoeff - 1)
        #     xvn = (x + (v + dxv) * Dt / (vcoeff - 2)) if n > 1 else (v + dxv)
        
        nextState = np.concatenate([mass, xxn, xvn], axis = -1)

        return nextState
    
    def _computeX(self, state, Dt, maxIter, xi = None):

        x, v = state[:, 1:3], state[:, 3:5]
        mass = state[:, 0:1]

        massinv = 1. / mass
        xxn = x.copy() if xi is None else xi[:, 1:3]


        Dtsq = Dt * Dt

        for n in range(maxIter, 0, -1):
            xcoeff = 2.0 * n

            acc = self.gradient(xxn, mass) * massinv
            dxx = v * Dt + acc * Dtsq / xcoeff
            xxn = x + dxx / (xcoeff - 1)
        
        return xxn
    
    def _computeV(self, state, Dt, maxIter, xi = None):

        x, v = state[:, 1:3], state[:, 3:5]
        mass = state[:, 0:1]

        massinv = 1. / mass
        xvn = x.copy() if xi is None else xi[:, 1:3]

        for n in range(maxIter, 0, -1):
            vcoeff = 2.0 * n

            acc = self.gradient(xvn, mass) * massinv
            dxv = acc * Dt / (vcoeff - 1)
            xvn = (x + (v + dxv) * Dt / (vcoeff - 2)) if n > 1 else (v + dxv)

        return xvn
    # attn the first experiment, get f(x + n*dx) by f(x + (n-1)*dx)
    def init_loop(self, maxIter: int):

        trajs: np.ndarray = np.zeros_like(self.orbit)
        ttrajs: np.ndarray = np.zeros_like(self.orbit)

        derror: np.ndarray = np.zeros_like(self.times)
        verror: np.ndarray = np.zeros_like(self.times)
        traderror: np.ndarray = np.zeros_like(self.times)
        traverror: np.ndarray = np.zeros_like(self.times)

        start = self.times[0]
        init = self.getState(0)
        trajs[0] = init
        ttrajs[0] = init

        # * Parallel
        with ProcessPoolExecutor(max_workers = 12) as executor:

            exefunc = partial(self.computeEdSr, init, maxIter = maxIter, xi = None)

            futures = list(tqdm(executor.map(exefunc, self.times[1:] - start), total = trajs.shape[0] - 1))
            

        for i, future in enumerate(futures, start = 1):

            Dt = self.times[i] - start

            # nextState = self.computeEdSr(init, Dt, maxIter, xi = None)
            nextState = future
            tra_nextState = self.compute_classic(init, Dt)

            labelState = self.getState(i)

            trajs[i] = nextState
            ttrajs[i] = tra_nextState
            # attn displacement Mean Square Error
            deps: np.ndarray = MAE(nextState[:, 1:3], labelState[:, 1:3])
            tra_deps: np.ndarray = MAE(tra_nextState[:, 1:3], labelState[:, 1:3])
            # attn velocity Mean Square Error
            veps: np.ndarray = MAE(nextState[:, 3:5], labelState[:, 3:5])
            tra_veps: np.ndarray = MAE(tra_nextState[:, 3:5], labelState[:, 3:5])

            derror[i] = deps
            verror[i] = veps
            traderror[i] = tra_deps
            traverror[i] = tra_veps

            
        return trajs, derror, verror, traderror, traverror

    # attn the second experiment, get f(x_0 + n*dx) by f(x_0)
    def current_loop(self, maxIter: int):

        trajs: np.ndarray = np.zeros_like(self.orbit)
        ttrajs: np.ndarray = np.zeros_like(self.orbit)
        derror: np.ndarray = np.zeros_like(self.times)
        verror: np.ndarray = np.zeros_like(self.times)
        traderror: np.ndarray = np.zeros_like(self.times)
        traverror: np.ndarray = np.zeros_like(self.times)

        start = self.times[0]
        init = self.getState(0)
        trajs[0] = init
        ttrajs[0] = init
        for i in tqdm(range(trajs.shape[0] - 1)):

            Dt = self.times[i + 1] - self.times[i]

            nextState = self.computeEdSr(trajs[i], Dt, maxIter)
            tra_nextState = self.compute_classic(ttrajs[i], Dt)

            labelState = self.getState(i + 1)
            trajs[i + 1] = nextState
            ttrajs[i + 1] = tra_nextState
            # displacement Mean Square Error
            deps: np.ndarray = MAE(nextState[:, 1:3], labelState[:, 1:3])
            tra_deps: np.ndarray = MAE(tra_nextState[:, 1:3], labelState[:, 1:3])
            # velocity Mean Square Error
            veps: np.ndarray = MAE(nextState[:, 3:5], labelState[:, 3:5])
            tra_veps: np.ndarray = MAE(tra_nextState[:, 3:5], labelState[:, 3:5])

            derror[i + 1] = deps
            verror[i + 1] = veps
            traderror[i + 1] = tra_deps
            traverror[i + 1] = tra_veps

        return trajs, derror, verror, traderror, traverror
        
def RMSE(predict: np.ndarray, label: np.ndarray) -> np.ndarray:
    error_each_atom = np.sqrt(np.sum(np.square(predict - label), axis = -1))
    tot_error = np.mean(error_each_atom, axis = -1)
    return tot_error

def MAE(predict: np.ndarray, label: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(predict - label), axis = (-1, -2))


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    nbodies    = 2
    tStart     = 0.
    tStep      = 300
    interval   = 0.05
    mass       = 1.0
    min_radius = 1.0
    max_radius = 2.0
    maxIter    = 500
    orbit_noise = 0.05

    model = TwoBody(
        Nbodies     = nbodies,
        tStart      = tStart,
        tStep       = tStep,
        interval    = interval,
        mass        = mass,
        min_radius  = min_radius,
        max_radius  = max_radius,
        orbit_noise = orbit_noise,
    )

    trajs, derror, verror, traderror, traverror = model.current_loop(maxIter)
    times = model.times
    
    fontsize = font_manager.FontProperties(size = 25)

    eps_max = max(derror.max(), verror.max())

    fig = plt.figure(figsize=[10,3], dpi=200)
    plt.xlabel(r'time(s)', fontproperties = fontsize) ; plt.ylabel(r'error(unitless)', fontproperties = fontsize)
    plt.title(f'MAE compared with RK45', loc = 'center', fontproperties = fontsize)
    plt.axis()
    plt.yscale('log')
    plt.plot(times, derror, label = 'Displacement, Ours vs RK45')
    plt.plot(times, verror, label = 'Velocity, Ours vs RK45')
    plt.plot(times, traderror, label = 'Displacement, Velocity-Verlet vs RK45')
    plt.plot(times, traverror, label = 'Velocity, Velocity-Verlet vs RK45')


    plt.yticks(np.logspace(-8, 2, 6))
    plt.tick_params(axis = 'both', labelsize = 17)
    plt.legend(fontsize = 15, loc = 'lower right')
    plt.show()