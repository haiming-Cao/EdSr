from data import get_trajectory, hamiltonian_fn
import numpy as np

import autograd.numpy as anp
import autograd 

from tqdm import tqdm

def hamilton(q, v):
    return 4* (1 - np.cos(q)) + v**2 / 2

class IdealPendulum(object):

    def __init__(
        self,
        tStart  : float,
        tStep   : int,
        interval: float,
        mass    : float = 1.0,
        radius  : float = 2.0,
        gravity : float = 2.0,
    ) -> None:
        
        self.mass    = mass
        self.radius  = radius
        self.gravity = gravity

        if interval < 0.:
            inv_tStart = tStart + tStep * interval
            inv_interval = -interval
            q, p, dqdt, dpdt, times = get_trajectory(inv_tStart, tStep, inv_interval, m = self.mass, g = self.gravity, l = self.radius)
            # print(dqdt.shape, dpdt.shape)
            q, p, dqdt, dpdt, times = q[::-1], p[::-1], dqdt[:, ::-1], dpdt[:, ::-1], times[::-1]
        else:
            q, p, dqdt, dpdt, times = get_trajectory(tStart, tStep, interval, m = self.mass, g = self.gravity, l = self.radius)
            # print(dqdt.shape, dpdt.shape)
        self.q, self.p, dqdt, dpdt, self.times = q, p, dqdt, dpdt, times
        self.dqdt = np.squeeze(dqdt, 0)
        self.dpdt = np.squeeze(dpdt, 0)


    def getState(self, idx):
        return self.q[idx], self.dqdt[idx]

    def gradient(self, x):
        return -4* np.sin(x)


    def computeTaylor(self, status, Dt, maxIter, xi = None):

        x, v = status

        xxn = x.copy() if xi is None else xi[0]
        xvn = x.copy() if xi is None else xi[0]

        massinv = 1.0 / self.mass

        Dtsq = Dt * Dt

        for n in range(maxIter, 0, -1):

            # * compute displacement
            xcoeff = 2.0 * n
            force = self.gradient(xxn)
            dxx = v * Dt + force * massinv * Dtsq / xcoeff
            xxn = x + dxx / (xcoeff - 1)

            # * compute velocity
            vcoeff = 2.0 * n
            force = self.gradient(xvn)
            dxv = force * massinv * Dt / (vcoeff - 1)
            xvn = (x + (v + dxv) * Dt / (vcoeff - 2)) if n > 1 else (v + dxv)

        return xxn, xvn
    
    def compute_classic(self, state, Dt):
        x, v = state
        f = self.gradient(x)
        acc = f / self.mass
        new_r = x + v * Dt + 0.5 * Dt * Dt * acc
        f = self.gradient(new_r)
        new_acc = f / self.mass
        new_v = v + (acc + new_acc) * Dt * 0.5
        return new_r, new_v
    
    # attn the first experiment, get f(x + n*dx) by f(x + (n-1)*dx)
    def init_loop(self, maxIter: int):
        
        trajs: np.ndarray = np.zeros((self.times.shape[0], 2))
        derror: np.ndarray = np.zeros_like(self.times)
        verror: np.ndarray = np.zeros_like(self.times)
        traderror: np.ndarray = np.zeros_like(self.times)
        traverror: np.ndarray = np.zeros_like(self.times)
        start = self.times[0]
        trajs[0] = self.getState(0)
        
        for i in tqdm(range(1, trajs.shape[0])):
            Dt = self.times[i] - start
            
            nextX, nextV = self.computeTaylor(trajs[0], Dt, maxIter)
            traX, traV = self.compute_classic(trajs[0], Dt)
            labelx, labelv = self.getState(i)
            trajs[i] = nextX, nextV
            derror[i] = np.fabs((nextX - labelx))
            verror[i] = np.fabs((nextV - labelv))
            traderror[i] = np.fabs((traX - labelx))
            traverror[i] = np.fabs((traV - labelv))
            # print(f"after {Dt:5.2f}s, taylor: [{nextX:+.11f},{nextV:+.11f}], GT: [{labelx:+.11f},{labelv:+.11f}], abs(diff/labelx) is {derror[i]:.11f}, abs(diff/labelv) is {verror[i]:.11f}")

        return trajs, derror, verror, traderror, traverror
    
    # attn the second experiment, get f(x_0 + n*dx) by f(x_0)
    def current_loop(self, maxIter: int):
        
        trajs: np.ndarray = np.zeros((self.times.shape[0], 2))
        ttrajs: np.ndarray = np.zeros((self.times.shape[0], 2))
        derror: np.ndarray = np.zeros_like(self.times)
        verror: np.ndarray = np.zeros_like(self.times)
        traderror: np.ndarray = np.zeros_like(self.times)
        traverror: np.ndarray = np.zeros_like(self.times)
        start = self.times[0]
        trajs[0] = self.getState(0)
        ttrajs[0] = self.getState(0)
        
        for i in tqdm(range(trajs.shape[0] - 1)):
            Dt = self.times[i + 1] - self.times[i]

            nextX, nextV = self.computeTaylor(trajs[i], Dt, maxIter)
            traX, traV = self.compute_classic(trajs[i], Dt)

            labelx, labelv = self.getState(i + 1)

            trajs[i + 1] = nextX, nextV
            ttrajs[i + 1] = traX, traV

            derror[i + 1] = np.fabs((nextX - labelx))
            verror[i + 1] = np.fabs((nextV - labelv))

            traderror[i + 1] = np.fabs((traX - labelx))
            traverror[i + 1] = np.fabs((traV - labelv))
            
            # print(f"after {self.times[i + 1] - start:5.2f}s, taylor: [{nextX:+.11f},{nextV:+.11f}], GT: [{labelx:+.11f},{labelv:+.11f}], abs(diff/labelx) is {derror[i + 1]:.11f}, abs(diff/labelv) is {verror[i + 1]:.11f}")

        return trajs, derror, verror, traderror, traverror

    def generate(self, maxIter: int):

        trajs: np.ndarray = np.zeros((self.times.shape[0], 2))

        start = self.times[0]
        trajs[0] = self.getState(0)
        
        for i in tqdm(range(trajs.shape[0] - 1)):
            Dt = self.times[i + 1] - self.times[i]

            nextX, nextV = self.computeTaylor(trajs[i], Dt, maxIter)

            trajs[i + 1] = nextX, nextV
            
            # print(f"after {self.times[i + 1] - start:5.2f}s, taylor: [{nextX:+.11f},{nextV:+.11f}], GT: [{labelx:+.11f},{labelv:+.11f}], abs(diff/labelx) is {derror[i + 1]:.11f}, abs(diff/labelv) is {verror[i + 1]:.11f}")

        return trajs
    

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    tStart   : float = 0.0
    interval : float = 0.8
    tStep    : int   = 60
    maxIter  : int   = 1000

    mass    : float = 1.0
    radius  : float = 2.0
    gravity : float = 2.0

    pendulum = IdealPendulum(tStart, tStep, interval, mass, radius, gravity)

    trajs, derror, verror, traderror, traverror = pendulum.loop(maxIter)
    times = pendulum.times
    labelx = pendulum.q
    labelv = pendulum.dqdt

    fontsize = font_manager.FontProperties(size = 20)


    eps_max = max(derror.max(), verror.max())


    fig = plt.figure(figsize=[10,4], dpi=200)
    # plt.xlabel('$t = t_0 + \Delta t, \Delta t = interval * step$', fontproperties = fontsize) ; plt.ylabel(r'$Error = abs(\frac{predict - label}{label + \epsilon})$', fontproperties = fontsize)
    # plt.xlabel('$time$', fontproperties = fontsize) ; plt.ylabel(r'$Error = abs(\frac{predict - label}{label + \epsilon})$', fontproperties = fontsize)
    plt.xlabel('$time(s)$', fontproperties = fontsize) ; plt.ylabel(r'$error$($rad \cdot s^{-1}$ or $rad$)', fontproperties = fontsize)
    # plt.xlabel('$t = t_0 + \Delta t, \Delta t = interval * step$', fontproperties = fontsize) ; plt.ylabel(r'$q$(degree)', fontproperties = fontsize)
    # plt.title(f'interval = {interval}s, step = {tStep}, $t \in [{tStart}, {tStart + interval * tStep}]$', loc = 'center')
    plt.title(f'Mean Absolute Error', loc = 'center', fontsize = fontsize.get_size())
    # plt.title(f'Evolution for Ideal Pendulum', loc = 'center', fontsize = fontsize.get_size())
    plt.axis()

    plt.yscale('log')
    plt.plot(times, derror, label = 'Ours displacement error')
    plt.plot(times, verror, label = 'Ours velocity error')
    plt.plot(times, traderror, label = 'Velocity-Verlet displacement error')
    plt.plot(times, traverror, label = 'Velocity-Verlet velocity error')

    # plt.plot(times, trajs[:, 0], label = 'Ours displacement')
    # plt.plot(times, labelx, label = 'label displacement')
    # plt.scatter(times, trajs[:, 0], s = 8)
    # plt.scatter(times, labelx , s = 2)

    # plt.plot(times, trajs[:, 1], label = 'Ours velocity')
    # plt.plot(times, labelv, label = 'label velocity')
    # plt.scatter(times, trajs[:, 1], s = 8)
    # plt.scatter(times, labelv , s = 2)

    # plt.yticks([-np.pi / 4, -np.pi / 6,-np.pi / 9, 0, np.pi / 9, np.pi / 6, np.pi / 4], 
    #            ['$-\\frac{\pi}{4}$', '$-\\frac{\pi}{6}$', '$-\\frac{\pi}{9}$', '0', '$\\frac{\pi}{9}$', '$\\frac{\pi}{6}$', '$\\frac{\pi}{4}$'])
    if interval > 0:
        plt.xticks(np.linspace(tStart, tStart + interval * (tStep + 1), 7, dtype = np.int16))
    else:
        plt.xticks(np.linspace(tStart + interval * (tStep - 1), tStart, 7, dtype = np.int16))

    plt.tick_params(axis = 'both', labelsize = 17)

    # labelh = hamilton(labelx, labelv, mass, radius, gravity)
    # h = hamilton(trajs[:, 0], trajs[:, 1], mass, radius, gravity)
    # print(h)
    # print(labelh)
    # plt.plot(times, labelh, label = 'label Hamiltonian')
    # plt.plot(times, h, label = 'Hamiltonian')
    plt.legend(loc = 'lower right')
    plt.show()