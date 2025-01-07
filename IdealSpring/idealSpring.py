import numpy as np

import autograd.numpy as anp
import autograd 

class IdealSpring(object):

    def __init__(
        self,
        tStart  : float,
        tStep   : int,
        interval: float,
        mass    : float = 1.0,
        k       : float = 1.0,
        hamilton: float = 1.0,

    ) -> None:
        
        self.times = np.arange(tStart, (tStep + 1) * interval + tStart, interval)
        # print(self.times)
        self.mass     = mass
        self.hamilton = hamilton
        self.k        = k
        self.labelx   = np.sqrt(2 * hamilton) * np.sin(self.times)
        self.labelv   = np.sqrt(2 * hamilton) * np.cos(self.times)



    def getState(self, idx):
        
        t = self.times[idx]

        x = np.sqrt(2 * self.hamilton) * np.sin(t)
        v = np.sqrt(2 * self.hamilton) * np.cos(t)

        return x, v
    
    def gradient(self, x):
        return -x
    

    def computeEdSr(self, state, Dt, maxIter, split = 8):

        x, v = state    

        xn = np.array(x)
        vn = np.array(x)

        Dtsq = Dt * Dt

        # * compute displacement 
        for n in range(maxIter, 0, -1):
            xcoeff = 2.0 * n
            xacc = self.gradient(xn)
            dx = v * Dt + xacc * Dtsq / xcoeff 
            xn = x + dx / (xcoeff - 1)

            # * compute velocity
            vcoeff = 2.0 * n
            vacc = self.gradient(vn)
            dv = vacc * Dt / (vcoeff - 1)
            vn = (x + (v + dv) * Dt / (vcoeff - 2)) if n > 1 else (v + dv)
            
        return xn, vn
    def VelocityVerlet(self, state, Dt):
        x, v = state
        acc = self.gradient(x)
        new_r = x + v * Dt + 0.5 * Dt * Dt * acc
        new_acc = self.gradient(new_r)
        new_v = v + (acc + new_acc) * Dt / 2
        return new_r, new_v
    
    # attn the first experiment, get f(x + n*dx) by f(x + (n-1)*dx)
    def current_loop(self, maxIter: int):
        
        trajs : np.ndarray = np.zeros((self.times.shape[0], 2))
        ttrajs : np.ndarray = np.zeros((self.times.shape[0], 2))
        derror: np.ndarray = np.zeros_like(self.times)
        verror: np.ndarray = np.zeros_like(self.times)
        traderror: np.ndarray = np.zeros_like(self.times)
        traverror: np.ndarray = np.zeros_like(self.times)
        start = self.times[0]

        trajs[0] = self.getState(0)
        ttrajs[0] = self.getState(0)
        
        for i in range(trajs.shape[0] - 1): 
            # ! get interval
            Dt = self.times[i + 1] - self.times[i]

            # attn EdSr computation
            nextX, nextV = self.computeEdSr(trajs[i], Dt, maxIter)

            # ! compute displacement and velocity using  xt = x0 + vt + 0.5 * a * t * t 
            traX, traV = self.VelocityVerlet(ttrajs[i], Dt)

            # * copmpute label
            labelx, labelv = self.getState(i + 1)

            trajs[i + 1] = nextX, nextV
            ttrajs[i + 1] = traX, traV

            derror[i + 1] = np.fabs((nextX - labelx))
            verror[i + 1] = np.fabs((nextV - labelv))

            traderror[i + 1] = np.fabs((traX - labelx))
            traverror[i + 1] = np.fabs((traV - labelv))

            # // print(f"after {self.times[i + 1]:5.2f}s, taylor: [{nextX:+.11f},{nextV:+.11f}], GT: [{labelx:+.11f},{labelv:+.11f}], abs(diff/labelx) is {derror[i + 1]:.11f}, abs(diff/labelv) is {verror[i + 1]:.11f}")
        return trajs, derror, verror, traderror, traverror

    # attn the second experiment, get f(x_0 + n*dx) by f(x_0)
    def init_loop(self, maxIter: int):
        
        trajs : np.ndarray = np.zeros((self.times.shape[0], 2))
        derror: np.ndarray = np.zeros_like(self.times)
        verror: np.ndarray = np.zeros_like(self.times)
        traderror: np.ndarray = np.zeros_like(self.times)
        traverror: np.ndarray = np.zeros_like(self.times)
        start = self.times[0] 
        trajs[0] = self.getState(0)

        for i in range(1, trajs.shape[0]): 

            # attn get interval
            Dt = self.times[i] - start

            # attn EdSr computation
            nextX, nextV = self.computeEdSr(trajs[0], Dt, maxIter)

            # ! compute displacement and velocity using  xt = x0 + vt + 0.5 * a * t * t 
            traX, traV = self.VelocityVerlet(trajs[0], Dt)

            labelx, labelv = self.getState(i)
            trajs[i] = nextX, nextV

            derror[i] = np.fabs((nextX - labelx)) # attn error of displacement
            verror[i] = np.fabs((nextV - labelv))

            traderror[i] = np.fabs((traX - labelx)) # attn error of velocity
            traverror[i] = np.fabs((traV - labelv))
            
            # // print(f"after {Dt:5.2f}s, taylor: [{nextX:+.11f},{nextV:+.11f}], GT: [{labelx:+.11f},{labelv:+.11f}], abs(diff/labelx) is {derror[i]:.11f}, abs(diff/labelv) is {verror[i]:.11f}")

        return trajs, derror, verror, traderror, traverror
    

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    tStart   : float = 0.0
    interval : float = 0.1
    tStep    : int   = 100
    maxIter  : int   = 500

    spring = IdealSpring(tStart, tStep, interval)

    trajs, derror, verror, traderror, traverror = spring.init_loop(maxIter)
    times = spring.times

    labelx = spring.labelx
    labelv = spring.labelv

    fontsize = font_manager.FontProperties(size = 25)

    eps_max = max(derror.max(), verror.max())


    fig = plt.figure(figsize=[50,20], dpi=200)
    ax = fig.gca()
    
    # attn MAE derived from t(0)
    plt.title(f'MAE of trajectory derived from $t_0$', loc = 'center', fontproperties = fontsize)
    plt.xlabel(r'$t = t_0 + \Delta t$ (s)', fontproperties = fontsize); plt.ylabel(r'y(unitless)', fontproperties = fontsize)
    plt.plot(times, derror, label = 'Ours displacement error')
    plt.plot(times, verror, label = 'Ours velocity error')
    # plt.plot(times, traderror, label = 'vv displacement error')
    # plt.plot(times, traverror, label = 'vv velocity error')
    plt.plot(times, traderror, label = 'Velocity-Verlet displacement error')
    plt.plot(times, traverror, label = 'Velocity-Verlet velocity error')
    plt.yscale('log')

    # attn MAE derived from t(n-1)
    # plt.title(r'MAE of trajectory derived from $t_{n-1}$', loc = 'center', fontproperties = fontsize)
    # plt.xlabel(r'time(s)', fontproperties = fontsize); plt.ylabel(r'error(unitless)', fontproperties = fontsize)
    # plt.plot(times, derror, label = 'Ours displacement error')
    # plt.plot(times, verror, label = 'Ours velocity error')
    # # plt.plot(times, traderror, label = 'Velocity-Verlet displacement error')
    # # plt.plot(times, traverror, label = 'Velocity-Verlet velocity error')
    # plt.ticklabel_format(axis = 'y', style = 'sci', useOffset = True, useMathText = True)
    # # plt.ylim(top = 1.0)
    # # plt.yscale('log')
    # ax.yaxis.get_offset_text().set(size = 15)

    # attn generate trajectory derived from t(0)
    # plt.xlabel(r'$t = t_0 + \Delta t$ (s)', fontproperties = fontsize); plt.ylabel(r'y(unitless)', fontproperties = fontsize)
    # plt.title(f'trajectory derived from $t_0$', loc = 'center', fontproperties = fontsize)
    # plt.plot(times, trajs[:, 0], label = 'Ours displacement', linewidth = 2)
    # plt.scatter(times, trajs[:, 0], s = 8)
    # plt.plot(times, trajs[:, 1], label = 'Ours velocity', linewidth = 2)
    # plt.scatter(times, trajs[:, 1], s = 8)

    # plt.plot(times, labelx, label = 'label displacement', linewidth = 1)
    # plt.scatter(times, labelx , s = 4)
    # plt.plot(times, labelv, label = 'label velocity', linewidth = 1)
    # plt.scatter(times, labelv , s = 4)

    # attn generate trajectory derived from t(n-1)
    # plt.xlabel(r'time(s)', fontproperties = fontsize); plt.ylabel(r'error(unitless)', fontproperties = fontsize)
    # plt.title(r'trajectory derived from $t_{n-1}$', loc = 'center', fontproperties = fontsize)
    # plt.plot(times, trajs[:, 0], label = 'Ours displacement', linewidth = 2)
    # plt.plot(times, trajs[:, 1], label = 'Ours velocity', linewidth = 2)
    # plt.scatter(times, trajs[:, 0], s = 8)
    # plt.scatter(times, trajs[:, 1], s = 8)

    # plt.plot(times, labelx, label = 'label displacement', linewidth = 1)
    # plt.plot(times, labelv, label = 'label velocity', linewidth = 1)
    # plt.scatter(times, labelx , s = 4)
    # plt.scatter(times, labelv , s = 4)


    plt.axis()
    # plt.yticks([-np.pi / 2, -np.pi / 4,-np.pi / 8, 0, np.pi / 8, np.pi / 4, np.pi / 2], 
    #            ['$-\\frac{\pi}{2}$', '$-\\frac{\pi}{4}$', '$-\\frac{\pi}{8}$', '0', '$\\frac{\pi}{8}$', '$\\frac{\pi}{4}$', '$\\frac{\pi}{2}$'])
    plt.xticks(np.linspace(tStart, tStart + interval * (tStep + 2), 11, dtype = np.int16))

    plt.tick_params(axis = 'both', labelsize = 17)
    plt.legend(loc = 'lower left', fontsize = 10)
    plt.tight_layout()
    plt.show()