import numpy as np

import autograd.numpy as anp
import autograd 

# solve y = ln(x)
# dy/dx = 1 / x, d2y/dx2 = - 1 / x^2 = -exp(-2y)

class Equation(object):

    def __init__(
        self,
        tStart  : float,
        tStep   : int,
        interval: float,

    ) -> None:
        
        self.values = np.arange(tStart, (tStep + 1) * interval + tStart, interval)
        print(self.values)


    def getState(self, idx):
        
        x = self.values[idx]

        # y = 1 / (1 + np.exp(-x))
        # dydx = y * (1 - y)

        # y = np.sin(x)
        # dydx = np.cos(x)

        # y = np.exp(-0.1 * x)
        # dydx = -0.1 * y
        # d2ydx2 = 0.01 * y

        # y = x * (x - 2) - 5
        # dydx = 2 * x - 2

        y = x**3
        dydx = 3 * x**2
        d2ydx2 = 6 * x

        return y, dydx, d2ydx2
    
    def thirdOrder(self, y):
        return -1e-3 * y
        # return 6

    def computeTaylor(self, state, Dx, maxIter):

        y, dydx, d2ydx2 = state

        yn = np.array(y)

        for n in range(maxIter, 0, -1):
            coeff = 3.0 * n

            # * compute displacement 
            dy2 = d2ydx2 + self.thirdOrder(yn) / coeff * Dx
            dy1 = dydx + dy2 / (coeff - 1) * Dx
            yn = y + dy1 / (coeff - 2) * Dx
    
        return yn
    

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

            Dt = self.times[i + 1] - self.times[i]


            nextX, nextV = self.computeTaylor(trajs[i], Dt, maxIter)

            traX, traV = self.compute_classic(ttrajs[i], Dt)

            labelx, labelv = self.getState(i + 1)

            trajs[i + 1] = nextX, nextV
            ttrajs[i + 1] = traX, traV

            derror[i + 1] = np.fabs((nextX - labelx)/labelx)
            verror[i + 1] = np.fabs((nextV - labelv)/labelv)

            traderror[i + 1] = np.fabs((traX - labelx)/labelx)
            traverror[i + 1] = np.fabs((traV - labelv)/labelv)

            # // print(f"after {self.times[i + 1]:5.2f}s, taylor: [{nextX:+.11f},{nextV:+.11f}], GT: [{labelx:+.11f},{labelv:+.11f}], abs(diff/labelx) is {derror[i + 1]:.11f}, abs(diff/labelv) is {verror[i + 1]:.11f}")
        return trajs, derror, verror, traderror, traverror


    def init_loop(self, maxIter: int):
        
        trajs : np.ndarray = np.zeros(self.values.shape[0])
        derror: np.ndarray = np.zeros_like(self.values)
        start = self.values[0] # * initial x
        init_state = self.getState(0)
        trajs[0] = self.getState(0)[0]

        for i in range(1, trajs.shape[0]): 


            Dx = self.values[i] - start


            nextX = self.computeTaylor(init_state, Dx, maxIter)

            labelx = self.getState(i)[0]
            trajs[i] = nextX

            derror[i] = np.fabs((nextX - labelx))
            
            # // print(f"after {Dt:5.2f}s, taylor: [{nextX:+.11f},{nextV:+.11f}], GT: [{labelx:+.11f},{labelv:+.11f}], abs(diff/labelx) is {derror[i]:.11f}, abs(diff/labelv) is {verror[i]:.11f}")

        return trajs, derror
    

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    tStart   : float = 0.0
    interval : float = 1.0
    tStep    : int   = 360
    maxIter  : int   = 1000

    equation = Equation(tStart, tStep, interval)

    trajs, derror = equation.loop(maxIter)
    
    values = equation.values
    label = np.exp(-0.1 * values)
    # label = values * (values - 2) - 5.
    # label = np.sin(values)
    # label = 1. / (1. + np.exp(-values))
    # label = values ** 3
    fontsize = font_manager.FontProperties(size = 25)

    mae = np.abs(trajs - label)

    fig = plt.figure(figsize=[10,4], dpi=200)
    # plt.xlabel('$T_{time}$') ; plt.ylabel(r'$Error = \frac{predict - label}{label}$')
    plt.xlabel(r'$x = x_0 + \Delta x$ (unitless)', fontproperties = fontsize); plt.ylabel('f(x)', fontproperties = fontsize)
    # plt.title(f'Mean Absolute Error compared with $y = e^{{-0.1x}}$', loc = 'center', fontproperties = fontsize)
    plt.yscale('symlog')
    plt.title(f'Trajectory of EdSr derived from $x_0$ and Label ', loc = 'center', fontproperties = fontsize)
    plt.tick_params(axis = 'both', labelsize = 15)
    plt.axis()
    # plt.plot(values, derror, label = 'y error')
    # plt.plot(values, verror, label = 'dydx error')

    plt.plot(values, trajs, label = 'Ours')
    plt.plot(values, label , label = r'$y = e^{-0.1x}$')
    plt.scatter(values, trajs, s = 8)
    plt.scatter(values, label , s = 2)
    # plt.plot(values, mae, label = 'MAE')

    # plt.plot(values, traderror, label = 'leap frog error')
    # plt.plot(values, traverror, label = 'leap frog error')
    plt.legend(fontsize = 15)
    plt.show()