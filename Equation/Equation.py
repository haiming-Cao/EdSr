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
        equation: str,
    ) -> None:
        
        self.equation = equation
        self.values = np.arange(tStart, (tStep + 1) * interval + tStart, interval)

        match self.equation:
            case r'$f(x) = e^{-0.1x}$':
                src = np.exp(-0.1 * self.values)
                derivative = -0.1 * src
            case r'$f(x) = e^{0.1x}$':
                src = np.exp(0.1 * self.values)
                derivative = 0.1 * src
            case r'$f(x) = x^2 - 2x - 5$':
                src = self.values * (self.values - 2) - 5
                derivative = 2 * self.values - 2
            case r'$f(x) = \sin x$':
                src = np.sin(self.values)
                derivative = np.cos(self.values)
            case r'$f(x) = \frac{1}{1 + e^{-x}}$':
                src = 1 / (1 + np.exp(-self.values))
                derivative = src * (1 - src)
            case r'$f(x) = x^3$':
                src = self.values**3
                derivative = 3 * (self.values**2)
            case _:
                raise TypeError
        # print(self.values)
        self.src = src
        self.derivative = derivative


    def getState(self, idx):
        
        x = self.values[idx]
        y = self.src[idx]
        dydx = self.derivative[idx]

        return y, dydx
    
    def secondOrder(self, y):
        match self.equation:
            case r'$f(x) = e^{-0.1x}$':
                return 0.01 * y
            case r'$f(x) = e^{0.1x}$':
                return 0.01 * y
            case r'$f(x) = x^2 - 2x - 5$':
                return 2
            case r'$f(x) = \sin x$':
                return -y
            case r'$f(x) = \frac{1}{1 + e^{-x}}$':
                return y - 3 * y*y + 2*y*y*y
            case r'$f(x) = x^3$':
                return (6 * np.cbrt(y)) if y >= 0. else -(6 * np.cbrt(-y))
            case _:
                raise TypeError

        # return 0.01 * y # y = e^-0.1x
        # return 6 * np.cbrt(y) # x^3
        # return 2 # y = x^2- x - 5
        # return y - 3 * y*y + 2*y*y*y # sigmoid
        # return -y # sinx

    def computeEdSr(self, state, Dx, maxIter):

        y, dydx = state

        yn = np.array(y)
        dydxn = np.array(y)

        Dxsq = Dx * Dx

        for n in range(maxIter, 0, -1):
            xcoeff = 2.0 * n
            vcoeff = 2.0 * n

            # * compute displacement 
            dy = dydx * Dx + self.secondOrder(yn) * Dxsq / xcoeff 
            yn = y + dy / (xcoeff - 1)

            ddydx = self.secondOrder(dydxn) * Dx / (vcoeff - 1)
            dydxn = (y + (dydx + ddydx) * Dx / (vcoeff - 2)) if n > 1 else (dydx + ddydx)
    
        return yn, dydxn
    
    def compute_classic(self, state, Dt):
        y, dydx = state
        acc = self.secondOrder(y)
        new_r = y + dydx * Dt + 0.5 * Dt * Dt * acc
        new_acc = self.secondOrder(new_r)
        new_v = dydx + (acc + new_acc) * Dt / 2
        return new_r, new_v
    
    # attn the second experiment, get f(x_0 + n*dx) by f(x_0)
    def current_loop(self, maxIter: int):
        
        trajs : np.ndarray = np.zeros((self.values.shape[0], 2))
        ttrajs : np.ndarray = np.zeros((self.values.shape[0], 2))
        derror: np.ndarray = np.zeros_like(self.values)
        verror: np.ndarray = np.zeros_like(self.values)
        traderror: np.ndarray = np.zeros_like(self.values)
        traverror: np.ndarray = np.zeros_like(self.values)

        mapederror: np.ndarray = np.zeros_like(self.values)
        mapeverror: np.ndarray = np.zeros_like(self.values)

        start = self.values[0]

        trajs[0] = self.getState(0)
        ttrajs[0] = self.getState(0)
        
        for i in range(trajs.shape[0] - 1): 
            # ! 下一时刻减去当前时刻得到时间
            Dt = self.values[i + 1] - self.values[i]

            # attn EdSr computation
            nextX, nextV = self.computeEdSr(trajs[i], Dt, maxIter)

            traX, traV = self.compute_classic(ttrajs[i], Dt)

            labelx, labelv = self.getState(i + 1)

            trajs[i + 1] = nextX, nextV
            ttrajs[i + 1] = traX, traV

            derror[i + 1] = np.fabs((nextX - labelx))
            verror[i + 1] = np.fabs((nextV - labelv))

            mapederror[i + 1] = np.fabs((nextX - labelx) / (labelx + 1e-34)) * 100
            mapeverror[i + 1] = np.fabs((nextV - labelv) / (labelv + 1e-34)) * 100

            traderror[i + 1] = np.fabs((traX - labelx))
            traverror[i + 1] = np.fabs((traV - labelv))

            # // print(f"after {self.values[i + 1]:5.2f}s, EdSr: [{nextX:+.11f},{nextV:+.11f}], GT: [{labelx:+.11f},{labelv:+.11f}], abs(diff/labelx) is {derror[i + 1]:.11f}, abs(diff/labelv) is {verror[i + 1]:.11f}")
        return trajs, derror, verror, traderror, traverror, mapederror, mapeverror

    # attn the first experiment, get f(x + n*dx) by f(x + (n-1)*dx)
    def init_loop(self, maxIter: int):
        
        trajs : np.ndarray = np.zeros((self.values.shape[0], 2))
        derror: np.ndarray = np.zeros_like(self.values)
        verror: np.ndarray = np.zeros_like(self.values)
        traderror: np.ndarray = np.zeros_like(self.values)
        traverror: np.ndarray = np.zeros_like(self.values)

        mapederror: np.ndarray = np.zeros_like(self.values)
        mapeverror: np.ndarray = np.zeros_like(self.values)

        start = self.values[0] # * initial x

        init_state = self.getState(0)
        trajs[0] = self.getState(0)

        for i in range(1, trajs.shape[0]): 

            # attn get interval
            Dx = self.values[i] - start

            # attn EdSr computation
            nextX, nextV = self.computeEdSr(init_state, Dx, maxIter)

            traX, traV = self.compute_classic(init_state, Dx)

            labelx, labelv = self.getState(i)
            trajs[i] = nextX, nextV

            derror[i] = np.fabs((nextX - labelx)) # attn error of displacement
            verror[i] = np.fabs((nextV - labelv))

            mapederror[i] = np.fabs((nextX - labelx) / (labelx + 1e-34)) * 100
            mapeverror[i] = np.fabs((nextV - labelv) / (labelv + 1e-34)) * 100

            traderror[i] = np.fabs((traX - labelx)) # attn error of velocity
            traverror[i] = np.fabs((traV - labelv))
            
            # // print(f"after {Dt:5.2f}s, EdSr: [{nextX:+.11f},{nextV:+.11f}], GT: [{labelx:+.11f},{labelv:+.11f}], abs(diff/labelx) is {derror[i]:.11f}, abs(diff/labelv) is {verror[i]:.11f}")

        return trajs, derror, verror, traderror, traverror, mapederror, mapeverror
    

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.axes import Axes

    tStart   : float = -40.0
    interval : float = 1.0
    tStep    : int   = 60
    maxIter  : int   = 500


    # src_label: str = r'$y = e^{0.1x}$'
    # src_label: str = r'$y = x^2 - 2x - 5$'
    # src_label: str = r'$y = \sin x$'
    # src_label: str = r'$y = \frac{1}{1 + e^{-x}}$'
    src_label: str = r'$y = x^3$'

    equation = Equation(tStart, tStep, interval, src_label)

    trajs, derror, verror, traderror, traverror, mapederror, mapeverror = equation.init_loop(maxIter)
    
    values = equation.values
    label = equation.src

    fontsize = font_manager.FontProperties(size = 25)

    eps_max = max(derror.max(), verror.max())


    fig = plt.figure(figsize=[50,20], dpi=200)
    ax: Axes = plt.gca()
    # plt.xlabel('$T_{time}$') ; plt.ylabel(r'$Error = \frac{predict - label}{label}$')
    plt.axis()
    # plt.tick_params(axis = 'both', labelsize = fontsize.get_size())
    plt.yscale('log')

    # attn MAE derived from x0
    # plt.title(f'Mean Absolute Error compared with {src_label}', loc = 'center', fontproperties = fontsize)
    # plt.xlabel(r'$x = x_0 + \Delta x$ (unitless)', fontproperties = fontsize); plt.ylabel('error(unitless)', fontproperties = fontsize)
    # plt.plot(values, derror, label = 'y error', linewidth = 2.5)
    # plt.plot(values, verror, label = '$\\frac{dy}{dx}}$ error', linewidth = 1)


    # attn generate trajectory derived from x0
    # plt.title(f'Trajectory of EdSr derived from $x_{{0}}$ and Label ', loc = 'center', fontproperties = fontsize)
    # plt.xlabel(r'$x = x_0 + \Delta x$ (unitless)', fontproperties = fontsize); plt.ylabel('f(x)', fontproperties = fontsize)
    # plt.plot(values, trajs[:, 0], label = 'Ours', linewidth = 2)
    # plt.plot(values, label , label = src_label, linewidth = 1)
    # plt.scatter(values, trajs[:, 0], s = 8)
    # plt.scatter(values, label , s = 2)

    # attn MAE derived from x(n-1)
    # plt.title(f'Mean Absolute Error compared with {src_label}', loc = 'center', fontproperties = fontsize)
    # # plt.ticklabel_format(axis = 'y', style = 'sci', useOffset = True, useMathText = True, scilimits = (-1, 3))
    # plt.xlabel(r'x(unitless)', fontproperties = fontsize); plt.ylabel('error(unitless)', fontproperties = fontsize)
    # plt.plot(values, derror, label = 'y error', linewidth = 2.5)
    # plt.plot(values, verror, label = '$\\frac{dy}{dx}}$ error', linewidth = 1)
    # ax.yaxis.get_offset_text().set(size = 15)

    # attn MAPE derived from x(n-1)
    plt.title(f'Mean Absolute Percentage Error compared with {src_label}', loc = 'center', fontproperties = font_manager.FontProperties(size = 22))
    # plt.ticklabel_format(axis = 'y', style = 'sci', useOffset = True, useMathText = True, scilimits = (-1, 3))
    plt.xlabel(r'x(unitless)', fontproperties = fontsize); plt.ylabel('error(%)', fontproperties = fontsize)
    plt.plot(values, mapederror, label = 'y error', linewidth = 2.5)
    plt.plot(values, mapeverror, label = '$\\frac{dy}{dx}}$ error', linewidth = 1)
    plt.ylim(top = 1e6)
    ax.yaxis.get_offset_text().set(size = 15)


    # attn generate trajectory derived from x(n-1)
    # plt.title(f'Trajectory of EdSr derived from $x_{{n-1}}$ and Label ', loc = 'center', fontproperties = fontsize)
    # plt.xlabel(r'x(unitless)', fontproperties = fontsize); plt.ylabel('f(x)', fontproperties = fontsize)
    # plt.plot(values, trajs[:, 0], label = 'Ours', linewidth = 2)
    # plt.plot(values, label , label = src_label, linewidth = 1)
    # plt.scatter(values, trajs[:, 0], s = 8)
    # plt.scatter(values, label , s = 2)
    # # plt.ticklabel_format(axis = 'y', style = 'sci', useOffset = False, useMathText = True)
    # ax.yaxis.get_offset_text().set(size = 15)
    

    # plt.plot(values, np.abs(trajs[:, 0] - label), label = 'Error = abs(Ours - label)')

    # attn compares with leap frog integrator. MAE
    # plt.plot(values, traderror, label = 'leap frog error')
    # plt.plot(values, traverror, label = 'leap frog error')
    # plt.yticks([0., 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6])
    plt.tick_params(axis = 'both', labelsize = 17)
    plt.legend(fontsize = 20, loc = 'lower right')
    # plt.legend(fontsize = 15)
    plt.show()