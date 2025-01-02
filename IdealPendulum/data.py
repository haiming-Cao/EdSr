# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import autograd
import autograd.numpy as anp
import math
import scipy.integrate

solve_ivp = scipy.integrate.solve_ivp

def hamiltonian_fn(coords):

    q, p = anp.split(coords,2)
    H = 4* (1 - anp.cos(q)) + anp.square(p) / 2  # pendulum hamiltonian
    return H

def dynamics_fn(t, coords):
    
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dHdq, dHdp = anp.split(dcoords,2)
    S = anp.concatenate([dHdp, -dHdq], axis=-1)
    # dqdt, dpdt = anp.split(dcoords,2)
    # S = anp.concatenate([dpdt, -dqdt], axis=-1)
    return S

def get_trajectory(tStart, tStep, interval, y0 = None, m = None, g = None, l = None):

    if m is None:
        m = 1.0
    if g is None:
        g = 2.0
    if l is None:
        l = 2.0
    
    tEnd = tStart + tStep * interval
    t_eval = anp.arange(tStart, tEnd + interval, interval)
    # print(t_eval, tStart, tEnd)

    # get initial state
    if y0 is None:
        y0 = anp.array([math.pi / 3, 0.])
    # if y0 is None:
    #     y0 = anp.random.rand(2)*2.-1
    # print(y0)
    # if l is None:
    #     radius = anp.random.rand() + 1.3 # sample a range of radii
    # y0 = y0 / anp.sqrt((y0**2).sum()) * l ## set the appropriate radius

    spring_ivp = solve_ivp(fun = dynamics_fn, t_span=[tStart, tEnd + interval], y0 = y0, t_eval = t_eval, rtol = 3e-14, dense_output = True)
    q, p = spring_ivp['y'][0], spring_ivp['y'][1]
    dydt = [dynamics_fn(None, y) for y in spring_ivp['y'].T]
    dydt = anp.stack(dydt).T
    dqdt, dpdt = anp.split(dydt,2)
    
    return q, p, dqdt, dpdt, t_eval