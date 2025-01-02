# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import numpy as np
import scipy.integrate
from einops import rearrange
solve_ivp = scipy.integrate.solve_ivp

from pprint import pprint

# G = 6.67e-2 # cm^3*g/s^2
G = 1.0
# G_std = 6.67e-11 # N*m/kg^2 = m^3*kg/s^2
np.random.seed(2)

BASIS_INTERVAL = 0.001

##### ENERGY #####
def potential_energy(state) -> np.ndarray:
    '''U=sum_i,j>i G m_i m_j / r_ij'''
    Uenergy = np.zeros(state.shape[0]) 
    
    for i in range(state.shape[-2] - 1): # number of bodies
        bodies_i = state[..., i:i+1, :] # * get current particle
        bodies_j = state[..., i+1:, :]  # * get others
        displacements = bodies_j[..., 1:3] - bodies_i[..., 1:3] # * compute displacement
        r_ij = (displacements**2).sum(axis = -1)**0.5 # * compute distance
        m_i = bodies_i[..., 0]
        m_j = bodies_j[..., 0]
        # // print(displacements.shape, r_ij.shape, m_i.shape, m_j.shape)
        energy_i = np.sum(m_j / r_ij * m_i * G, axis = -1)
        Uenergy = np.add(Uenergy, energy_i, casting = 'no') 
    # tot_energy += m_i * m_j / r_ij
    return -Uenergy  # define ∞ as 0

def kinetic_energy(state: np.ndarray) -> np.ndarray:
    '''T = sum_i .5*m*v^2'''
    v_scalar_sq = (state[..., 3:5]**2).sum(axis = -1)
    mass        =  state[..., 0]
    # // print(v_scalar_sq.shape, mass.shape)
    energies = 0.5 * mass * v_scalar_sq
    T = energies.sum(axis = -1)
    return T

def total_energy(state) -> np.ndarray:
    return potential_energy(state) + kinetic_energy(state)


##### DYNAMICS #####
def get_accelerations(state, epsilon = 0):
    # * attributes of partical is (mass, qx, qy, px, py)
    # // print(state.shape)
    net_accs = [] # [nbodies x 2]

    for i in range(state.shape[0]): # number of bodies

        bodies_j = np.concatenate([state[:i, :], state[i+1:, :]], axis = 0) 
        bodies_i = state[i:i+1, 1:3]

        displacements_ij = bodies_j[:, 1:3] - bodies_i

        dis_sq = (displacements_ij**2).sum(axis = -1, keepdims = True) 
        dis = dis_sq**0.5
        # // print(bodies_j.shape, displacements_ij.shape, dis.shape)
        other_masses = bodies_j[:, 0:1] # * 其他微粒的质量
        pointwise_accs = other_masses * displacements_ij / (dis_sq * dis + epsilon) * G 
        # // print(pointwise_accs.shape)
        net_acc = pointwise_accs.sum(0, keepdims=True)  # 加速度累加
        net_accs.append(net_acc)

    net_accs = np.concatenate(net_accs, axis = 0)

    return net_accs # return a
  
def update(t, state):
    state = state.reshape(-1,5) # [bodies, properties]
    deriv = np.zeros_like(state)
    deriv[:,1:3] = state[:,3:5] # dx/dt, dy/dt = vx, vy
    deriv[:,3:5] = get_accelerations(state)  # dv/dt = a
    return deriv.reshape(-1)


##### INTEGRATION SETTINGS #####
def get_orbit(state, update_fn=update, t_start=0., t_step=100, interval = 0.5, **kwargs):
    if not 'rtol' in kwargs.keys():
        kwargs['rtol'] = 1e-10

    orbit_settings = locals()

    Bintv = BASIS_INTERVAL if interval > 0 else BASIS_INTERVAL * -1

    skip = int(interval / Bintv)

    nbodies = state.shape[0]

    t_end = t_start + t_step * interval
    t_eval = np.arange(t_start, t_end + Bintv, Bintv)
    # t_eval = np.linspace(t_span[0], t_span[1], t_points)


    path = solve_ivp(fun=update_fn, t_span=[t_start, t_end], y0=state.flatten(),
                     t_eval=t_eval, **kwargs)
    
    # ATTN 获取到的 path 的维度是(solver, time), 要把维度转换成 (time, solver), 最后将维度转换成 (time, nbodies, attributes)
    # ATTN 表达式具体含义为 "(nbodies attribute) time -> time nbodies attributes"
    orbit = rearrange(path['y'], "(nb x) t -> t nb x", x = 5)
    orbit = orbit[::skip]
    
    t_eval = t_eval[::skip]

    orbit_settings['t_eval'] = t_eval

    return orbit, orbit_settings


def rotate(x, theta):
    # * 旋转变换
    cos, sin = np.cos(theta), np.sin(theta)
    mat = np.array([[cos, -sin],
                    [sin, cos]
                    ])
    res = np.einsum("ij,j->i", mat, x)
    return res
    

##### INITIALIZE THE BODIES #####
def state_generator(
    nbodies: int, 
    mass: float | list | np.ndarray = 1.0, # grams
    coord: np.ndarray | None = None, # cm
    velocity: np.ndarray | None = None, # cm/s
    **kwargs
):
    """
    ## Description: 
    函数返回一个初始的系统状态，该状态用二维数组来表示，二维数组所代表的信息如下所示：\n
        initial state information: shape -> (nparticals, info)\n
        [
            partical 1 : [mass, qx, qy, px, py],\n
            partical 2 : [mass, qx, qy, px, py],\n
            ...\n
            partical n : [mass, qx, qy, px, py],\n
        ]\n

    ## Parameters:
        nbodies (float)                 : 粒子数。\n
        mass (float | list | np.ndarray): 粒子质量, 单位为 grams. Defaults to 1.0.\n
        coord (np.ndarray | None)       : 每个粒子对应的笛卡尔坐标，单位为 cm. Defaults to None. \n
        velocity (np.ndarray | None)    : 每个粒子对应的速度，单位为 cm/s. Defaults to None. \n
        kwargs :
            orbit_noise (float): 向速度里面所添加的噪声的标准差 stddev. Defaults to None.\n
            min_radius (float) : 运动的最小半径，当 `coord` 为 `None` 时，该参数有效. Defaults to 0.5.\n
            max_radius (float) : 运动的最大半径. 当 `coord` 为 `None` 时，该参数有效. Defaults to 1.5.\n

    ## Returns:
        np.ndarray: 系统初始状态数组
    """
    init_state = np.zeros((nbodies, 5))
    # set coordinate
    if coord is None:
        coord = np.zeros((2))
        min_radius = kwargs.get('min_radius', 0.5)
        max_radius = kwargs.get('max_radius', 1.5)
        coord = np.random.rand(2) * (max_radius - min_radius) + min_radius

        degree = (2 * np.pi) / nbodies
        for i in range(nbodies):
            init_state[i, 1:3] = coord
            coord = rotate(coord, degree)
    else:
        assert isinstance(coord, np.ndarray), "coord type should be a numpy array"
        assert coord.shape == (nbodies, 2), f"coord shape expected to ({nbodies}, 2), but found {coord.shape}"
        init_state[:, 1:3] = coord

    # set velocity
    if velocity is None:
        coord = init_state[:, 1:3]
        r = np.sqrt( np.sum((coord**2), axis = -1, keepdims = True) )
        velocity = np.fliplr(coord) / (2 * r**1.5)
        velocity[:, 1] *= -1

    if kwargs.get('orbit_noise') is not None:
        velocity *= 1 + kwargs['orbit_noise'] * np.random.randn()

    init_state[:, 3:5] = velocity

    # set mass
    assert isinstance(mass, (float, list, np.ndarray)), "mass type should be among (float, list, np.ndarray)"
    init_state[:, 0] = mass

    return init_state

    

##### HELPER FUNCTION #####
def coords2state(coords, nbodies=2, mass=1):
    timesteps = coords.shape[0]
    state = coords.T
    state = state.reshape(-1, nbodies, timesteps).transpose(1,0,2)
    mass_vec = mass * np.ones((nbodies, 1, timesteps))
    state = np.concatenate([mass_vec, state], axis=1)
    return state

if __name__ == "__main__":
    # * 获取初始状态
    state: np.ndarray = state_generator(2, mass = 1.0)
    # * 通过积分获取轨迹
    orbit, orbit_settings = get_orbit(state)
    # * 获取体系随时间变化的势能
    potential = potential_energy(orbit)
    # * 获取体系随时间变化的动能
    kinetic = kinetic_energy(orbit)