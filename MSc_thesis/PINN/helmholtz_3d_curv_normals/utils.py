import jax.numpy as jnp
import jax.random as jrandom
from jax import lax
import numpy as np
from scipy.interpolate import interp1d
import jax

import os

speed_sources_3d = np.array([
    (-0.08985056150054974, -0.008882406449874009, 0.04019719764706847),
    (0.0581112260698049, 0.058453904571396736, -0.02104289838781495),
    (0.019424839824088352, -0.02676395112787662, -0.08269633589534109),
    (-0.08440624592882484, 0.07395002003195347, -0.07151556071708548),
    (-0.06081282737864278, -0.05554740654884103, 0.08807620364131338),
    (0.06845928237086446, -0.04850082131944389, -0.025122832182760083),
    (-0.034297348975401806, 0.016270208831026728, -0.09505357563931414),
    (-0.01187167676363235, -0.011372412393442305, 0.013167872287392163),
    (0.06817318510689108, 0.07170327061888765, -0.06894661548157868),
    (-0.015584063812601134, -0.04227735964968067, -0.04846895234853399),
]) * 10

sources_3d = [
    (-0.1541418001906502, 0.17613574803971815, -0.8403192399137376),
    (-0.159748719130701, -0.559347469307465, 0.7727151434385062),
    (-0.8701900521739256, -0.029648105575562442, 0.29328783838926775),
    (0.44387657656970386, -0.3588745079602344, 0.6545577753130077),
    (-0.34681626346433464, -0.790177951083181, -0.05720987392637132),
    (-0.3629227862437404, 0.6344444924547548, -0.48293502176960823),
    (0.767831443671653, -0.2687944839523762, -0.09059188111695685),
    (-0.2025168627301938, 0.5476142541036333, -0.7304706633705653),
    (0.35384885237442526, -0.6201454597637217, 0.004343070431033864),
    (-0.021848926707657413, -0.06499607012744174, 0.9105251699667609),
]

spacing = 5
frequencies_pink = jnp.arange(1, 80 + spacing, spacing)

# Pink noise
def phase(coords, t, coords0, coords_v, sphere_radius, v, norm=False):
    coords0_t = [x0 + vs_x * t for x0, vs_x in zip(coords0, coords_v)]
    # v [m/s] := gamma [/s] * r [m] (10*0.001=0.01) * 10 # too costly
    if norm:
        norm0_t = jnp.linalg.norm(jnp.array(coords0_t))
        norm0 = jnp.linalg.norm(jnp.array(coords0))
        coords0_t = [x0_t / norm0_t * norm0 for x0_t in coords0_t]
    return jnp.sqrt(sum((x - x0_t)**2 for x, x0_t in zip(coords, coords0_t))) / v

def find_sources(x, mult):
    mr, ms = mult
    sources = sources_3d * mr
    speed_sources = speed_sources_3d * ms
    return sources, speed_sources

def space_time_signal(t, x, y, z, noise, sphere_radius, freq_denom, mult, norm, v, alpha=1): 
    sources, speed_sources = find_sources(x, mult)
    a0 = noise/(len(frequencies_pink)*len(sources))
    Qs = a0 * sum(
        sum(
            1/(f**alpha) * jnp.sin(2 * jnp.pi * f * (t + phase([x, y, z], t, source, speed, sphere_radius, v, norm)) / freq_denom) 
            for source, speed in zip(sources, speed_sources)
        )
        for f in frequencies_pink
    )
    return Qs

# parameters for hyperbolic PDE
def find_idx(matrix, target_row):
    """jax tracer method for jnp.where; x5 faster"""
    def body_fun(carry, index):
        found, idx = carry
        row = matrix[index]
        condition = jnp.all(row == target_row)
        found = lax.cond(condition, lambda _: True, lambda _: found, None)
        idx = lax.cond(condition, lambda _: index, lambda _: idx, None)
        return (found, idx), None

    init_val = (False, -1)
    result, _ = lax.scan(body_fun, init_val, jnp.arange(matrix.shape[0]))
    _, idx = result
    return idx

def create_dir(wandb_name, wandb_dir = '/vol/bitbucket/sc3719/JAXPI'):
    wandb_dir = f'{wandb_dir}/{wandb_name}'
    # wandb_dir = os.path.expanduser(wandb_dir)

    print(f"\nTrying directory: {wandb_dir}")
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)
    if not os.access(wandb_dir, os.W_OK):
        print("\nWARNING: You do not have write permission to this directory!\n")
    return wandb_dir

# def load_Qs_noise(c, mesh_size): 
#     x_old = jnp.linspace(0, mesh_size, c.XN)
#     t_old = jnp.linspace(0, c.T_FEM, c.TN)   
#     noise_matrix = jnp.load(c.Qs_path)

#     def func_Qs(x_new, t_new):
#         """x_new is the idx of the parcell/node, t_new is the absolute time"""
#         interpolator_x = interp1d(x_old, noise_matrix, kind='quadratic', axis=1) # spatial interpolator
#         noise_matrix_int_x = interpolator_x(x_new)
#         interpolator_t = interp1d(t_old, noise_matrix_int_x, kind='quadratic', axis=0) # temporal interpolator
#         noise_matrix_int_tx = interpolator_t(t_new)
#         return noise_matrix_int_tx
#     x_new = jnp.linspace(0, mesh_size, mesh_size)
#     t_new = jnp.linspace(0, c.T, int(np.ceil(c.T/c.tr)))
#     return jnp.array(func_Qs(x_new, t_new))
#     # return func_Qs

def load_Qs_noise(c, mesh_size): 
    noise_matrix = jnp.load(c.Qs_path(xn=mesh_size, tn=int(jnp.ceil(c.T_FEM/c.dt))))
    return noise_matrix
    # return func_Qs

def scipy_interp(x_old, t_old, noise_matrix, x_new, t_new):
    interpolator_x = interp1d(x_old, noise_matrix, kind='quadratic', axis=1)
    noise_matrix_int_x = interpolator_x(x_new)
    interpolator_t = interp1d(t_old, noise_matrix_int_x, kind='quadratic', axis=0)
    noise_matrix_int_tx = interpolator_t(t_new)
    return noise_matrix_int_tx

def load_Qs_noise_callback(c, mesh_size): 
    x_old = jnp.linspace(0, mesh_size, c.XN)
    t_old = jnp.linspace(0, c.T_FEM, c.TN)
    noise_matrix = np.load(c.Qs_path) 

    def func_Qs(t_new, x_new):
        return scipy_interp(x_old, t_old, noise_matrix, x_new, t_new)

    @jax.jit
    def jax_func_Qs(t_new, x_new):
        result_shape = jax.core.ShapedArray((t_new.shape, x_new.shape), 'float32')
        return jax.pure_callback(func_Qs, result_shape, t_new, x_new)

    return jax_func_Qs

def downsample_data(t_star, u_ref, coords, Qs, cd):
    files = lambda t, dt=cd.dt: int(t/dt) # temporal resolution of files [a.u.]
    if cd.parcellations_to_use == -1:
        cd.parcellations_to_use = u_ref.shape[1]
    t_star_p = t_star[files(cd.t_avoid):files(cd.T):files(cd.tr)]
    u_ref_p = u_ref[files(cd.t_avoid):files(cd.T):files(cd.tr), cd.parcellations_avoid:cd.parcellations_to_use:cd.use_every_voxel]
    Qs_p = Qs[files(cd.t_avoid):files(cd.T):files(cd.tr), cd.parcellations_avoid:cd.parcellations_to_use:cd.use_every_voxel]
    coords_p = coords[cd.parcellations_avoid:cd.parcellations_to_use:cd.use_every_voxel, :]
    return t_star_p, u_ref_p, coords_p, Qs_p

def downsample_mesh(curv, normals, u_ref, cd):
    if cd.ncurv_path is not None:
        if cd.parcellations_to_use == -1:
            cd.parcellations_to_use = u_ref.shape[1]
        curv = curv[cd.parcellations_avoid:cd.parcellations_to_use:cd.use_every_voxel]
        normals = normals[cd.parcellations_avoid:cd.parcellations_to_use:cd.use_every_voxel, :]
    return curv, normals

def create_permutation(arr, key=jrandom.PRNGKey(22), permutation=None, ax=1):
    if permutation is None:
        permutation = jrandom.permutation(key, arr.shape[ax])
    shuffled_arr = jnp.take(arr, permutation, axis=ax)
    return shuffled_arr, permutation

def inverse_permutation(arr, permutation, ax=1):
    inverse_permutation = jnp.argsort(permutation)
    unshuffled_arr = jnp.take(arr, inverse_permutation, axis=ax)
    return unshuffled_arr

def load_data(cd): 
    data = jnp.load(cd.data_path, allow_pickle=True).item()
    u_ref = jnp.array(data["phi_e"]) # T, p
    Qs = jnp.array(data["Qs"]) # T, p
    coords = jnp.array(data["mesh_coordinates"]) # p, 3
    t_star = jnp.array(data["t_star"]) # T
    return [t_star, u_ref, coords, Qs]

def load_mesh_info(coords, cd):
    if cd.ncurv_path is not None:
        if cd.ncurv_path == 'unit_sphere':
            curv = jnp.ones(coords.shape[0]) # constant curvature of 1 for unit sphere
            normals = jnp.copy(coords)
        else:
            ncurv_data = jnp.load(cd.ncurv_path, allow_pickle=True).item()
            curv = jnp.array(ncurv_data['curv']) # (p,)
            normals = jnp.array(ncurv_data['norm']) # (p, 3)
    else:
        curv, normals = None, None
    return curv, normals

def log_data_change(t_star_p, u_ref_p, coords_p, Qs_p, t_star, u_ref, coords, Qs, cd):
    print('\nINFO: Data was cut and permuted:')
    print(f'Time was cut from: {len(t_star)} -> {len(t_star_p)}')
    print(f'---- from: {len(t_star)*cd.tr:.2f}s -> {len(t_star_p)*cd.tr:.2f}s')
    print(f'Voxels were cut from: {u_ref.shape[1]} -> {u_ref_p.shape[1]}\n')
    print(f'Shape of arrays:\nu_ref:{u_ref_p.shape}\nt_star:{t_star_p.shape}\ncoords:{coords_p.shape}\n')

def permute_data(t_star, u_ref, coords, Qs, config_dataset):
    if config_dataset.permute_data:
        u_ref, permutation = create_permutation(arr=u_ref)
        Qs, _ = create_permutation(arr=Qs, permutation=permutation)
        coords, _ = create_permutation(arr=coords, ax=0, permutation=permutation)
    return t_star, u_ref, coords, Qs

def get_dataset(config_dataset):
    data = load_data(cd=config_dataset)
    processed_data = downsample_data(*data, cd=config_dataset)
    mesh_strct = load_mesh_info(data[2], cd=config_dataset)
    mesh_strct = downsample_mesh(*mesh_strct, data[1], cd=config_dataset)
    permutted_data = permute_data(*processed_data, config_dataset)    
    log_data_change(*permutted_data, *data, config_dataset)
    return *permutted_data, *mesh_strct #t_star, u_ref, Qs, coords, curv, normals


# array = jnp.array([[2,1, 2], [1, 4, 2], [1,4, 3], [9, 4, 2], [6, 4, 2]])
# i = jnp.array([1, 4, 2])
# array = jnp.array([3, 1, 4, 1, 5, 9, 2, 6, 3])
# i = jnp.array(5)
# index = find_idx(array, i)
# print(index)  

# Example usage and testing: 
from configs.default import get_config
config = get_config()
get_dataset(config.data)


# x_new = jnp.linspace(0, u_ref.shape[1], u_ref.shape[1])
# t_new = jnp.linspace(0, config.data.T, int(np.ceil(config.data.T / config.data.tr)))

# jax_func_Qs = load_Qs_noise_callback(config.data, u_ref.shape[1])
# interpolated_noise = jax_func_Qs(t_new, x_new)
# print(interpolated_noise)

# XN_ = u_ref.shape[1] # number of space samples after interpolating
# TN_ = int(np.ceil(config.data.T/config.data.dt)) # number of time samples after interpolating

# x_new = np.linspace(0, XN_, XN_)
# t_new = np.linspace(0, config.data.T, TN_)

# noise = Qs_func(x_new, t_new)
# print(noise)
# print(noise.shape)

# print(t_star)
# t_star_expanded = jnp.tile(t_star, (coords.shape[0] // t_star.size,))
# t_star_expanded = jnp.append(t_star_expanded, t_star[:(coords.shape[0] % t_star.size)], axis=0)

# gamma = 150
# r = 40 
# noise = 0.1

# t0 = t_star[0]
# t1 = t_star[-1]

# # Define domain
# temporal_dom = jnp.array([t0, t1])

# print((u_ref[-1, -1]/(gamma * t_star[-1])**2) / noise)
# print(u_ref[-1, -1]/(gamma * t_star[-1] * noise))
# print(u_ref[-1, -1] / noise)

# # normalise along each feature (parcellation) - subtract average in time for each parcellation
# u_ref, U_star = nondimensionalise(u_ref, default=True)
# # subtract average of each dimension
# coords, L_star = nondimensionalise(coords, default=True)
# # u norm
# noise = noise / U_star
# # t norm
# gamma = gamma * t1
# # coords norm
# r = r / L_star

# t_star = t_star / t1
# print((u_ref[-1, -1]/(gamma * t_star[-1])**2) / noise)
# print(u_ref[-1, -1]/(gamma * t_star[-1] * noise))
# print(u_ref[-1, -1] / noise)

# shuffled_u, permutation = create_permutation(arr=u_ref)
# unshuffled_u = inverse_permutation(shuffled_u, permutation)

# shuffled_coords, _ = create_permutation(arr=coords, ax=0, permutation=permutation)
# unshuffled_coords = inverse_permutation(shuffled_coords, permutation, ax=0)




#### Example: vmap

# None: not map, axis out (default = 0): where mapped axis appears in the output
# nested vmaps: compute solution to func for each t[i] for every x[j], y[j], z[j]
# apply r_net over every combination of (t,x,y,z) with params constant

# import jax.numpy as jnp
# from jax import vmap

# vv = lambda x, y: jnp.vdot(x, y)

# # applies vv along 0th axis, and keeps 1st axis constant.
# mv = vmap(vv, (0, None))

# x = jnp.array([[1, 2], [3, 4]])
# y = jnp.array([2, 3])

# result = mv(x, y)
# print(result) # [1*2 + 2*3, 2*3 + 4*3] = [8, 18]