import jax.numpy as jnp
import jax.random as jrandom
from jax import lax

import os

# def find_idx(matrix, target_row):
#     """jax tracer method for jnp.where; x5 faster"""
#     def body_fun(carry, index):
#         found, idx = carry
#         row = matrix[index]
#         condition = jnp.all(row == target_row)
#         found = lax.cond(condition, lambda _: True, lambda _: found, None)
#         idx = lax.cond(condition, lambda _: index, lambda _: idx, None)
#         return (found, idx), None

#     init_val = (False, -1)
#     result, _ = lax.scan(body_fun, init_val, jnp.arange(matrix.shape[0]))
#     _, idx = result
#     return idx

# bands_mean = {
#     "delta": 2,
#     "theta": 6,
#     "alpha": 12,
#     "beta": 22,
#     "gamma": 40,
#     "gamma high": 75,
#     "gamma very high": 125,
# }

# spatial and temporal signal
# def generate_spatial_signal(t, x, y, noise, bands_mean=bands_mean):
#     """Generates a composite signal given time, frequency bands, and noise level."""
#     v_x, v_y = 0.001, 0.002 # v [m/s] := gamma [/s] * r [m]
#     rs = 10
#     N = len(bands_mean)
#     a0 = noise / N
#     signal = a0 * sum(
#         jnp.sin(2 * jnp.pi * freq * (t * v_x - x)/rs) * jnp.sin(2 * jnp.pi * freq * (t * v_y - y)/rs)
#         for freq in bands_mean.values()
#     )
#     return signal 

# ONLY temporal signal
# def generate_composite_signal(t, noise, bands_mean=bands_mean):
#     """Generates a composite signal given time, frequency bands, and noise level."""
#     signal = jnp.zeros_like(t)
#     N = len(bands_mean)
#     for freq in bands_mean.values():
#         # TODO: add spatial dependency
#         signal += noise / N * jnp.sin(2 * jnp.pi * freq * t)
#     return signal


sources = [
    (0.029718512853793344, 0.8716452743203177),
    (0.21498408231411958, 0.7388398620023181), 
    (0.7966767681926875, 0.6171640367876514),
    (0.37128565670282987, 0.22007066406771558), 
    (0.8526190238138224, 0.39868153519242533), 
    (0.6292637997263526, 0.9127000046877554), 
    (0.5460196221875436, 0.18761271119370937), 
    (0.23557099720366548, 0.10894846451508955),
    (0.8183231022581631, 0.9187431685788614), 
]

speed_sources = jnp.array([
    (0.009321947790091989, -0.030790495763255196), 
    (0.08299397351448928, 0.018574396994712722), 
    (0.024070081562605303, 0.08141237022889888), 
    (-0.08806395242434038, 0.06796280386378259), 
    (0.09132779567073915, -0.04828039163215658), 
    (0.001551250632302839, -0.01391199301498433), 
    (0.09258741245705379, -0.015208889749087046), 
    (0.09780450156428427, -0.08610253403192539), 
    (-0.08858189316897747, -0.028171353302825705), 
    (0.05318691362062716, -0.04807110182762042)
]) * 10

spacing = 5
frequencies_pink = jnp.arange(1, 80 + spacing, spacing)

# Pink noise
def phase(x, y, t, x0, y0, vs_x, vs_y, v=0.1):
    x0_t, y0_t = x0 + vs_x * t, y0 + vs_y * t # position of source varies in time
    # v [m/s] := gamma [/s] * r [m] (10*0.001=0.01) * 10 # too costly
    return jnp.sqrt((x-x0_t)**2 + (y-y0_t)**2) / v

def space_time_signal(t, x, y, noise, alpha=1, v=0.1): 
    a0 = noise/(len(frequencies_pink)*len(sources))
    Qs = a0 * sum(
        sum(
            1/(f**alpha) * jnp.sin(2 * jnp.pi * f * (t + phase(x, y, t, *source, *speed, v))) 
            for source, speed in zip(sources, speed_sources)
        )
        for f in frequencies_pink
    )
    return Qs

def find_idx(array, target):
    """find the index of target in array."""
    if array.ndim == 1:
        # 1D vector
        matches = array == target
    else:
        # 2D matrix
        target = target.reshape(1, -1)
        matches = jnp.all(array == target, axis=1)
    indices = jnp.where(matches, size=1)[0]
    idx = jnp.where(indices.size > 0, indices[0], -1)
    return idx

def create_dir(wandb_name, wandb_dir = '~/../../vol/bitbucket/sc3719/JAXPI'):
    wandb_dir = f'{wandb_dir}/{wandb_name}'
    wandb_dir = os.path.expanduser(wandb_dir)

    print(f"\nTrying directory: {wandb_dir}")
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)
    if not os.access(wandb_dir, os.W_OK):
        print("\nWARNING: You do not have write permission to this directory!\n")
    return wandb_dir

def downsample_data(t_star, u_ref, coords, Qs, u_t, u_tt, cd):
    files = lambda t, dt=cd.dt: int(t/dt) # temporal resolution of files [a.u.]
    if cd.parcellations_to_use == -1:
        cd.parcellations_to_use = u_ref.shape[1]
    t_star = t_star[files(cd.t_avoid):files(cd.T):files(cd.tr)]
    u_ref = u_ref[files(cd.t_avoid):files(cd.T):files(cd.tr), :cd.parcellations_to_use:cd.use_every_voxel]
    if len(Qs) == 1:
        Qs = Qs[files(cd.t_avoid):files(cd.T):files(cd.tr), :cd.parcellations_to_use:cd.use_every_voxel]
    if cd.sobolev_loss:
        u_t = u_t[files(cd.t_avoid):files(cd.T):files(cd.tr), :cd.parcellations_to_use:cd.use_every_voxel]
        u_tt = u_tt[files(cd.t_avoid):files(cd.T):files(cd.tr), :cd.parcellations_to_use:cd.use_every_voxel]
    coords = coords[:cd.parcellations_to_use:cd.use_every_voxel, :]
    return t_star, u_ref, coords, Qs, u_t, u_tt

def downsample_mesh(curv, normals, cd):
    curv = curv[:cd.parcellations_to_use:cd.use_every_voxel]
    normals = normals[:cd.parcellations_to_use:cd.use_every_voxel, :]
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
    if cd.sobolev_loss:
        u_t_ref = jnp.array(data["p_t"]) # T, p
        u_tt_ref = jnp.array(data["p_tt"]) # T, p
        return [t_star, u_ref, coords, Qs, u_t_ref, u_tt_ref]
    else:
        return [t_star, u_ref, coords, Qs, None, None]

def load_mesh_info(cd):
    curv = jnp.loadtxt(cd.curv_path, delimiter=',')
    normals = jnp.loadtxt(cd.norm_path, delimiter=',')
    curv = jnp.array(curv) # (p,)
    normals = jnp.array(normals) # (p, 3)
    return curv, normals

def log_data_change(t_star_p, u_ref_p, coords_p, t_star, u_ref, cd):
    print('\nINFO: Data was cut and permuted:')
    print(f'Time was cut from: {len(t_star)} -> {len(t_star_p)}')
    print(f'---- from: {len(t_star)*cd.tr:.2f}s -> {len(t_star_p)*cd.tr:.2f}s')
    print(f'Voxels were cut from: {u_ref.shape[1]} -> {u_ref_p.shape[1]}\n')
    print(f'Shape of arrays:\nu_ref:{u_ref_p.shape}\nt_star:{t_star_p.shape}\ncoords:{coords_p.shape}\n')

def permute_data(t_star, u_ref, coords, config_dataset):
    if config_dataset.permute_data:
        u_ref, permutation = create_permutation(arr=u_ref)
        coords, _ = create_permutation(arr=coords, ax=0, permutation=permutation)
    return t_star, u_ref, coords

def get_dataset(config_dataset):
    data = load_data(cd=config_dataset)
    # mesh_strct = load_mesh_info(cd=config_dataset)
    processed_data = downsample_data(*data, cd=config_dataset)
    # mesh_strct = downsample_mesh(*mesh_strct, cd=config_dataset)
    # permutted_data = permute_data(*processed_data, config_dataset)    
    log_data_change(*processed_data[:3], *data[:2], config_dataset)
    return processed_data #t_star, u_ref, Qs, coords, curv, normals

# import jax
# import jax.numpy as jnp
# from jax import lax
# import time

# st = time.time()
# matrix = jnp.array([[2,1, 2], [1, 4, 2], [1,4, 3], [9, 4, 2], [6, 4, 2]])
# target_row = matrix[2]
# matrix = jnp.array([3, 1, 4, 1, 5, 9, 2, 6, 3])
# target_row = matrix[3]
# index = find_idx(matrix, target_row)
# et = time.time()
# print(index)  


# Example usage and testing: 
# from configs.default import get_config
# config = get_config()
# t_star, u_ref, coords, Qs, _, _ = get_dataset(config.data)
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