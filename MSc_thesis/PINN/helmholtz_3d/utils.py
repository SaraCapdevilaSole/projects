import jax.numpy as jnp
import jax.random as jrandom
from jax import lax
import numpy as np
from scipy.interpolate import interp1d
import jax
import os

SPEED_SOURCES_3D = np.array([
    (-0.0899, -0.0089, 0.0402),
    (0.0581, 0.0585, -0.0210),
    (0.0194, -0.0268, -0.0827),
    (-0.0844, 0.0740, -0.0715),
    (-0.0608, -0.0555, 0.0881),
    (0.0685, -0.0485, -0.0251),
    (-0.0343, 0.0163, -0.0951),
    (-0.0119, -0.0114, 0.0132),
    (0.0682, 0.0717, -0.0689),
    (-0.0156, -0.0423, -0.0485),
]) * 10

SOURCES_3D = np.array([
    (-0.1541, 0.1761, -0.8403),
    (-0.1597, -0.5593, 0.7727),
    (-0.8702, -0.0296, 0.2933),
    (0.4439, -0.3589, 0.6546),
    (-0.3468, -0.7902, -0.0572),
    (-0.3629, 0.6344, -0.4829),
    (0.7678, -0.2688, -0.0906),
    (-0.2025, 0.5476, -0.7305),
    (0.3538, -0.6201, 0.0043),
    (-0.0218, -0.0650, 0.9105),
])

SPACING = 5
FREQUENCIES_PINK = jnp.arange(1, 80 + SPACING, SPACING)


# Utility
def create_dir(wandb_name, base_dir='/vol/bitbucket/sc3719/JAXPI'):
    """
    Creates a directory for saving data. Ensures write permission.
    """
    dir_path = f'{base_dir}/{wandb_name}'
    print(f"Attempting to create/access directory: {dir_path}")

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if not os.access(dir_path, os.W_OK):
        print("WARNING: No write permission to the directory!")
    
    return dir_path

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

# Noise Functions
def phase(coords, t, coords0, coords_v, sphere_radius, v, normalize=False):
    """
    Computes the phase shift between moving and fixed coordinates.
    """
    coords0_t = [x0 + v_x * t for x0, v_x in zip(coords0, coords_v)]
    
    if normalize:
        norm0_t = jnp.linalg.norm(jnp.array(coords0_t))
        norm0 = jnp.linalg.norm(jnp.array(coords0))
        coords0_t = [x0_t / norm0_t * norm0 for x0_t in coords0_t]
    
    return jnp.sqrt(sum((x - x0_t)**2 for x, x0_t in zip(coords, coords0_t))) / v


def find_sources(scale_factors):
    """
    Scales the source and speed source arrays based on scaling factors.
    """
    scaling_sources, scaling_speeds = scale_factors
    sources = SOURCES_3D * scaling_sources
    speed_sources = SPEED_SOURCES_3D * scaling_speeds
    return sources, speed_sources


def space_time_signal(t, x, y, z, noise, sphere_radius, freq_denom, scale_factors, normalize, velocity, alpha=1):
    """
    Generates a signal based on the pink noise model across space and time.
    """
    sources, speed_sources = find_sources(scale_factors)
    amplitude = noise / (len(FREQUENCIES_PINK) * len(sources))
    
    signal = amplitude * sum(
        sum(
            (1 / (f**alpha)) * jnp.sin(
                2 * jnp.pi * f * (t + phase([x, y, z], t, source, speed, sphere_radius, velocity, normalize)) / freq_denom
            )
            for source, speed in zip(sources, speed_sources)
        )
        for f in FREQUENCIES_PINK
    )
    return signal

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


# Loading and Preprocesing
def load_Qs_noise(c, mesh_size):
    """
    Loads Qs noise data and interpolates based on mesh size.
    """
    noise_matrix = jnp.load(c.Qs_path(xn=mesh_size, tn=int(jnp.ceil(c.T_FEM / c.dt))))
    return noise_matrix


def downsample_mesh(curv, normals, u_ref, cd):
    if cd.ncurv_path is not None:
        if cd.parcellations_to_use == -1:
            cd.parcellations_to_use = u_ref.shape[1]
        curv = curv[cd.parcellations_avoid:cd.parcellations_to_use:cd.use_every_voxel]
        normals = normals[cd.parcellations_avoid:cd.parcellations_to_use:cd.use_every_voxel, :]
    return curv, normals


def downsample_data(t_star, u_ref, coords, Qs, cd):
    """
    Downsamples the dataset based on configuration parameters.
    """
    files = lambda t, dt=cd.dt: int(t / dt)
    
    if cd.parcellations_to_use == -1:
        cd.parcellations_to_use = u_ref.shape[1]

    t_star_p = t_star[files(cd.t_avoid):files(cd.T):files(cd.tr)]
    u_ref_p = u_ref[files(cd.t_avoid):files(cd.T):files(cd.tr), cd.parcellations_avoid:cd.parcellations_to_use:cd.use_every_voxel]
    Qs_p = Qs[files(cd.t_avoid):files(cd.T):files(cd.tr), cd.parcellations_avoid:cd.parcellations_to_use:cd.use_every_voxel]
    coords_p = coords[cd.parcellations_avoid:cd.parcellations_to_use:cd.use_every_voxel, :]

    return t_star_p, u_ref_p, coords_p, Qs_p


def create_permutation(arr, key=jrandom.PRNGKey(22), permutation=None, axis=1):
    """
    Permutates an array along the specified axis.
    """
    if permutation is None:
        permutation = jrandom.permutation(key, arr.shape[axis])
    return jnp.take(arr, permutation, axis=axis), permutation


def inverse_permutation(arr, permutation, axis=1):
    """
    Restores an array to its original order using the inverse permutation.
    """
    inverse_perm = jnp.argsort(permutation)
    return jnp.take(arr, inverse_perm, axis=axis)


def load_data(cd):
    """
    Loads the dataset containing time, reference data, coordinates, and Qs noise.
    """
    data = jnp.load(cd.data_path, allow_pickle=True).item()
    u_ref = jnp.array(data["phi_e"])
    Qs = jnp.array(data["Qs"])
    coords = jnp.array(data["mesh_coordinates"])
    t_star = jnp.array(data["t_star"])
    return t_star, u_ref, coords, Qs


def load_mesh_info(cd):
    """
    Loads the mesh information.
    """
    if cd.ncurv_path is not None:
        ncurv_data = jnp.load(cd.ncurv_path, allow_pickle=True).item()
        curv = jnp.array(ncurv_data['curv']) # (p,)
        normals = jnp.array(ncurv_data['norm']) # (p, 3)
    else:
        curv, normals = None, None
    return curv, normals


# Logging and Metadata
def log_data_change(t_star_p, u_ref_p, coords_p, Qs_p, t_star, u_ref, coords, Qs, cd):
    """
    Logs information on data changes and reductions after downsampling.
    """
    print('\nINFO: Data has been downsampled and permuted:')
    print(f'Time reduced from {len(t_star)} -> {len(t_star_p)} ({len(t_star) * cd.tr:.2f}s -> {len(t_star_p) * cd.tr:.2f}s)')
    print(f'Voxels reduced from {u_ref.shape[1]} -> {u_ref_p.shape[1]}')
    print(f'Shapes: u_ref={u_ref_p.shape}, t_star={t_star_p.shape}, coords={coords_p.shape}\n')


def permute_data(t_star, u_ref, coords, Qs, config_dataset):
    """
    Optionally permutes the dataset based on configuration settings.
    """
    if config_dataset.permute_data:
        u_ref, permutation = create_permutation(arr=u_ref)
        Qs, _ = create_permutation(arr=Qs, permutation=permutation)
        coords, _ = create_permutation(arr=coords, axis=0, permutation=permutation)
    
    return t_star, u_ref, coords, Qs


# Main Dataset Handler
def get_dataset(config_dataset):
    """
    Prepares the dataset by loading, downsampling, and optionally permuting the data.
    """
    data = load_data(cd=config_dataset)
    processed_data = downsample_data(*data, cd=config_dataset)
    permuted_data = permute_data(*processed_data, config_dataset)
    log_data_change(*permuted_data, *data, config_dataset)
    return permuted_data

