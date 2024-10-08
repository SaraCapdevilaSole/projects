import jax.numpy as jnp
import jax.random as jrandom
from jax import lax
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import KDTree
from scipy.io import loadmat
import jax
import os


# Define speed and position of sources in 3D space
SPEED_SOURCES_3D = np.array([
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
]) * 10  # Scale speeds by a factor of 10

SOURCES_3D = [
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

FREQUENCIES_PINK = jnp.arange(1, 85, 5)

# Pink noise
def phase(coords, t, coords0, coords_v, sphere_radius, v, norm=False):
    """Calculate the phase difference based on spatial coordinates and time."""
    coords0_t = [x0 + vs_x * t for x0, vs_x in zip(coords0, coords_v)]
    
    if norm:
        norm0_t = jnp.linalg.norm(jnp.array(coords0_t))
        norm0 = jnp.linalg.norm(jnp.array(coords0))
        coords0_t = [x0_t / norm0_t * norm0 for x0_t in coords0_t]
    
    return jnp.sqrt(sum((x - x0_t) ** 2 for x, x0_t in zip(coords, coords0_t))) / v

def find_sources(x, mult):
    """Scale sources and their speeds based on given multipliers."""
    mr, ms = mult
    sources = SOURCES_3D * mr
    speed_sources = SPEED_SOURCES_3D * ms
    return sources, speed_sources

def space_time_signal(t, x, y, z, noise, sphere_radius, freq_denom, mult, norm, v, alpha=1): 
    """Generate a spatiotemporal signal using pink noise."""
    sources, speed_sources = find_sources(x, mult)
    a0 = noise / (len(FREQUENCIES_PINK) * len(sources))
    
    Qs = a0 * sum(
        sum(
            1 / (f ** alpha) * jnp.sin(2 * jnp.pi * f * (t + phase([x, y, z], t, source, speed, sphere_radius, v, norm)) / freq_denom) 
            for source, speed in zip(sources, speed_sources)
        )
        for f in FREQUENCIES_PINK
    )
    return Qs


# Alternative Noise
bands_mean = {
    "delta": 2,
    "theta": 6,
    "alpha": 12,
    "beta": 22,
    "gamma": 40,
    "gamma high": 75,
    "gamma very high": 125,
}

@jax.jit
def find_frequency_bands(min_f, max_f, step_f, mult_f, max_steps=20):
    """lax for loop for: jnp.array(min_f, max_f, step_f) with varying step"""
    def scan_body(carry, _):
        f, i, step_f, f_arr = carry
        f_arr = jax.lax.cond(
            f <= max_f,
            lambda x: f_arr.at[i].set(f),
            lambda x: f_arr,
            None
        )
        f += step_f
        i += 1
        step_f = step_f + mult_f * i                  
        return (f, i, step_f, f_arr), None
    
    init_val = (min_f, 0, step_f, jnp.full(max_steps, min_f))
    (final_f, final_i, final_step_f, final_f_arr), _ = lax.scan(scan_body, init_val, None, length=max_steps)
    
    return final_f_arr

def generate_temporal_signal(t, alpha, min_f, max_f, step_f, mult_f): #, bands_names=bands_mean.keys()):
    """Generates a composite signal given time, frequency bands, and noise level."""
    selected_bands = find_frequency_bands(min_f, min_f + max_f, step_f, mult_f) #find_bands(bands_names, max_frequency)
    signal = sum(
        1/(freq**alpha) * jnp.sin(2 * jnp.pi * freq * t) 
        for freq in selected_bands#.values()
    )
    return signal


# Utility
def get_idxs(model_terms):
    term_indices = {}
    index = 1  
    for term, is_active in model_terms.items():
        if is_active:
            term_indices[term] = index
            index += 1
    if index == 1:
        return None
    else:
        return term_indices

def find_idx(matrix, target_row):
    """Find the index of a target row in a matrix using JAX."""
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

def create_dir(wandb_name, wandb_dir='/vol/bitbucket/sc3719/JAXPI'):
    """Create a directory for storing outputs, ensuring write permissions."""
    wandb_dir = os.path.join(wandb_dir, wandb_name)
    print(f"\nTrying directory: {wandb_dir}")
    
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)
    
    if not os.access(wandb_dir, os.W_OK):
        print("\nWARNING: You do not have write permission to this directory!\n")
    
    return wandb_dir

def load_Qs_noise(c, mesh_size): 
    """Load noise matrix from a specified path."""
    noise_matrix = jnp.load(c.Qs_path(xn=mesh_size, tn=int(jnp.ceil(c.T_FEM / c.dt))))
    return noise_matrix

def scipy_interp(x_old, t_old, noise_matrix, x_new, t_new):
    """Perform spatial and temporal interpolation on the noise matrix."""
    interpolator_x = interp1d(x_old, noise_matrix, kind='quadratic', axis=1)
    noise_matrix_int_x = interpolator_x(x_new)
    interpolator_t = interp1d(t_old, noise_matrix_int_x, kind='quadratic', axis=0)
    noise_matrix_int_tx = interpolator_t(t_new)
    return noise_matrix_int_tx

def load_Qs_noise_callback(c, mesh_size): 
    """Create a JAX-compatible callback for loading and interpolating noise."""
    x_old = jnp.linspace(0, mesh_size, c.XN)
    t_old = jnp.linspace(0, c.T_FEM, c.TN)
    noise_matrix = np.load(c.Qs_path) 

    def func_Qs(t_new, x_new):
        return scipy_interp(x_old, t_old, noise_matrix, x_new, t_new)

    @jax.jit
    def jax_func_Qs(t_new, x_new):
        result_shape = jax.core.ShapedArray((t_new.shape[0], x_new.shape[0]), 'float32')
        return jax.pure_callback(func_Qs, result_shape, t_new, x_new)

    return jax_func_Qs

# Data Preprocessing
def downsample_data(t_star, u_ref, coords, Qs, cd):
    """Downsample temporal and spatial data based on specified parameters."""
    files = lambda t, dt=cd.dt: int(t / dt)  # Temporal resolution of files [a.u.]
    
    if cd.parcellations_to_use == -1:
        cd.parcellations_to_use = u_ref.shape[1]
    
    t_star_p = t_star[files(cd.t_avoid):files(cd.T):files(cd.tr)]
    u_ref_p = u_ref[files(cd.t_avoid):files(cd.T):files(cd.tr), 
                     cd.parcellations_avoid:cd.parcellations_to_use:cd.use_every_voxel]
    Qs_p = Qs[files(cd.t_avoid):files(cd.T):files(cd.tr), 
               cd.parcellations_avoid:cd.parcellations_to_use:cd.use_every_voxel]
    coords_p = coords[cd.parcellations_avoid:cd.parcellations_to_use:cd.use_every_voxel, :]
    return t_star_p, u_ref_p, coords_p, Qs_p

def downsample_mesh(curv, normals, cd):
    """Downsample mesh curvature and normal vectors based on user configuration."""
    curv = curv[cd.parcellations_avoid:cd.parcellations_to_use:cd.use_every_voxel]
    normals = normals[cd.parcellations_avoid:cd.parcellations_to_use:cd.use_every_voxel, :]
    return curv, normals

def create_permutation(arr, key=jrandom.PRNGKey(22), permutation=None, ax=1):
    """Create a random permutation of the given array along a specified axis."""
    if permutation is None:
        permutation = jrandom.permutation(key, arr.shape[ax])
    shuffled_arr = jnp.take(arr, permutation, axis=ax)
    return shuffled_arr, permutation

def inverse_permutation(arr, permutation, ax=1):
    """Reorder an array based on the inverse of a given permutation."""
    inverse_permutation = jnp.argsort(permutation)
    unshuffled_arr = jnp.take(arr, inverse_permutation, axis=ax)
    return unshuffled_arr

def load_data_fd(cd): 
    """Load dataset from a specified path (firedrake)."""
    data = jnp.load(cd.data_path, allow_pickle=True).item()
    u_ref = jnp.array(data["phi_e"]) # Shape: (T, p)
    Qs = jnp.array(data["Qs"]) # Shape: (T, p)
    coords = jnp.array(data["mesh_coordinates"]) # Shape: (p, 3)
    t_star = jnp.array(data["t_star"]) # Shape: (T,)
    return [t_star, u_ref, coords, Qs]

def load_res_mesh_coords(data_path): 
    """Load region mesh coordinates from a specified data path."""
    data = jnp.load(data_path, allow_pickle=True).item()
    nn_coords = jnp.array(data['nn_coords'])  # Shape: (data_size, 3)
    res_coords = jnp.array(data['region'])     # Shape: (nodes_in_mesh_precision, 3)
    return nn_coords, res_coords

def read_ecog_data(file_path):
    """Read ECoG data from a MATLAB file."""
    return loadmat(file_path)

def smooth_gaussian(data, sigma=5):
    """Smooth data using a Gaussian filter with a specified sigma."""
    return gaussian_filter1d(data, sigma=sigma)

def smooth_data(data, sigma):
    """Smooth data for each node if sigma is provided; otherwise, return the original data."""
    if sigma is None:
        return data
    else:
        T, nodes = data.shape
        smoothed_data = np.zeros_like(data)
        for node in range(nodes):
            smoothed_data[:, node] = smooth_gaussian(data[:, node], sigma)
        return smoothed_data

def load_real_data(cd, sf=1000): 
    """Load and preprocess real ECoG data."""
    data = read_ecog_data(cd.data_path)
    u_ref = jnp.array(smooth_data(data['data'], cd.sigma_gauss))  # Shape: (T, p)
    
    nn_coords, res_coords = load_res_mesh_coords(cd.res_data_path)
    
    Qs = None  # Placeholder for Qs, Shape: (T, p)
    t_star = jnp.linspace(0, data['data'].shape[0] / sf, data['data'].shape[0])  # Shape: (T,)
    
    return [t_star, u_ref, nn_coords, Qs], res_coords

def load_data(cd): 
    """Load dataset based on the configuration dictionary."""
    if cd.model_terms['real_data']:
        data, res_data = load_real_data(cd)
    else:
        data = load_data_fd(cd)  
        res_data = None
    return data, res_data

def load_mesh_info(cd):
    """Load mesh curvature and normals from specified paths."""
    curv = np.loadtxt(cd.curv_path, delimiter=',')  # Shape: (p,)
    normals = np.loadtxt(cd.norm_path, delimiter=',')  # Shape: (p, 3)
    return jnp.array(curv), jnp.array(normals)

def log_data_change(t_star_p, u_ref_p, coords_p, Qs_p, t_star, u_ref, coords, Qs, cd):
    """Log changes in data dimensions after processing."""
    print('\nINFO: Data was cut and permuted:')
    print(f'Time was cut from: {len(t_star)} -> {len(t_star_p)}')
    print(f'---- from: {len(t_star) * cd.tr:.2f}s -> {len(t_star_p) * cd.tr:.2f}s')
    print(f'Voxels were cut from: {u_ref.shape[1]} -> {u_ref_p.shape[1]}\n')
    print(f'Shape of arrays:\nu_ref: {u_ref_p.shape}\nt_star: {t_star_p.shape}\ncoords: {coords_p.shape}\n')

def permute_data(t_star, u_ref, coords, config_dataset):
    """Permute data arrays if specified in the configuration."""
    if config_dataset.permute_data:
        u_ref, permutation = create_permutation(arr=u_ref)
        coords, _ = create_permutation(arr=coords, ax=0, permutation=permutation)
    return t_star, u_ref, coords

def get_dataset(config_dataset):
    """Load and process the dataset based on the configuration."""
    data, res_data = load_data(cd=config_dataset)

    processed_data = downsample_data(*data, cd=config_dataset)
    log_data_change(*processed_data, *data, config_dataset)
    
    res_data = res_data if res_data is not None else data[2]  # Pass non-downsampled coordinates
    return *processed_data, res_data  # Return: t_star, u_ref, coords, Qs, curv, normals
