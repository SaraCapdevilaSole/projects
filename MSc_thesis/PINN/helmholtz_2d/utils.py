import jax.numpy as jnp
import jax.random as jrandom
from jax import lax

import os

import jax.numpy as jnp
import jax.random as jrandom
from jax import lax
import os

# Constants: Frequency bands for EEG signals
BANDS_MEAN = {
    "delta": 2,
    "theta": 6,
    "alpha": 12,
    "beta": 22,
    "gamma": 40,
    "gamma high": 75,
    "gamma very high": 125,
}

# Random EEG source coordinates
SOURCES = [
    (0.0297, 0.8716),
    (0.2149, 0.7388), 
    (0.7966, 0.6171),
    (0.3712, 0.2200), 
    (0.8526, 0.3986), 
    (0.6292, 0.9127), 
    (0.5460, 0.1876), 
    (0.2355, 0.1089),
    (0.8183, 0.9187), 
]

# Random speed of sources
SPEED_SOURCES = jnp.array([
    (0.0093, -0.0307), 
    (0.0829, 0.0185), 
    (0.0240, 0.0814), 
    (-0.0880, 0.0679), 
    (0.0913, -0.0482), 
    (0.0015, -0.0139), 
    (0.0925, -0.0152), 
    (0.0978, -0.0861), 
    (-0.0885, -0.0281), 
    (0.0531, -0.0480)
]) * 10

FREQUENCIES_PINK = jnp.arange(1, 85, 5)  # Pink noise frequencies (1 to 80 with spacing of 5)


def phase(x, y, t, x0, y0, vs_x, vs_y, velocity=0.1):
    """Calculate phase based on distance from moving source."""
    x0_t, y0_t = x0 + vs_x * t, y0 + vs_y * t
    return jnp.sqrt((x - x0_t) ** 2 + (y - y0_t) ** 2) / velocity


def space_time_signal(t, x, y, noise, alpha=1, velocity=0.1):
    """Generates a space-time signal using pink noise and moving sources."""
    amplitude = noise / (len(FREQUENCIES_PINK) * len(SOURCES))
    signal = amplitude * sum(
        sum(
            (1 / (freq ** alpha)) * jnp.sin(2 * jnp.pi * freq * (t + phase(x, y, t, *source, *speed, velocity)))
            for source, speed in zip(SOURCES, SPEED_SOURCES)
        )
        for freq in FREQUENCIES_PINK
    )
    return signal


def find_idx(array, target):
    """Find the index of a target in an array."""
    if array.ndim == 1:
        matches = array == target
    else:
        target = target.reshape(1, -1)
        matches = jnp.all(array == target, axis=1)
    indices = jnp.where(matches, size=1)[0]
    idx = jnp.where(indices.size > 0, indices[0], -1)
    return idx


def create_dir(wandb_name, wandb_dir='/vol/bitbucket/sc3719/JAXPI'):
    """Create a directory for storing experiment results."""
    wandb_dir = os.path.expanduser(f'{wandb_dir}/{wandb_name}')
    print(f"\nAttempting to create directory: {wandb_dir}")
    os.makedirs(wandb_dir, exist_ok=True)
    
    if not os.access(wandb_dir, os.W_OK):
        print("\nWARNING: Directory is not writable!\n")
    return wandb_dir


def downsample_data(t_star, u_ref, coords, Qs, u_t, u_tt, cd):
    """Downsample temporal and spatial data for training."""
    def files(t, dt=cd.dt):
        return int(t / dt)
    
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
    """Downsample mesh curvature and normals."""
    curv = curv[:cd.parcellations_to_use:cd.use_every_voxel]
    normals = normals[:cd.parcellations_to_use:cd.use_every_voxel, :]
    return curv, normals

def create_permutation(arr, key=jrandom.PRNGKey(22), permutation=None, ax=1):
    """Create a random permutation of the array."""
    if permutation is None:
        permutation = jrandom.permutation(key, arr.shape[ax])
    shuffled_arr = jnp.take(arr, permutation, axis=ax)
    return shuffled_arr, permutation

def inverse_permutation(arr, permutation, ax=1):
    """Invert the permutation of an array."""
    inverse_permutation = jnp.argsort(permutation)
    unshuffled_arr = jnp.take(arr, inverse_permutation, axis=ax)
    return unshuffled_arr

def load_data(config_data):
    """Load the dataset based on the provided configuration."""
    data = jnp.load(config_data.data_path, allow_pickle=True).item()
    u_ref = jnp.array(data["phi_e"])
    Qs = jnp.array(data["Qs"])
    coords = jnp.array(data["mesh_coordinates"])
    t_star = jnp.array(data["t_star"])
    
    if config_data.sobolev_loss:
        u_t_ref = jnp.array(data["p_t"])
        u_tt_ref = jnp.array(data["p_tt"])
        return t_star, u_ref, coords, Qs, u_t_ref, u_tt_ref
    
    return t_star, u_ref, coords, Qs, None, None

def load_mesh_info(config_data):
    """Load mesh curvature and normals."""
    curv = jnp.loadtxt(config_data.curv_path, delimiter=',')
    normals = jnp.loadtxt(config_data.norm_path, delimiter=',')
    return jnp.array(curv), jnp.array(normals)

def log_data_change(t_star_processed, u_ref_processed, coords_processed, t_star, u_ref, config_data):
    """Log information about the data processing steps."""
    print('\nINFO: Data has been downsampled and processed:')
    print(f'Time downsampled from {len(t_star)} -> {len(t_star_processed)} steps')
    print(f'Voxels reduced from {u_ref.shape[1]} -> {u_ref_processed.shape[1]} voxels\n')
    print(f'Shape of processed data:\n  u_ref: {u_ref_processed.shape}\n  t_star: {t_star_processed.shape}\n  coords: {coords_processed.shape}\n')

def permute_data(t_star, u_ref, coords, config_dataset):
    """Optionally shuffle the data during training."""
    if config_dataset.permute_data:
        u_ref, permutation = create_permutation(arr=u_ref)
        coords, _ = create_permutation(arr=coords, ax=0, permutation=permutation)
    return t_star, u_ref, coords

def get_dataset(config_dataset):
    """Main function to load and process the dataset."""
    data = load_data(cd=config_dataset)
    processed_data = downsample_data(*data, cd=config_dataset)
    log_data_change(*processed_data[:3], *data[:2], config_dataset)
    return processed_data 