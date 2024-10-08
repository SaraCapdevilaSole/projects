from src.utils.utils import find_XN_TN

# General configuration
noise = 0.3 
T = 1e-2
dt = 1e-6
sn = 1e-6

# Mesh configuration
shape = 'real_mesh' #
# shape = 'sphere'  # Options: 'square', 'sphere', 'real_mesh'
if shape == 'square':
    size = 100  # Range [10, 100] for square mesh
    max_size = 100
    config_kwargs = {}
elif shape == 'sphere':
    size = 6  # Range [1, 10] for sphere mesh
    max_size = 7
    sphere_radius = 100  # Physically realistic = 100
    config_kwargs = {'radius': sphere_radius}
elif shape == 'real_mesh':
    size = 164  # thousand points
    max_size = 164  # 164, 131, 80, 65, 40, 32, 20, 3
    mesh_type = 'inflated'
    hemi = 'lh'
    config_kwargs = {
        'mesh_type': mesh_type, 
        'hemi': hemi, 
        'prop_to_keep': 'jm', # [0.99,1.0],  #
    }

# Run configuration
ic_is_0 = False # NOTE: CHANGED
use_noise = True
save_derivatives = False

# Noise configuration
noise_args = {}
noise_type = "pink_noise"  # Options: "sinusoidal", "load_from_file", etc.
if noise_type == "pink_noise":
    noise_args.update({
        'v': 1,
        'alpha': 1,
        'frequency_factor': 1,
        'mr_ms': (1,1),
        'coords_div': 1,
    })
    if shape == 'real_mesh':
        noise_args['mr_ms'] = (100, 100)
        noise_args['v'] = 100
        noise_args['coords_div'] = 1

_extra_str = f"_{noise_type}_ic=Q" 
if 'prop_to_keep' in config_kwargs:
    if isinstance(config_kwargs['prop_to_keep'], str):
        _extra_str += '_subset_' + config_kwargs['prop_to_keep']

# Interpolation parameters (if noise_type == "load_from_file")
XN, TN = None, None
if 'radius' in config_kwargs:
    XN, TN = find_XN_TN(XN=None, TN=None, T=T, dt=dt, size=size, r=sphere_radius) # e.g. TN=N*T -> N points for every second

# Bands paramteres (noise_type = "bands"):
bands_names = ["delta", "theta", "alpha", "beta", "gamma", "gamma high", "gamma very high"]
