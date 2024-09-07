import ml_collections

import jax.numpy as jnp

def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "Helmholtz_2d_IC"
    wandb.name = 'hemlholtz_2d_pink_noise_corrected'#"hemlholtz_2d_3" # fix' #
    wandb.tag = None

    # Loss mode
    config.surface_loss = False

    # Nondimensionalization
    config.nondim = False

    config.data = config_data = ml_collections.ConfigDict()
    # loss mode
    config_data.sobolev_loss = False
    # dt = 1e-4 # integration step FEM
    sampling_dt = 1e-3 # temporal resolution of saved data
    config_data.dt = sampling_dt 
    config_data.tr = 1e-2 # from dt -> tr 
    config_data.T = 1 # use T seconds
    config_data.t_avoid = 0
    config_data.parcellations_to_use = -1 #5125 #645 # 838
    config_data.use_every_voxel = 10 # use every x voxel, e.g. with 1k mesh, use every 10 creates downsampled 100 voxel mesh
    config_data.permute_data = False #True # permute data order during training
    # config_data.msh_path = '/vol/bitbucket/sc3719/firedrake_simulation/data/input_data/sphere_hemis/lh_sphere_lc=1mm_r=1cm.msh'
    # type_data = 'lc=1mm_r=1cm'
    config_data.noise = 0.3 #0.001
    
    config_data.data_path = '/vol/bitbucket/sc3719/firedrake_simulation/data/formatted_data_square/PINN-NFT_data_size=100_dt=0.001_sn=0.001_T=1_XN=None_TN=None_Q=0.3_pink_noise.npy'
    #'/vol/bitbucket/sc3719/firedrake_simulation/data/formatted_data_square/PINN-NFT_data_size=10_dt=0.001_sn=0.001_T=5_XN=None_TN=None_Q=0.3_NOISE_Q=all_bands_spatial.npy'
    # config_data.curv_path = f'data/curv_normals/{type_data}/curv.txt'
    # config_data.norm_path = f'data/curv_normals/{type_data}/normals.txt'
    # config_data.msh_path = f'data/curv_normals/{type_data}/sphere_{type_data}.msh'

    # IC path:
    # '/vol/bitbucket/sc3719/firedrake_simulation/data/formatted_data_square/PINN-NFT_data_size=50_dt=0.001_sn=0.001_T=0.5_XN=None_TN=None_Q=0.3_no_noise_ic=Q.npy'
    # Q = 0
    # Q path (sinusoidal):
    # Q = self.noise * jnp.cos(jnp.pi * t) * jnp.sin(jnp.pi * x / 200) * jnp.sin(jnp.pi * y / 200)
    # /vol/bitbucket/sc3719/firedrake_simulation/data/formatted_data_square/PINN-NFT_data_size=30_dt=0.001_sn=0.001_T=0.5_XN=None_TN=None_Q=0.3_NOISE_Q=SINE.npy
    # Q (alpha, beta, theta waves):
    # /vol/bitbucket/sc3719/firedrake_simulation/data/formatted_data_square/PINN-NFT_data_size=10_dt=0.001_sn=0.001_T=5_XN=None_TN=None_Q=0.3_NOISE_Q=bands.npy
    # Q spatial with gamma waves
    # /vol/bitbucket/sc3719/firedrake_simulation/data/formatted_data_square/PINN-NFT_data_size=10_dt=0.001_sn=0.001_T=5_XN=None_TN=None_Q=0.3_NOISE_Q=all_bands_spatial.npy
    # Q is pink noise
    # /vol/bitbucket/sc3719/firedrake_simulation/data/formatted_data_square/PINN-NFT_data_size=100_dt=0.001_sn=0.001_T=1_XN=None_TN=None_Q=0.3_pink_noise.npy

    # derivs:
    # # config_data.data_path = '/vol/bitbucket/sc3719/firedrake_simulation/data/formatted_data_square/PINN-NFT_data_size=30_dt=0.001_sn=0.001_T=0.1_XN=None_TN=None_Q=0.001_NOISE_Q=SINE_derivs.npy'

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    arch.num_layers = 3
    arch.hidden_dim = 128
    arch.out_dim = 1
    arch.activation = "gelu" 
    arch.periodicity = None
    # arch.periodicity = ml_collections.ConfigDict(
    #     {"period": (jnp.pi,), "axis": (1,), "trainable": (False,)}
    # )
    arch.fourier_emb = ml_collections.ConfigDict(
        {"embed_scale": 10.0, "embed_dim": 256} 
    )
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 0.5, "stddev": 0.1}
    ) 

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9 # how much its decayed
    optim.decay_steps = 2000 # number of steps after which lr decays
    optim.grad_accum_steps = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 200000 
    training.batch_size_per_device = 1024 #2048 #4096 # 8192

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"

    if config.surface_loss:
        # init_dict = {"nabla": 1.0}
        init_dict = {"ics": 1.0, "data": 1.0, "res": 10.0, "H": 1.0, "n": 1.0}
    elif config_data.sobolev_loss:
        init_dict = {"ics": 1.0, "data": 1.0, "res": 10.0, "u_t": 1.0, "u_tt": 1.0}
    else:
        init_dict = {"ics": 1.0, "data": 1.0, "res": 10.0} 
    weighting.init_weights = ml_collections.ConfigDict(init_dict)
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000  # 100 for grad norm and 1000 for ntk

    weighting.use_causal = False  
    weighting.causal_tol = 1.0
    weighting.num_chunks = 32 

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_grads = False
    logging.log_ntk = False
    logging.log_preds = False
    logging.log_correlation = True

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000 
    saving.num_keep_ckpts = 10

    # Input shape for initializing Flax models
    config.input_dim = 3

    # Integer for PRNG random seed.
    config.seed = 42

    return config
