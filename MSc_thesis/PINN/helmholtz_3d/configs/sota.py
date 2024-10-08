import ml_collections

import jax.numpy as jnp

def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "Helmholtz_3d"
    wandb.name = 'helm_jm_slab_sota_data'
    wandb.tag = None

    # Data-related settings
    config.data = config_data = ml_collections.ConfigDict()
    config_data.dt = 1e-6  # Temporal resolution of saved data
    config_data.tr = 1e-5  # Temporal resolution for training
    config_data.T = 1e-2   # Total duration in seconds
    config_data.t_avoid = 0
    config_data.parcellations_to_use = -1
    config_data.parcellations_avoid = 0  # For test: shift by one and evaluate on unseen data
    config_data.use_every_voxel = 1  # Downsample by using every x voxel
    config_data.noise = 0.3  # Noise amplitude
    config_data.radius = 100

    # Nondimensionalization
    config.nondim = True
    config_data.permute_data = False 

    # Loss mode
    config.surface_loss = False

    # Spatial Equation Args (based on radius)
    if config_data.radius == 100:
        config_data.gamma = 116 
        config_data.r = 30 
        mult = (100,100) 
        v = 100
        normalise = False
    else:
        config_data.gamma = 10 
        config_data.r = 0.001 
        mult = (1,1)
        v = 1
        normalise = False

    config_data.spatial_equation_args = {
        'freq_denom': 1,
        'mult': mult,
        'norm': normalise,
        'v': v
    }

    config_data.save_eval_data = True
    config_data.save_eval_dir = f'/vol/bitbucket/sc3719/firedrake_simulation/data/output_pinn_inflated'
    config_data.data_path = '/vol/bitbucket/sc3719/firedrake_simulation/data/formatted_data_inflated/PINN-NFT_data_size=164_dt=1e-06_sn=1e-06_T=0.01_XN=None_TN=None_Q=0.3_pink_noise_ic=Q_subset_jm.npy'
    config_data.ncurv_path = None  # Optional path for curvature data

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    arch.num_layers = 5
    arch.hidden_dim = 256
    arch.out_dim = 1
    arch.activation = "tanh" 
    arch.periodicity = None
    arch.fourier_emb = ml_collections.ConfigDict(
        {"embed_scale": 20.0, "embed_dim": 128} 
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
    optim.decay_rate = 0.9 
    optim.decay_steps = 2000 
    optim.grad_accum_steps = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 200000 
    training.num_time_windows = 1  # Train a separate model for each time window
    training.batch_size_per_device = 1024

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    if config.surface_loss:
        init_dict = {"ics": 1.0, "data": 1.0, "res": 10.0, "H": 1.0, "n": 1.0}
    else:
        init_dict = {"ics": 1.0, "data": 10.0, "res": 100.0} 
    weighting.init_weights = ml_collections.ConfigDict(init_dict)
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000
    weighting.use_causal = False
    weighting.causal_tol = 10.0
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
    logging.log_correlation = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000 
    saving.num_keep_ckpts = 10

    # Input shape for initializing Flax models
    config.input_dim = 4

    # Integer for PRNG random seed.
    config.seed = 42

    return config
