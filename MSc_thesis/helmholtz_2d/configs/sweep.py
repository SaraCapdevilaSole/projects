import ml_collections

def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # Mode setting
    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "Helmholtz_sweep"
    wandb.name = None
    wandb.tag = None

    # Loss mode
    config.surface_loss = False

    # Data Configuration
    config.data = config_data = ml_collections.ConfigDict()
    config_data.dt = 1e-4  # Temporal resolution of data
    config_data.tr = 1e-4  # Downsampling rate
    config_data.T = 1e-2  # Total time duration to use (in seconds)
    config_data.t_avoid = 0  # Time to avoid at the beginning
    config_data.parcellations_to_use = -1  # Use all voxels (-1 for full data)
    config_data.use_every_voxel = 1  # Downsample the voxel mesh (e.g. x = 10, uses 1 every 10 voxels)
    config_data.noise = 0.3  # Noise amplitude
    config_data.data_path = '/vol/bitbucket/sc3719/firedrake_simulation/data/helmholtz_2d_pinns/u_size=50_dt=1e-06_sn=0.0001_T=0.01/PINN-NFT_data_size=50_dt=1e-06.npy'

    # Data Normalization & Permutation
    config.nondim = False 
    config_data.permute_data = False  

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    arch.num_layers = 4
    arch.hidden_dim = 128
    arch.out_dim = 1
    arch.activation = "gelu" 
    arch.periodicity = None
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
    optim.decay_rate = 0.9 
    optim.decay_steps = 2000 
    optim.grad_accum_steps = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 100000 
    training.batch_size_per_device = 1024 

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"

    if config.surface_loss:
        init_dict = {"ics": 1.0, "data": 1.0, "res": 1.0, "H": 1.0, "n": 1.0}
    else:
        init_dict = {"ics": 1.0, "data": 1.0, "res": 1.0} 
    weighting.init_weights = ml_collections.ConfigDict(init_dict)
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000  

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
    saving.save_every_steps = None 
    saving.num_keep_ckpts = 10

    # Input shape for initializing Flax models
    config.input_dim = 3

    # Integer for PRNG random seed.
    config.seed = 42

    return config
