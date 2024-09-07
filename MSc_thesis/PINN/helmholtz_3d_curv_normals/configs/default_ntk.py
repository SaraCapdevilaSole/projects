import ml_collections

def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "Helmholtz"
    wandb.name = "hemlholtz_sphere_ntk"
    wandb.tag = None

    # Loss mode
    config.surface_loss = True

    # Nondimensionalization
    config.nondim = False

    config.data = config_data = ml_collections.ConfigDict()
    # dt = 1e-4 # integration step FEM
    sampling_dt = 1e-3 # temporal resolution of saved data
    config_data.dt = sampling_dt 
    config_data.tr = config_data.dt # from dt -> tr 
    config_data.T = 0.1 #30 # use T seconds (total = 600 - 20)
    config_data.t_avoid = 0 #0.3#0
    config_data.parcellations_to_use = 838 #5125 #645 # 838
    config_data.permute_data = False #True # permute data order during training
    config_data.msh_path = '/vol/bitbucket/sc3719/firedrake_simulation/data/input_data/sphere_hemis/lh_sphere.msh'
    config_data.data_path = '/vol/bitbucket/sc3719/firedrake_simulation/data/formatted_data/sphere_sn0.001_0.0001_5_PINN-NFT_data.npy'
    type_data = 'sphere'
    config_data.curv_path = f'data/{type_data}/curv.txt'
    config_data.norm_path = f'data/{type_data}/normals.txt'

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "ModifiedMlp"
    arch.num_layers = 3
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
    optim.decay_rate = 0.9 # how much its decayed
    optim.decay_steps = 2000 # number of steps after which lr decays
    optim.grad_accum_steps = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 100000 
    training.batch_size_per_device = 1024 #2048 #4096 # 8192

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "ntk"

    if config.surface_loss:
        # init_dict = {"nabla": 1.0}
        init_dict = {"ics": 1.0, "data": 1.0, "res": 1.0, "H": 1.0, "n": 1.0}
    else:
        init_dict = {"ics": 1.0, "res": 1.0} #, "data": 1.0}) 
    weighting.init_weights = ml_collections.ConfigDict(init_dict)
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000  # 100 for grad norm and 1000 for ntk

    weighting.use_causal = True  
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
    config.input_dim = 4

    # Integer for PRNG random seed.
    config.seed = 42

    return config
