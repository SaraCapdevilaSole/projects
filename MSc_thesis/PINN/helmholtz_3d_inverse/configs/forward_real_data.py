import ml_collections

import jax.numpy as jnp
import os

def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "Helmholtz_forward_real_data"
    wandb.name = "rs_and_Q0_prediction_1s"
    wandb.tag = None

    # Loss mode
    config.surface_loss = False

    # Nondimensionalization
    config.nondim = True

    config.data = config_data = ml_collections.ConfigDict()
    config_data.dt = 1e-3  
    config_data.tr = 1e-3
    config_data.T = 2 
    config_data.t_avoid = 1
    config_data.parcellations_to_use = -1 
    config_data.permute_data = False 
    config_data.noise = -0.35371
    config_data.radius = 100

    config_data.gamma = 116
    config_data.r = 2.55146 
    mult = (100,100)
    v = 100 
    normalise = False
    params_sigmoid = {
        'rs': {'xl': 0, 'xh': 100}, 
        'gamma_s': {'xl': 0, 'xh': 250}, 
        'ms': {'xl': 10, 'xh': 1000}, 
    }
    
    config_data.params_sigmoid = params_sigmoid
    config_data.spatial_equation_args = {
        'freq_denom': 1,
        'mult': mult,
        'norm': normalise,
        'v': v,
        'alpha': 1,
    }
    
    name = 'zt' 
    data_path = './real_data'
    real_data_path = f'{data_path}/fixation_PAC/data'
    file_path = os.path.join(real_data_path, name, f'{name}_base.mat')
    config_data.data_path = file_path
    n = 3
    config_data.res_data_path = f'{data_path}/mesh_{n}k_patches/{name}_{n}.npy'

    config_data.sigma_gauss = None 
    config_data.model_terms = {
        'real_data': 'real_data' in config_data.data_path,
        'terms': {
            'Qs': False,
            'alpha': False, 
            'mr': False, 
            'ms': False, 
            'v': False, 
            'freq_denom': False, 
            'noise': False, 
            'rs': False,
            'gamma': False,
            'grad': False,
            'tgrad': False,
        }
    }

    config_data.save_eval_data = True
    config_data.save_eval_dir = f'/vol/bitbucket/sc3719/firedrake_simulation/data/output_pinn_inflated_real_data'

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    arch.num_layers = 3
    arch.hidden_dim = 64
    arch.out_dim = 1 + sum(config_data.model_terms['terms'].values())  # iPINN parameters to solve
    arch.activation = "tanh" 
    arch.periodicity = None
    arch.fourier_emb = ml_collections.ConfigDict(
        {"embed_scale": 15.0, "embed_dim": 256} 
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
    optim.learning_rate = 1e-4
    optim.decay_rate = 0.9  #
    optim.decay_steps = 2000  
    optim.grad_accum_steps = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 150000 
    training.num_time_windows = 1 
    training.res_batch_size = 1024
    training.ic_batch_size = 1024
    training.data_batch_size = 1024

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"

    if config.surface_loss:
        init_dict = {"ics": 1.0, "res": 10.0, "data": 1.0, "H": 1.0, "n": 1.0}
    else:
        init_dict = {"ics": 1.0, "res": 1.0, "data": 10.0}
    
    weighting.init_weights = ml_collections.ConfigDict(init_dict)
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000  
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
    logging.log_preds = True
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
