import os
import jax.numpy as jnp
from jaxpi.utils import restore_checkpoint
import ml_collections

import models
from utils import get_dataset, create_dir
from plotting import plotting, animate

def evaluate(config: ml_collections.ConfigDict, workdir: str):

    workdir = create_dir(config.wandb.project, workdir)
    
    # Load dataset
    config.data.permute_data = False # change to no permutation
    t_star, u_ref, coords, Qs, u_t, u_tt = get_dataset(config_dataset=config.data) 

    # to test the inside is not being generated:
    # coords = random.uniform(jax.random.PRNGKey(0), minval=jnp.min(coords), maxval=jnp.max(coords), shape=coords.shape)
    gamma = 10
    r = 0.001 

    # Nondimensionalization
    if config.nondim == True:    
        raise NotImplementedError    
        # U_star = noise
        # T_star = 1 / gamma
        # L_star = r

        # # Qs = Qs / U_star
        # u_ref = u_ref / U_star 
        # coords = coords / L_star
        t_star = t_star / T_star

    # Initialize model
    model = models.NFT3D(
        config,
        u_ref, 
        t_star,
        coords,
        gamma,
        r,
        Qs,
        u_t,
        u_tt
    )

    # Restore checkpoint
    ckpt_path = os.path.join(workdir, config.wandb.name, "ckpt")
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    u_pred = model.u_pred_fn(params, t_star, coords[:, 0], coords[:, 1])

    # Compute L2 error
    u_error = model.compute_l2_error(params, t_star, coords, u_ref)
    print("absolute error of u: {:.4e}".format(u_error))
    print("average value of u: ", jnp.mean(u_ref))
    
    # Plot and save
    save_dir = create_dir(wandb_dir=workdir, wandb_name=os.path.join("figures", config.wandb.name))

    # Dimensionalize
    if config.nondim == True:
        u_ref = u_ref * U_star 
        u_pred = u_pred * U_star 
        coords = coords * L_star
        t_star = t_star * T_star

    u_error = jnp.abs(u_pred - u_ref)

    plot_first = True
    animation = True
    test = True # if we are using data points unseen during training

    if plot_first:
        plotting.plot_fig_tgt(t_star, u_pred, u_ref, u_error, save_dir, p=100, test=test)
        plotting.plot_fig_separate(t_star, u_pred, u_ref, u_error, save_dir, p=10, test=test)
        plotting.plot_time_space_cross_section(t_star, coords, u_ref, u_pred, u_error, save_dir, test=test)
        plotting.plot_2d_parcels(config.data.tr, coords, u_ref, u_pred, u_error, t_star, save_dir, test=test)

    if animation:
        fname = "firing_rate_visualisation"
        animate.make_2d_animation(
            mesh_file=None, #config.data.msh_path, 
            mesh_coords=coords,
            firing_rate_data=u_ref, 
            firing_rate_pred=u_pred,
            time_data=t_star,
            path=save_dir, 
            cd=config.data,
            name=fname + "_test" if test else fname
        )
