import os
import jax.numpy as jnp
import numpy as np 
from jaxpi.utils import restore_checkpoint
import ml_collections

import models
from utils import get_dataset, create_dir
from plotting import plotting, animate

def evaluate(config: ml_collections.ConfigDict, workdir: str):

    workdir = create_dir(config.wandb.project, workdir)
    
    # Load dataset
    config.data.permute_data = False # change to no permutation
    t_star, u_ref, coords, Qs = get_dataset(config_dataset=config.data) 

    gamma = config.data.gamma
    r = config.data.r

    # Nondimensionalization
    if config.nondim == True:
        L_star = jnp.max(jnp.abs(coords)) # scale coords to [-1,1]
        coords = coords / L_star
        r = r / L_star
    else:
        L_star = 1

    u_pred_list = []

    idx_window = int(len(t_star) / config.training.num_time_windows)

    for idx in range(config.training.num_time_windows):
        s_idx = idx_window*idx
        f_idx = len(t_star) if idx+1 == config.training.num_time_windows else idx_window*(idx+1)

        t_star_window = t_star[s_idx:f_idx]
        u_ref_window = u_ref[s_idx:f_idx, :]

        u0 = None 
        # u1 = None

        # Initialize model
        model = models.NFT3D(
            config,
            u_ref_window, 
            t_star_window,
            u0,
            # u1,
            coords,
            gamma,
            r,
            L_star
        )
        
        # Restore checkpoint
        ckpt_path = os.path.join(workdir, config.wandb.name, "ckpt", "time_window_{}".format(idx + 1))
        if os.path.exists(ckpt_path):
            model.state = restore_checkpoint(model.state, ckpt_path)
            params = model.state.params

            # Predict
            u_pred = model.u_pred_fn(params, t_star_window, coords[:, 0], coords[:, 1], coords[:, 2])
            u_pred_list.append(u_pred)
            del params
        else:
            print(f"No checkpoint found at {ckpt_path}")
        del model

    u_pred = jnp.concatenate(u_pred_list, axis=0)

    if config.data.save_eval_data:
        os.makedirs(config.data.save_eval_dir, exist_ok=True)
        data_eval = {
            'u_pred': np.array(u_pred),
            't_star': np.array(t_star),
            'coords': np.array(coords)
        }
        jnp.save(os.path.join(config.data.save_eval_dir, config.wandb.name), data_eval)

    # Dimensionalize
    if config.nondim == True:   
        # u_ref = u_ref * U_star 
        # u_pred = u_pred * U_star 
        coords = coords * L_star
        r = r * L_star
        # t_star = t_star * T_star
    
    if u_ref.shape[0] != u_pred.shape[0]:
        u_ref = u_ref[:u_pred.shape[0], :]
        t_star = t_star[:u_pred.shape[0]]
    
    # Compute L2 error
    u_error = jnp.sqrt(jnp.mean((u_pred - u_ref) ** 2))
    print("absolute error of u: {:.4e}".format(u_error))
    print("average value of u: ", jnp.mean(u_ref))
    
    # Plot and save
    save_dir = create_dir(wandb_dir=workdir, wandb_name=os.path.join("figures", config.wandb.name))

    u_error = jnp.abs(u_pred - u_ref)

    plot_first = True
    animation = False
    test = True # if we are using data points unseen during training

    if plot_first:
        plotting.plot_fig_tgt(t_star, u_pred, u_ref, u_error, save_dir, p=100, test=test)
        plotting.plot_fig_separate(t_star, u_pred, u_ref, u_error, save_dir, p=10, test=test)
        # plotting.plot_time_space_cross_section(t_star, coords, u_ref, u_pred, u_error, save_dir, test=test)
        # plotting.plot_3d_parcels(config.data.tr, coords, u_ref, u_pred, u_error, save_dir, test=test)
        plotting.plot_3d_snapshot(
            mesh_file=None, #config.data.msh_path, 
            mesh_coords=coords,
            firing_rate_data=u_ref, 
            firing_rate_pred=u_pred,
            time_data=t_star,
            path=save_dir, 
            cd=config.data,
            name="3d_test" if test else "3d"
        )

    if animation:
        fname = "firing_rate_visualisation"
        animate.make_3d_animation(
            mesh_file=None, #config.data.msh_path, 
            mesh_coords=coords,
            firing_rate_data=u_ref, 
            firing_rate_pred=u_pred,
            time_data=t_star,
            path=save_dir, 
            cd=config.data,
            name=fname + "_test" if test else fname
        )
