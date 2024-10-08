import time
import os

from absl import logging
from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap, jacrev, random, pmap
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map

import ml_collections

import wandb

import matplotlib.pyplot as plt

from jaxpi.samplers import TimeSpaceSampler, UniformSampler, SpaceSampler, BaseSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint, restore_checkpoint

import models
from utils import get_dataset, create_dir

class MeshSampler(BaseSampler):
    def __init__(self, curv, normals, t_star, spatial_coords, batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.curv = curv
        self.normals = normals
        self.t_star = t_star
        self.spatial_coords = spatial_coords

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        key1, key2 = random.split(key)

        temporal_idx = random.choice(
            key1, self.t_star.shape[0], shape=(self.batch_size,)
        )

        spatial_idx = random.choice(
            key2, self.spatial_coords.shape[0], shape=(self.batch_size,)
        )
        
        temporal_batch = self.t_star[temporal_idx].reshape(self.batch_size, 1)
        spatial_batch = self.spatial_coords[spatial_idx, :]
        curv_batch = self.curv[spatial_idx]
        normals_batch = self.normals[spatial_idx, :]

        ts_batch = jnp.concatenate([temporal_batch, spatial_batch], axis=1)
        batch = (ts_batch, curv_batch, normals_batch)
        return batch
        
def train_one_window(config, workdir, model, res_sampler, idx):
    # Initialize evaluator
    evaluator = models.NFTEvaluator(config, model)

    logger = Logger()

    step_offset = idx * config.training.max_steps

    print("Waiting for JIT...")
    start_time = time.time()
    for step in range(config.training.max_steps):
        batch = next(res_sampler)
        model.state = model.step(model.state, batch)

        # Update weights if necessary
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch) 
                wandb.log(log_dict, step + step_offset)

                end_time = time.time()
                # Report training metrics
                logger.log_iter(step, start_time, end_time, log_dict)
                start_time = time.time()

        # Save checkpoint
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                ckpt_path = os.path.join(workdir, config.wandb.name, "ckpt", "time_window_{}".format(idx + 1))
                save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)

    return model


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    wandb_config = config.wandb
    workdir = create_dir(wandb_config.project, workdir)
    wandb.init(project=wandb_config.project, name=wandb_config.name, dir=workdir, config=config.to_dict()) # change dir of saved logs

    t_star, u_ref, coords, Qs, curv, normals = get_dataset(config_dataset=config.data) 

    # Get dataset
    gamma = config.data.gamma
    r = config.data.r

    if config.nondim == True:
        L_star = jnp.max(jnp.abs(coords)) # scale coords to [-1,1]
        coords = coords / L_star
        r = r / L_star
    else:
        L_star = 1

    T_window = (config.data.T - config.data.t_avoid) / config.training.num_time_windows
    idx_window = int(len(t_star) / config.training.num_time_windows)

    u0 = u_ref[0, :]

    for idx in range(config.training.num_time_windows):
        logging.info("Training time window {}/{}; T={}s".format(idx + 1, config.training.num_time_windows, T_window))
        
        s_idx = idx_window*idx
        f_idx = len(t_star) if idx+1 == config.training.num_time_windows else idx_window*(idx+1)

        t_star_window = t_star[s_idx:f_idx]
        u_ref_window = u_ref[s_idx:f_idx, :]

        t0 = t_star_window[0]
        t1 = t_star_window[-1]

        # Define domain
        temporal_dom = jnp.array([t0, t1])

        # Initialize model
        model = models.NFT3D(
            config,
            u_ref_window, 
            t_star_window,
            u0,
            coords,
            gamma,
            r,
            L_star,
            curv,
            normals
        )

        # Initialize residual sampler
        if config.surface_loss:
            res_sampler = iter(MeshSampler(curv, normals, t_star, coords, config.training.batch_size_per_device))
        else:
            res_sampler = iter(TimeSpaceSampler(temporal_dom, coords, config.training.batch_size_per_device))

        # Train model for the current time window
        model = train_one_window(config, workdir, model, res_sampler, idx)

        # Update the initial condition for the next time window
        if config.training.num_time_windows > 1:
            state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], model.state))
            params = state.params
            u0 = vmap(model.u_net, (None, None, 0, 0, 0))(
                params, t1, coords[:, 0], coords[:, 1], coords[:, 2]
            )
            del model
    
