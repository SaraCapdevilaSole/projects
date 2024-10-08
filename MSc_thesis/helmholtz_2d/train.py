import time
import os

from absl import logging

import jax
import jax.numpy as jnp
from jax import vmap, jacrev, random
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map

import ml_collections

import wandb

import matplotlib.pyplot as plt

from jaxpi.samplers import TimeSpaceSampler, UniformSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models
from utils import get_dataset, create_dir


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    wandb_config = config.wandb
    workdir = create_dir(wandb_config.project, workdir)
    wandb.init(project=wandb_config.project, name=wandb_config.name, dir=workdir, config=config.to_dict()) # Set W&B directory for saved logs

    logger = Logger()

    # Get dataset
    t_star, u_ref, coords, Qs, u_t, u_tt = get_dataset(config_dataset=config.data) 
    gamma = 10
    r = 0.001 

    t0 = t_star[0]
    t1 = t_star[-1]

    # Define domain
    temporal_dom = jnp.array([t0, t1])

    if config.nondim == True:
        raise NotImplementedError
        
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

    # Initialize evaluator
    evaluator = models.NFTEvaluator(config, model)

    # Initialize residual sampler
    res_sampler = iter(TimeSpaceSampler(temporal_dom, coords, config.training.batch_size_per_device))

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
                log_dict = evaluator(state, batch, t_star, coords, u_ref)
                wandb.log(log_dict, step)

                end_time = time.time()
                # Report training metrics
                logger.log_iter(step, start_time, end_time, log_dict)
                start_time = time.time()

        # Save checkpoint
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                ckpt_path = os.path.join(workdir, config.wandb.name, "ckpt")
                save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)

    return model
