import os

# os.environ["XLA_FLAGS"] = '--xla_gpu_autotune_level=0'
# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # For better reproducible!  ~35% slower !
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8' 

from absl import app
from absl import flags

from ml_collections import config_flags

import wandb
import train

FLAGS = flags.FLAGS

_WORK_DIR = "~/../../vol/bitbucket/sc3719/JAXPI"
flags.DEFINE_string("workdir", _WORK_DIR, "Directory to store model data.")

config_flags.DEFINE_config_file(
    "config",
    "./configs/sweep_real_data.py",
    "File path to the training hyperparameter configuration.",
)


def main(argv):
    config = FLAGS.config
    workdir = FLAGS.workdir

    sweep_config = {
        "method": "grid",
        "name": "sweep",
        "metric": {"goal": "minimize", "name": "data_loss"},
    }

    parameters_dict = {
        "num_layers": {"values": [3, 4, 5]},
        "hidden_dim": {"values": [64, 128, 256, 512]},
        "activation": {"values": ["tanh", "gelu"]},
        "arch_reparam": {
            "values": [
                None,
                {"type": "weight_fact", "mean": 0.5, "stddev": 0.1},
                {"type": "weight_fact", "mean": 1.0, "stddev": 0.1},
            ]
        },
        "fourier_emb": {"values": [
            {"embed_scale": 15.0, "embed_dim": 256},
            {"embed_scale": 10.0, "embed_dim": 256},
            {"embed_scale": 5.0, "embed_dim": 256},
            {"embed_scale": 1.0, "embed_dim": 256},
            
            {"embed_scale": 15.0, "embed_dim": 128},
            {"embed_scale": 10.0, "embed_dim": 128},
            {"embed_scale": 5.0, "embed_dim": 128},
            {"embed_scale": 1.0, "embed_dim": 128},
        ]},
        "decay_steps": {"values": [2000, 5000]},
        "learning_rate": {"values": [1e-4, 1e-3]},
        "use_causal": {"values": [True, False]},
        "causal_tol": {"values": [1.0, 10.0]},
        "num_chunks": {"values": [16, 32]},
        "init_weights": {"values": [
            {"ics": 1.0, "data": 1.0, "res": 100.0},
            {"ics": 1.0, "data": 1.0, "res": 1000.0},
            {"ics": 1.0, "data": 10.0, "res": 1.0},
            {"ics": 1.0, "data": 100.0, "res": 1.0},
            {"ics": 1.0, "data": 1000.0, "res": 1.0},
            {"ics": 1.0, "data": 10.0, "res": 10.0},
            {"ics": 1.0, "data": 10.0, "res": 100.0},
            {"ics": 1.0, "data": 100.0, "res": 10.0},
        ]},
        "model_terms": {
            "values": [
                {'real_data': True, 'terms': {'Qs': False, 'alpha': True, 'mr': False, 'ms': False, 'v': False, 'freq_denom': False, 'noise': True, 'rs': False, 'gamma': False, 'grad': False, 'tgrad': False}},
                {'real_data': True, 'terms': {'Qs': False, 'alpha': False, 'mr': True, 'ms': True, 'v': False, 'freq_denom': False, 'noise': True, 'rs': False, 'gamma': False, 'grad': False, 'tgrad': False}},
                {'real_data': True, 'terms': {'Qs': False, 'alpha': True, 'mr': False, 'ms': False, 'v': True, 'freq_denom': False, 'noise': True, 'rs': False, 'gamma': False, 'grad': False, 'tgrad': False}},
                {'real_data': True, 'terms': {'Qs': False, 'alpha': True, 'mr': False, 'ms': True, 'v': False, 'freq_denom': False, 'noise': True, 'rs': False, 'gamma': False, 'grad': False, 'tgrad': False}},
                {'real_data': True, 'terms': {'Qs': False, 'alpha': True, 'mr': True, 'ms': True, 'v': False, 'freq_denom': False, 'noise': True, 'rs': False, 'gamma': False, 'grad': False, 'tgrad': False}},
                # {'real_data': True, 'terms': {'Qs': False, 'alpha': True, 'mr': True, 'ms': True, 'v': True, 'freq_denom': False, 'noise': True, 'rs': False, 'gamma': False, 'grad': False, 'tgrad': False}},
                # {'real_data': True, 'terms': {'Qs': False, 'alpha': False, 'mr': False, 'ms': False, 'v': False, 'freq_denom': False, 'noise': True, 'rs': True, 'gamma': False, 'grad': False, 'tgrad': False}},
                # {'real_data': True, 'terms': {'Qs': False, 'alpha': False, 'mr': False, 'ms': False, 'v': False, 'freq_denom': False, 'noise': True, 'rs': False, 'gamma': True, 'grad': False, 'tgrad': False}},
                # {'real_data': True, 'terms': {'Qs': False, 'alpha': False, 'mr': False, 'ms': False, 'v': False, 'freq_denom': False, 'noise': True, 'rs': True, 'gamma': True, 'grad': False, 'tgrad': False}},
                # {'real_data': True, 'terms': {'Qs': False, 'alpha': False, 'mr': False, 'ms': False, 'v': False, 'freq_denom': False, 'noise': True, 'rs': False, 'gamma': False, 'grad': False, 'tgrad': True}},
                # {'real_data': True, 'terms': {'Qs': False, 'alpha': False, 'mr': False, 'ms': False, 'v': False, 'freq_denom': False, 'noise': True, 'rs': False, 'gamma': False, 'grad': True, 'tgrad': False}},
                # {'real_data': True, 'terms': {'Qs': False, 'alpha': False, 'mr': False, 'ms': False, 'v': False, 'freq_denom': False, 'noise': True, 'rs': False, 'gamma': False, 'grad': True, 'tgrad': True}},
            ]
        },
    }

    sweep_config["parameters"] = parameters_dict

    def train_sweep():
        config = FLAGS.config

        wandb.init(project=config.wandb.project, name=config.wandb.name)

        sweep_config = wandb.config

        # Update config with sweep parameters
        config.weighting.use_causal = sweep_config.use_causal
        config.weighting.init_weights = sweep_config.init_weights
        config.arch.hidden_dim = sweep_config.hidden_dim
        config.arch.num_layers = sweep_config.num_layers
        config.arch.activation = sweep_config.activation
        config.arch.reparam = sweep_config.arch_reparam

        config.optim.decay_steps = sweep_config.decay_steps
        config.optim.learning_rate = sweep_config.learning_rate

        # config.weighting.scheme = sweep_config.weighting_scheme
        config.weighting.causal_tol = sweep_config.causal_tol
        config.weighting.num_chunks = sweep_config.num_chunks

        config.data.model_terms = sweep_config.model_terms
        config.arch.out_dim = 1 + sum(sweep_config.model_terms['terms'].values())

        train.train_and_evaluate(config, workdir)

    sweep_id = wandb.sweep(
        sweep_config,
        project=config.wandb.project,
    )

    wandb.agent(sweep_id, function=train_sweep)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
