# DETERMINISTIC
import os

# os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_reductions --xla_gpu_autotune_level=0"
# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5' 

from absl import app
from absl import flags
from absl import logging

import jax
from ml_collections import config_flags

import train
import eval

FLAGS = flags.FLAGS

_WORK_DIR = "/vol/bitbucket/sc3719/JAXPI"
flags.DEFINE_string("workdir", _WORK_DIR, "Directory to store model data.")

config_flags.DEFINE_config_file(
    "config",
    "./configs/default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    if FLAGS.config.mode == "train":
        train.train_and_evaluate(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "eval":
        eval.evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
