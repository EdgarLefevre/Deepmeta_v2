#!/usr/bin/python3.10
# -*- coding: utf-8 -*-

import os

import ray
from ray import tune
from ray.tune.integration.wandb import WandbLogger
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB

import src.train as t
import src.utils.utils as utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

num_samples = 100  # -1 -> infinite, need stop condition
experiment_name = "unet_multiclass_hp_search"
METRIC = "metric"  # this is the name of the attribute in tune reporter
MODE = "max"


def create_folders():
    """
    This function creates the needed folders for ray and WandB (if needed).
    """
    os.makedirs("ray_result", exist_ok=True)
    os.makedirs("wandb", exist_ok=True)


if __name__ == "__main__":
    create_folders()
    ray.init(num_cpus=20, num_gpus=2)

    config = vars(utils.get_args())

    # WANDB
    # adding wandb keys
    config["wandb"] = {
        "project": experiment_name,
        "api_key_file": "/scratch/elefevre/Projects/DeepMeta/.wandb_key",
    }

    config["lr"] = tune.loguniform(1e-4, 1e-1)
    config["batch_size"] = tune.qrandint(16, 128, 8)
    config["model_name"] = "unet"
    config["w2"] = tune.randint(1, 15)
    config["w3"] = tune.randint(1, 20)
    config["drop_r"] = tune.quniform(0.1, 0.3, 0.005)
    config["filters"] = tune.choice([8, 16, 32])

    scheduler = HyperBandForBOHB(
        time_attr="training_iteration",
        metric=METRIC,
        mode=MODE,
        reduction_factor=2,
    )

    search_alg = TuneBOHB(
        metric=METRIC,
        mode=MODE,
        max_concurrent=5,
    )

    analysis = tune.run(
        t.train,  # fonction de train
        loggers=DEFAULT_LOGGERS + (WandbLogger,),
        config=config,
        local_dir="ray_results",
        name=experiment_name,
        num_samples=num_samples,
        search_alg=search_alg,
        scheduler=scheduler,
        resources_per_trial={"cpu": 10, "gpu": 1},
    )
    print(
        "Best hyperparameters found were: ",
        analysis.get_best_config(
            metric=METRIC,
            mode=MODE,
        ),
    )
    ray.shutdown()
