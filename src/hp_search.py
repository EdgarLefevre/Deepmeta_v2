#!/usr/bin/python3.10
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import optuna
from optuna.trial import TrialState
import wandb

import src.predict as p
import src.utils.utils as utils
import src.utils.data as data

if os.uname()[1] == "iss":
    BASE_PATH = "/home/edgar/Documents/Datasets/deepmeta/Data/3classes_metas/"
    TEST_PATH = "/home/edgar/Documents/Datasets/deepmeta/Data/Souris_Test/"
else:
    BASE_PATH = "/home/elefevre/Datasets/deepmeta/3classesv2/3classesv2_full/"
    TEST_PATH = "/home/elefevre/Datasets/deepmeta/3classesv2/Test/"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

EXPERIMENT_NAME = "unet_multiclass_hp_search"
ENTITY = "elefevre"


def create_folders():
    """
    This function creates the needed folders for ray and WandB (if needed).
    """
    os.makedirs("ray_result", exist_ok=True)
    os.makedirs("wandb", exist_ok=True)


def objective(trial):
    args = utils.get_args()
    net = utils.get_model(args).cuda()
    dataloader = data.get_datasets(
        f'{BASE_PATH}Images/', f'{BASE_PATH}Labels/', args
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, args.restart, args.restart_mult
    )
    # init tracking experiment.
    # hyper-parameters, trial id are stored.
    config = dict(trial.params)
    config["trial.number"] = trial.number
    wandb.init(
        project="DeepMeta Multiclass",
        entity=ENTITY,  # NOTE: this entity depends on your wandb account.
        config=config,
        group=EXPERIMENT_NAME,
        reinit=True,
    )
    for epoch in range(args.epochs):
        print(f"Training epoch: {epoch+1}")
        criterion = utils.FusionLoss(args,
                                     alpha=trial.suggest_float("alpha", 0.1, 1., log=True),
                                     beta=trial.suggest_float("beta", 0.1, 1., log=True),
                                     gamma=trial.suggest_float("gamma", 0.1, 1., log=True))
        net.train()
        dataset = dataloader["Train"]
        for inputs, labels in dataset:
            inputs, labels = inputs.cuda(), (labels.long()).cuda()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = net(inputs)
                loss = criterion(outputs, labels.squeeze(1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        net.eval()
        print('Testing....')
        test_names = [
            ("souris_8", True),
            ("souris_28", True),
            ("souris_56", True),
            ("m2Pc_c1_10Corr_1", False),
        ]
        stats_list = []
        for name, contrast in test_names:
            mouse = p.get_predict_dataset(
                f"{TEST_PATH}/{name}.tif", contrast=contrast
            )
            mouse_labels = p.get_labels(f"{TEST_PATH}/{name}/3classes/")
            output_stack = p.process_img(mouse, net)
            stats_list.append(p.stats(args, output_stack, mouse_labels))
        stat_value = np.array(stats_list).mean(0)
        trial.report(stat_value[2], epoch)
        # report validation accuracy to wandb
        wandb.log(data={"Metastases accuracy": stat_value[2],
                        "Lung accuracy": stat_value[1]}, step=epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            wandb.run.summary["state"] = "pruned"
            wandb.finish(quiet=True)
            raise optuna.exceptions.TrialPruned()
    return stat_value[2]


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
