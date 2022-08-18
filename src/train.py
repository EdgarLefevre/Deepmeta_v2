# -*- coding: utf-8 -*-

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import progressbar  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.utils.data as tud  # type: ignore
from sklearn.metrics import f1_score  # type: ignore

import src.utils.data as data
import src.utils.pprint as pprint
import src.utils.utils as utils

# widget list for the progress bar
widgets = [
    " [",
    progressbar.Timer(),
    "] ",
    progressbar.Bar(),
    " (",
    progressbar.ETA(),
    ") ",
]

if os.uname()[1] == "iss":
    BASE_PATH = "/home/edgar/Documents/Datasets/deepmeta/Data/3classes_metas/"
else:
    BASE_PATH = "/home/elefevre/Datasets/deepmeta/3classesv2/3classesv2_full/"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

SAVE_PATH = "data/model_inv_freq.pth"
LOSS = np.inf
METRIC = np.array([-1.0, -1.0, -1.0])


def save_model(net: nn.Module, loss: float) -> None:
    """
    Save the model if the loss is lower than the previous one.

    :param net: The model to save
    :type net: nn.Module
    :param loss: Loss value of the model during the validation step
    :type loss: float
    """
    global LOSS
    if loss < LOSS:
        LOSS = loss
        torch.save(net.state_dict(), SAVE_PATH)
        pprint.print_bold_red("Model saved")


def save_model_on_metric(net: nn.Module, metric: "np.ndarray") -> None:
    """
    Save the model if the loss is lower than the previous one.

    :param net: The model to save
    :type net: nn.Module
    :param metric: Metric values of the model during the validation step
    :type metric: np.array
    """
    global METRIC
    if metric[1] >= METRIC[1] and metric[2] >= METRIC[2]:
        METRIC = metric
        torch.save(net.state_dict(), SAVE_PATH)
        pprint.print_bold_red("Model saved")


def _step(
    net: nn.Module,
    dataloader: Dict[str, tud.DataLoader],
    args: argparse.Namespace,
    step: str,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: str = "cuda",
) -> float:
    """
    Train or validate the network

    :param net: The network to train
    :type net: nn.Module
    :param dataloader: The dataloader for the training and validation sets
    :type dataloader: Dict[str, tud.DataLoader]
    :param args: The arguments from the command line
    :type args: argparse.Namespace
    :param step: The step to train or validate
    :type step: str
    :param optimizer: The optimizer to use
    :type optimizer: torch.optim.Optimizer
    :return: The loss of the step
    :rtype: float
    """

    criterion = utils.get_loss(args, device=device)
    running_loss = []
    net.train() if step == "Train" else net.eval()
    dataset = dataloader[step]
    with progressbar.ProgressBar(max_value=len(dataset), widgets=widgets) as bar:
        f1 = []
        for i, (inputs, labels) in enumerate(dataset):
            bar.update(i)
            inputs, labels = inputs.to(device), (labels.long()).to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = net(inputs)
                loss = criterion(outputs, labels.squeeze(1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss.append(loss.item())
            outputs = outputs.max(1).indices
            labels = labels.squeeze(1)
            f1.append(
                utils.get_metric(args)(
                    torch.flatten(labels).cpu().detach().numpy(),
                    torch.flatten(outputs).cpu().detach().numpy(),
                    average=None,
                    labels=[0, 1, 2],
                    zero_division=1,
                )
            )
    f1_metric = np.array(f1).mean(0)
    epoch_loss = np.array(running_loss).mean()
    pprint.print_mag(
        f"[{step}] loss: {epoch_loss:.3f} {args.metric} bg: {f1_metric[0]:.5f}  "
        f"{args.metric} lungs: {f1_metric[1]:.5f} {args.metric} metas: {f1_metric[2]:.5f}"  # noqa
    )
    if step == "Val":
        save_model(net, epoch_loss)
    return epoch_loss


def train(
    net: nn.Module,
    dataloader: Dict[str, tud.DataLoader],
    args: argparse.Namespace,
    device: str = "cuda",
) -> Tuple[List[float], List[float], nn.Module]:
    """
    Train the network

    Parameters
    ----------
    net : nn.Module
        The network to train
    dataloader : Dict[str, tud.DataLoader]
        The dataloader for the training and validation sets
    args : argparse.Namespace
        The arguments from the command line
    device : str
        Device string
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, args.restart, args.restart_mult
    )
    history_train, history_val = [], []
    print(100 * "-")
    pprint.print_bold_green("Start training...")
    for epoch in range(args.epochs):
        pprint.print_gre(f"\nEpoch {epoch + 1}/{args.epochs} :")
        for step in ["Train", "Val"]:
            epoch_loss = _step(
                net, dataloader, args, step, optimizer, scaler, device=device
            )
            if step == "Val":
                history_val.append(epoch_loss)
            else:
                history_train.append(epoch_loss)
        scheduler.step()
    pprint.print_bold_green("Finished Training")
    return history_train, history_val, net


def pred_and_display(net: nn.Module, test_loader: tud.DataLoader) -> None:
    """
    Predict and display the results

    :param net: The network to use
    :type net: nn.Module
    :param test_loader: The dataloader for the test set
    :type test_loader: tud.DataLoader
    """
    net.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader, 0):
            inputs, labels = inputs.cuda(), (labels.long()).cuda()
            outputs = net(inputs)
            outputs = outputs.max(1).indices
            utils.display_img(
                inputs[0].cpu().detach().numpy(),
                outputs[0].cpu().detach().numpy(),
                labels[0].cpu().detach().numpy(),
            )
            utils.display_img(
                inputs[2].cpu().detach().numpy(),
                outputs[2].cpu().detach().numpy(),
                labels[2].cpu().detach().numpy(),
            )
            utils.display_img(
                inputs[5].cpu().detach().numpy(),
                outputs[5].cpu().detach().numpy(),
                labels[5].cpu().detach().numpy(),
            )
            break


def evaluate(net: nn.Module, test_loader: torch.utils.data.DataLoader) -> None:
    """
    Evaluate the network

    :param net: The network to use
    :type net: nn.Module
    :param test_loader: The dataloader for the test set
    :type test_loader: torch.utils.data.DataLoader
    """
    with progressbar.ProgressBar(max_value=len(test_loader), widgets=widgets) as bar:
        f1 = [0, 0, 0]
        for i, (inputs, labels) in enumerate(test_loader):
            bar.update(i)
            inputs, labels = inputs.cuda(), (labels.long()).cuda()
            outputs = net(inputs)
            outputs = outputs.max(1).indices
            labels = labels.reshape(
                (labels.shape[0], labels.shape[-1], labels.shape[-1])
            )
            f1 += (
                f1_score(
                    torch.flatten(labels).cpu().detach().numpy(),
                    torch.flatten(outputs).cpu().detach().numpy(),
                    average=None,
                )
                / labels.shape[0]  # noqa
            )
    f1_metric = np.array(f1) / float(i + 1)
    pprint.print_mag(
        f"F1 bg: {f1_metric[0]:.3f} "
        f"F1 lungs: {f1_metric[1]:.3f} "
        f"F1 metas: {f1_metric[2]:.3f}"
    )


if __name__ == "__main__":
    config = utils.get_args()
    model = nn.DataParallel(utils.get_model(config)).cuda()
    dataloader = data.get_datasets(BASE_PATH + "Images/", BASE_PATH + "Labels/", config)
    history_train, history_val, _ = train(model, dataloader, config)
    utils.plot_learning_curves(history_train, history_val)
    pred_and_display(model, dataloader["Val"])
    # evaluate(model, dataset_val)
