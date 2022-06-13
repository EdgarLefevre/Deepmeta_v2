# -*- coding: utf-8 -*-
import argparse
import os
import random
import re
from itertools import filterfalse as ifilterfalse
from typing import Any, List, Tuple

import numpy as np
import skimage.io as io
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.init as init
from matplotlib import pyplot as plt
from torch.autograd import Variable

import src.models.unet as unet
import src.models.unet_parts as up
import src.utils.pprint as pprint


def list_files_path(path: str) -> list:
    """
    List files from a path.

    :param path: Folder path
    :type path: str
    :return: A list containing all files in the folder
    :rtype: List
    """
    return sorted_alphanumeric(
        [path + f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    )


def shuffle_lists(lista: List, listb: List, seed: int = 42) -> Tuple[List, List]:
    """
    Shuffle two list with the same seed.

    :param lista: List of elements
    :type lista: List
    :param listb: List of elements
    :type listb: List
    :param seed: Seed number
    :type seed: int
    :return: lista and listb shuffled
    :rtype: (List, List)
    """
    random.seed(seed)
    random.shuffle(lista)
    random.seed(seed)
    random.shuffle(listb)
    return lista, listb


def sorted_alphanumeric(data: list[str]) -> list:
    """
    Sort function.

    :param data: str list
    :type data: List
    :return: Sorted list
    :rtype: List
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()  # noqa
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]  # noqa
    return sorted(data, key=alphanum_key)


def get_args() -> argparse.Namespace:
    """
    Argument parser.

    :return: Object containing all the parameters needed to train a model
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", "-e", type=int, default=10, help="number of epochs of training"
    )
    parser.add_argument(
        "--batch_size", "-bs", type=int, default=64, help="size of the batches"
    )
    parser.add_argument(
        "--size", "-s", type=int, default=128, help="Size of the image, one number"
    )
    parser.add_argument("--drop", "-d", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--filters",
        "-f",
        type=int,
        default=16,
        help="Number of filters in first conv block",
    )
    parser.add_argument("-w1", type=float, default=1.0, help="Weight for bg")
    parser.add_argument("-w2", type=float, default=5.0, help="Weight for lungs")
    parser.add_argument("-w3", type=float, default=15.0, help="Weight for metastases")
    parser.add_argument("-lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "-nworkers", type=int, default=4, help="Number of workers in dataloader."
    )
    parser.add_argument(
        "-classes", type=int, default=3, help="Number of classes to predict."
    )
    parser.add_argument(
        "--save", dest="save", action="store_true", help="If flag, save predictions."
    )
    parser.set_defaults(save=False)
    parser.add_argument(
        "--postprocess",
        dest="postprocess",
        action="store_true",
        help="If flag, apply postprocess on predictions.",
    )
    parser.set_defaults(postprocess=False)
    parser.add_argument(
        "--restart",
        type=int,
        default=5,
        help="Restart for cosine annealing. Default 5.\nSet this to 1 to disable.",
    )
    parser.add_argument(
        "--restart_mult",
        type=int,
        default=1,
        help="A factor increases Ti after a restart. Default 1.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        help="Model name. unet or unet_res",
        choices=[
            "unet",
            "unet_res",
            "unet3p",
            "res_unet3p",
            "urcnn",
            "att_unet",
            "att_unet3p",
            "unet++",
        ],
    )
    parser.add_argument(
        "--model_path", type=str, default="Unet_full", help="Weights file name."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="iou",
        help="Metric for stats. iou or f1",
        choices=["iou", "f1"],
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="ce",
        help="Loss function.",
        choices=["ce", "focal", "tanimoto", "lovasz", "fusion"],
    )
    parser.add_argument(
        "--conv",
        type=str,
        default="conv",
        help="Conv layer to use (conv or convsep).",
        choices=["conv", "convsep"],
    )
    parser.add_argument(
        "--contrast", dest="contrast", action="store_true", help="If flag, enhance contrast on image."
    )
    parser.set_defaults(contrast=False)
    parser.add_argument(
        "--img_path", type=str, default=None, help="Path to the tiff mouse stack."
    )
    parser.add_argument(
        "--label_path", type=str, default=None, help="Path to the labels folder, if exist. If not, no stats will be "
                                                     "processed. "
    )
    args = parser.parse_args()
    pprint.print_bold_red(args)
    return args


def display_img(
    img: np.ndarray, pred: np.ndarray, label: np.ndarray, cmap: str = "gray"
):
    """
    Display an image.

    :param img: Image to display
    :type img: numpy.ndarray
    :param pred: Prediction
    :type pred: numpy.ndarray
    :param label: Label
    :type label: numpy.ndarray
    :param cmap: Color map
    :type cmap: str
    """
    if img.shape[0] == 1:
        img = np.moveaxis(img, 0, -1)
        label = np.moveaxis(label, 0, -1)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(img, cmap=cmap)
    ax1.set_title("Input image")
    ax2.imshow(pred, cmap=cmap)
    ax2.set_title("Prediction")
    ax3.imshow(label, cmap=cmap)
    ax3.set_title("Label")
    plt.show()


def save_pred(img: np.ndarray, pred: np.ndarray, label: np.ndarray, path: str):
    """
    Save an image.

    :param img: Image to save
    :type img: numpy.ndarray
    :param pred: Prediction
    :type pred: numpy.ndarray
    :param label: Label
    :type label: numpy.ndarray
    :param path: Path to save
    :type path: str
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(img, cmap="gray")
    ax1.set_title("Input image")
    ax2.imshow(pred, cmap="gray")
    ax2.set_title("Prediction")
    if label is not None:
        ax3.imshow(label, cmap="gray")
        ax3.set_title("Label")
    plt.savefig(path)
    plt.close()


def weights_init_kaiming(m: Any) -> None:  # noqa
    # He initialization
    classname = m.__class__.__name__
    if classname.find("SeparableConv") != -1:
        init.kaiming_normal_(m.depthwise.weight.data, a=0, mode="fan_in")
        init.kaiming_normal_(m.pointwise.weight.data, a=0, mode="fan_in")
    elif classname.find("DoubleConv") != -1:
        for elt in m.double_conv:
            if elt.__class__.__name__.find("SeparableConv") != -1:
                init.kaiming_normal_(elt.depthwise.weight.data, a=0, mode="fan_in")
                init.kaiming_normal_(elt.pointwise.weight.data, a=0, mode="fan_in")
            elif elt.__class__.__name__.find("Conv") != -1:
                init.kaiming_normal_(elt.weight.data, a=0, mode="fan_in")
    elif classname.find("OutConv") != -1:
        try:
            init.kaiming_normal_(m.conv.weight.data, a=0, mode="fan_in")
        except Exception:
            init.kaiming_normal_(m.conv.depthwise.weight.data, a=0, mode="fan_in")
            init.kaiming_normal_(m.conv.pointwise.weight.data, a=0, mode="fan_in")
    elif classname.find("Conv") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def plot_learning_curves(
    hist_train: list[float], hist_val: list[float], path: str = "data/plots/"
) -> None:
    """
    Plot training curves.

    :param hist_train: List of epoch loss value for training
    :type hist_train: list
    :param hist_val: List of epoch loss value for validation
    :type hist_val: list
    :param path: Saving path
    :type path: str

    """
    os.system(f"mkdir -p {path}")
    epochs = range(len(hist_train))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, hist_train, label="Train loss")
    plt.plot(epochs, hist_val, label="Validation loss")
    plt.title("Loss - Training vs. Validation.")
    plt.ylabel("Loss (Cross Entropy)")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(f"{path}training_curves.png")


def get_model(args: argparse.Namespace) -> torch.nn.Module:
    """
    Load and init the model.

    :param args: Arguments
    :type args: argparse.Namespace
    :return: Model
    :rtype: torch.nn.Module
    """
    match args.model:
        case "unet":
            model = unet.Unet(
                filters=args.filters, classes=args.classes, conv_l=get_conv_l(args)
            )
        case "unet++":
            model = unet.Unetpp(
                filters=args.filters, classes=args.classes, conv_l=get_conv_l(args)
            )
        case "unet_res":
            model = unet.Unet_res(filters=args.filters, classes=args.classes)
        case "unet3p":
            model = unet.Unet3plus(
                filters=args.filters, classes=args.classes, conv_l=get_conv_l(args)
            )
        case "urcnn":
            model = unet.URCNN(
                filters=args.filters, classes=args.classes, conv_l=get_conv_l(args)
            )
        case "strided_unet":
            model = unet.StridedUnet(
                filters=args.filters, classes=args.classes, conv_l=get_conv_l(args)
            )
        case "att_unet":
            model = unet.Att_Unet(
                filters=args.filters, classes=args.classes, conv_l=get_conv_l(args)
            )
        case "att_unet3p":
            model = unet.Att_Unet3plus(
                filters=args.filters, classes=args.classes, conv_l=get_conv_l(args)
            )
        case "res_unet3p":
            model = unet.ResUnet3plus(
                filters=args.filters, classes=args.classes, conv_l=get_conv_l(args)
            )
        case _:
            raise ValueError(f"Unknown model {args.model}")
    model.apply(weights_init_kaiming)
    return model


def get_metric(args: argparse.Namespace) -> Any:
    """
    Get the metric.

    :param args: Arguments
    :type args: argparse.Namespace
    :return: Metric
    :rtype: sklearn.metrics.metrics
    """
    match args.metric:
        case "f1":
            metric = metrics.f1_score
        case "iou":
            metric = metrics.jaccard_score
        case _:
            raise ValueError(f"Unknown metric {args.metric}")
    return metric


def get_conv_l(args: argparse.Namespace) -> Any:
    """
    Get the convolutional layer.

    :param args: Arguments
    :type args: argparse.Namespace
    :return: Convolutional layer
    :rtype: torch.nn.Module
    """
    match args.conv:
        case "conv":
            conv_l = torch.nn.Conv2d
        case "convsep":
            conv_l = up.SeparableConv2d
        case _:
            raise ValueError(f"Unknown convolutional layer {args.conv}")
    return conv_l


def get_loss(args: argparse.Namespace) -> Any:
    """
    Get the loss function.

    :param args: Arguments
    :type args: argparse.Namespace
    :return: Convolutional layer
    :rtype: torch.nn.Module
    """
    match args.loss:
        case "ce":
            criterion = nn.CrossEntropyLoss(
                weight=torch.tensor([args.w1, args.w2, args.w3]).cuda(),
                label_smoothing=0.1,
            )
        case "focal":
            criterion = torch.hub.load(
                "adeelh/pytorch-multi-class-focal-loss",
                model="FocalLoss",
                alpha=torch.tensor([1.0, 5.0, 15.0]).cuda(),
                gamma=2,
                reduction="mean",
                force_reload=False,
            )
        case "tanimoto":
            criterion = TanimotoLoss().cuda()
        case "lovasz":
            criterion = LovaszLoss(per_image=True).cuda()
        case "fusion":
            criterion = FusionLoss(args).cuda()
        case _:
            raise ValueError(f"Unknown loss function {args.loss}")
    return criterion


class TanimotoLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6) -> None:
        super(TanimotoLoss, self).__init__()
        self.smooth = smooth

    @staticmethod
    def process_weights(y_true: torch.Tensor) -> torch.Tensor:
        zero = torch.tensor(0.0).cuda()
        wm = torch.where(
            y_true == 0, torch.tensor(1.0).cuda(), zero
        )  # torch.reciprocal((y_true == 0).sum() ** 2), zero)
        wm += torch.where(
            y_true == 1, torch.tensor(5.0).cuda(), zero
        )  # torch.reciprocal((y_true == 1).sum() ** 2), zero)
        wm += torch.where(
            y_true == 2, torch.tensor(50.0).cuda(), zero
        )  # torch.reciprocal((y_true == 2).sum() ** 2), zero)
        return wm * ((y_true == 2).sum() ** 2) * 5

    def tanimoto(
        self, preds: torch.Tensor, targets: torch.Tensor, wli: torch.Tensor
    ) -> torch.Tensor:
        pred_square = torch.square(preds)
        target_square = torch.square(targets)
        sum_square = torch.sum(torch.add(pred_square, target_square), dim=1)
        product = torch.mul(preds, targets)
        product_sum = torch.sum(product, dim=1)
        sum_prod_labels = torch.sum(torch.mul(wli, product_sum), dim=1)
        denominator = torch.sub(sum_square, product_sum)
        denominator_sum_labels = torch.sum(torch.mul(wli, denominator), dim=1)
        loss = torch.divide(
            sum_prod_labels + self.smooth, denominator_sum_labels + self.smooth
        )
        return 1.0 - torch.mean(loss)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        wli = self.process_weights(targets.type(torch.float))
        targets = torch.nn.functional.one_hot(targets, num_classes=3)
        targets = targets.permute(0, 3, 1, 2)
        loss1 = self.tanimoto(preds, targets, wli)
        preds = torch.sub(torch.ones_like(preds), preds)
        targets = torch.sub(torch.ones_like(targets), targets)
        loss2 = self.tanimoto(preds, targets, wli)
        return (loss1 + loss2) / 2


class LovaszLoss(nn.Module):
    def __init__(
        self, classes: str = "present", per_image: bool = False, ignore: Any = None
    ) -> None:
        super(LovaszLoss, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Multi-class Lovasz-Softmax loss
          probas: [B, C, H, W] Variable, class probabilities at each prediction
                    (between 0 and 1).
                  Interpreted as binary (sigmoid) output with outputs of size
                  [B, H, W].
          labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels,
          or a list of classes to average.
          per_image: compute the loss per image instead of per batch
          ignore: void class labels
        """
        if self.per_image:
            loss = mean(
                self.lovasz_softmax_flat(
                    *flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), self.ignore),
                    classes=self.classes,
                )
                for prob, lab in zip(y_pred, y_true)
            )
        else:
            loss = self.lovasz_softmax_flat(
                *flatten_probas(y_pred, y_true, self.ignore), classes=self.classes
            )
        return loss

    @staticmethod
    def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[:-1]
        return jaccard

    def lovasz_softmax_flat(
        self, probas: torch.Tensor, labels: torch.Tensor, classes: str = "present"
    ) -> torch.Tensor:
        """
        Multi-class Lovasz-Softmax loss
          probas: [P, C] Variable, class probabilities at each prediction
          (between 0 and 1)
          labels: [P] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels,
          or a list of classes to average.
        """
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.0
        C = probas.size(1)
        losses = []
        class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
        for c in class_to_sum:
            fg = (labels == c).float()  # foreground for class c
            if classes == "present" and fg.sum() == 0:
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError("Sigmoid output possible only with 1 class")
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (Variable(fg) - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(
                torch.dot(errors_sorted, Variable(self.lovasz_grad(fg_sorted)))
            )
        return mean(losses)


def flatten_probas(
    probas: torch.Tensor, labels: torch.Tensor, ignore: Any = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def isnan(x: torch.Tensor) -> torch.Tensor:
    return x != x


def mean(elt_list: Any, ignore_nan: bool = False, empty: int = 0) -> torch.Tensor:
    """
    nanmean compatible with generators.
    """
    elt_list = iter(elt_list)
    if ignore_nan:
        elt_list = ifilterfalse(isnan, elt_list)
    try:
        n = 1
        acc = next(elt_list)
    except StopIteration as e:
        if empty == "raise":
            raise ValueError("Empty mean") from e
        return empty
    for n, v in enumerate(elt_list, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class FusionLoss(nn.Module):
    def __init__(
        self, args: argparse.Namespace, alpha: float = 1, beta: float = 1, gamma: float = 1
    ) -> None:
        super(FusionLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(
                weight=torch.tensor([args.w1, args.w2, args.w3]).cuda(),
                label_smoothing=0.1,
            )
        self.focal = torch.hub.load(
                "adeelh/pytorch-multi-class-focal-loss",
                model="FocalLoss",
                alpha=torch.tensor([1.0, 5.0, 15.0]).cuda(),
                gamma=2,
                reduction="mean",
                force_reload=False,
            )
        self.lovasz = LovaszLoss(per_image=True)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.ce(y_pred, y_true) + \
               self.beta * self.lovasz(y_pred, y_true) + \
               self.gamma * self.focal(y_pred, y_true)

if __name__ == "__main__":
    tanimoto = TanimotoLoss()
    label = torch.Tensor(
        io.imread(
            "/home/edgar/Documents/Datasets/deepmeta/Data/3classes_metas/Labels/90_2115.tif"  # noqa
        ).reshape(1, 128, 128)
    )
    wm = tanimoto.process_weights(label)
    io.imshow(wm.numpy().reshape(128, 128))
    plt.show()
