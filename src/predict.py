# -*- coding: utf-8 -*-
import argparse  # noqa
import os
from typing import List

import numpy as np
import skimage.io as io
import skimage.transform as transform
import torch
import torch.nn as nn

import src.utils.data as data
import src.utils.postprocessing as pp
import src.utils.pprint as pprint
import src.utils.utils as utils

if os.uname()[1] == "iss":
    BASE_PATH = "/home/edgar/Documents/Datasets/deepmeta/Data/Souris_Test/"
    BASE_PATH2 = "/home/edgar/Documents/Datasets/deepmeta/Test_annotation_2/"
else:
    BASE_PATH = "/home/elefevre/Datasets/deepmeta/3classesv2/Souris_Test/"
    BASE_PATH2 = "/home/elefevre/Datasets/deepmeta/Test_annotation_2/"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def load_model(config: argparse.Namespace, device: str = "cuda") -> nn.Module:
    """
    Loads the model from the given path

    :param path: Path to the model
    :type path: str
    :param config: Configuration object
    :type config: argparse.Namespace
    :param device: Device to load the model on.
    :type device: str
    :return: The model with weights loaded
    :rtype: nn.Module
    """
    model = utils.get_model(config)
    model.load_state_dict(
        torch.load(f"data/{config.model_path}.pth", map_location=device)
    )
    model.eval()
    return model


def get_predict_dataset(path_souris, contrast=True):
    """
    Creates an image array from a file path (tiff file).

    :param path_souris: Path to the mouse file.
    :type path_souris: str
    :param contrast: Flag to run contrast and reshape
    :type contrast: Bool
    :return: Images array containing the whole mouse
    :rtype: np.array
    """
    mouse = io.imread(path_souris, plugin="tifffile").astype(np.uint8)
    mouse = transform.resize(mouse, (len(mouse), 128, 128), anti_aliasing=True)
    mouse = np.array(mouse) / np.amax(mouse)
    if contrast:
        mouse = data.contrast_and_reshape(mouse)
    else:
        mouse = np.array(mouse).reshape(-1, 1, 128, 128)
    return mouse


def get_labels(path: str) -> List[np.array]:
    """
    Loads the labels from the given path

    :param path: Path to the labels
    :type path: str
    :return: List of labels
    :rtype: List[np.array]
    """
    file_list = utils.list_files_path(path)
    return [io.imread(file, plugin="tifffile").astype(np.uint8) for file in file_list]


def process_img(mouse: torch.Tensor, model: nn.Module) -> List:
    output_stack = []
    for slice in mouse:
        slice = slice.reshape(1, 1, 128, 128)
        slice = torch.from_numpy(slice).float().cuda()
        output = model(slice)
        output = output.max(1).indices
        output_stack.append(output.cpu().detach().numpy())
    return output_stack


def stats(args, output_stack, mouse_labels):
    if mouse_labels is not None:
        res = []
        for i, output in enumerate(output_stack):
            img_f = output.flatten()
            label_f = mouse_labels[i].flatten()
            res.append(
                utils.get_metric(args)(
                    label_f, img_f, average=None, labels=[0, 1, 2], zero_division=1
                )
            )
        res = np.array(res).mean(0)
        print(res)
        return res


if __name__ == "__main__":
    args = utils.get_args()
    model = load_model(args).cuda()
    pprint.print_gre(f"Model {args.model_path} loaded")
    test_names = [
        ("souris_8", True),
        ("souris_28", True),
        ("souris_56", True),
        ("m2Pc_c1_10Corr_1", False),
    ]
    stats_list = []
    for name, contrast in test_names:
        pprint.print_gre("Predicting on {}".format(name))
        mouse = get_predict_dataset(f"{BASE_PATH}/{name}.tif", contrast=contrast)
        mouse_labels = get_labels(f"{BASE_PATH}/{name}/3classes/")
        output_stack = process_img(mouse, model)
        if args.postprocess:
            output_stack = pp.postprocess(mouse, np.array(output_stack))
            mouse_labels = pp.postprocess(mouse, np.array(mouse_labels))
        stats_list.append(stats(args, output_stack, mouse_labels))
        if args.save:
            os.system(f"mkdir -p data/{name}")
            for i, (slice, output, label) in enumerate(
                zip(mouse, output_stack, mouse_labels)
            ):
                utils.save_pred(
                    slice.reshape(128, 128),
                    output.reshape(128, 128),
                    mouse_labels[i],
                    f"data/{name}/{i}.png",
                )
    pprint.print_bold_green("Total stats:")
    print(np.array(stats_list).mean(0))

    print("\n\n\n")


    stats_list = []
    for name, contrast in test_names:
        pprint.print_gre("Predicting on {}".format(name))
        mouse = get_predict_dataset(f"{BASE_PATH2}/{name}.tif", contrast=contrast)
        mouse_labels = get_labels(f"{BASE_PATH2}/{name}/3classes/")
        output_stack = process_img(mouse, model)
        if args.postprocess:
            output_stack = pp.postprocess(mouse, np.array(output_stack))
            mouse_labels = pp.postprocess(mouse, np.array(mouse_labels))
        stats_list.append(stats(args, output_stack, mouse_labels))
        # if args.save:
        #     os.system(f"mkdir -p data/{name}")
        #     for i, (slice, output, label) in enumerate(
        #         zip(mouse, output_stack, mouse_labels)
        #     ):
        #         utils.save_pred(
        #             slice.reshape(128, 128),
        #             output.reshape(128, 128),
        #             mouse_labels[i],
        #             f"data/{name}/{i}.png",
        #         )
    pprint.print_bold_green("Total stats:")
    print(np.array(stats_list).mean(0))

# todo: args pour prendre un path d'image / un folder  et le label si il existe pour faire la prediction
# todo: arg pour faire sur le jeu de test
