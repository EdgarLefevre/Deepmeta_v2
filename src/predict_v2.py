import argparse  # noqa
import os
from typing import List

import numpy as np
import skimage.io as io  # type: ignore
import skimage.transform as transform  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore

import src.predict as predict
import src.utils.data as data
import src.utils.postprocessing as pp
import src.utils.pprint as pprint
import src.utils.utils as utils

if os.uname()[1] != "iss":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == "__main__":
    # Load model
    args = utils.get_args()
    model = predict.load_model(args)
    pprint.print_gre(f"Model {args.model_path} loaded")

    # Load images
    name = args.img_path.split("/")[-1].split(".")[0]
    pprint.print_gre("Predicting on {}".format(name))
    mouse = predict.get_predict_dataset(args.img_path, contrast=args.contrast)

    # Load labels
    if args.label_path is not None:
        mouse_labels = predict.get_labels(args.label_path)
    else:
        mouse_labels = None  # type: ignore

    # Process img stack
    output_stack = predict.process_img(mouse, model)

    if args.postprocess:
        output_stack = pp.postprocess(mouse, np.array(output_stack))  # type: ignore
        if mouse_labels is not None:
            mouse_labels = pp.postprocess(mouse, np.array(mouse_labels))  # type: ignore

    predict.stats(args, output_stack, mouse_labels)
    nb = predict.get_meta_nb(output_stack > 1.5)  # type: ignore
    print(f"Found: {nb} metastases.")

    # Save outputs
    if args.save:
        os.system(f"mkdir -p data/{name}")
        if mouse_labels is not None:
            for i, (slice, output, label) in enumerate(
                zip(mouse, output_stack, mouse_labels)
            ):
                utils.save_pred(
                    slice.reshape(128, 128),
                    output.reshape(128, 128),
                    mouse_labels[i],
                    f"data/{name}/{i}.png",
                )
        else:
            for i, (slice, output) in enumerate(zip(mouse, output_stack)):
                utils.save_pred(
                    slice.reshape(128, 128),
                    output.reshape(128, 128),
                    None,
                    f"data/{name}/{i}.png",
                )
