import os
import re

import numpy as np
import skimage.io as io


def inverse_binary_mask(msk):
    """
    Invert bytes of mask.

    :param msk: Binary mask (128x128)
    :type msk: np.array
    :return: Inverted mask
    :rtype: np.array
    """
    return np.ones((128, 128)) - msk


def stats_pixelbased(y_true, y_pred):
    """
    Process IoU between pred and gt.

    :param y_true: Ground truth
    :type y_true: np.array
    :param y_pred: Prediction
    :type y_pred: np.array
    :return: IoU coefficient
    :rtype: float
    """
    y_pred = y_pred.reshape(128, 128)
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Shape of inputs need to match. Shape of prediction is: {y_pred.shape}.  Shape of y_true is: {y_true.shape}"
        )

    pred = y_pred
    truth = y_true
    if truth.sum() == 0:
        pred = inverse_binary_mask(pred)
        truth = inverse_binary_mask(truth)
    # Calculations for IOU
    intersection = np.logical_and(pred, truth)
    union = np.logical_or(pred, truth)
    # precision = intersection.sum() / pred.sum()
    # recall = intersection.sum() / truth.sum()
    # Fmeasure = (2 * precision * recall) / (precision + recall)
    return -2 if union.sum() == 0 else intersection.sum() / union.sum()


def ratio(mask1, mask2):
    mask1 = np.array(mask1)
    mask2 = np.array(mask2)
    den = np.maximum(mask1.sum(), mask2.sum())
    num = np.minimum(mask1.sum(), mask2.sum())
    return 1.0 - (num / den) if den != num else 0.0


def diff(mask1, mask2):
    mask1 = np.array(mask1)
    mask2 = np.array(mask2)
    _max = np.maximum(mask1.sum(), mask2.sum())
    _min = np.minimum(mask1.sum(), mask2.sum())
    return _max - _min


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()  # noqa
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]  # noqa
    return sorted(data, key=alphanum_key)


def list_files_path(path):
    return sorted_alphanumeric(
        [path + f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    )


def main(path1, path2):
    file_list1 = list_files_path(path1)
    file_list2 = list_files_path(path2)
    res_ratio = []
    res_iou = []
    res_diff = []
    for i, file1 in enumerate(file_list1):
        file2 = file_list2[i]
        im_em = io.imread(file1, plugin="tifffile") / 255
        im_ed = io.imread(file2, plugin="tifffile") / 255

        # print("SUM : {} em ; {} ed \nratio = {}".format(im_em.sum(), im_ed.sum(), ratio(im_em, im_ed) ))
        # print("SUM : {} em ; {} ed \ndiff = {}".format(im_em.sum(), im_ed.sum(), diff(im_em, im_ed) ))
        res_diff.append(diff(im_ed, im_em))
        iou = stats_pixelbased(im_ed, im_em)
        # print("iou = {}".format(iou))
        # print(100*'-'+'\n')
        res_ratio.append(ratio(im_em, im_ed))
        res_iou.append(iou)
    print(f"mean ratio {np.array(res_ratio).mean()}")
    print(f"mean iou {np.array(res_iou).mean()}")
    print(f"mean diff {np.array(res_diff).mean()}")


if __name__ == "__main__":
    BASE_PATH_TEST = "/home/edgar/Documents/Datasets/deepmeta/deepmeta_dataset/Test/"
    BASE_PATH_ANNOT = "/home/edgar/Documents/Datasets/deepmeta/Test_annotation_2/"

    mice = ["m2Pc_c1_10Corr_1", "56_2PLc_day106", "28_2Pc_day50", "8_m2PL_day25"]
    for mouse in mice:
        print(f"{mouse} lungs")
        main(BASE_PATH_TEST + mouse + "/lungs/", BASE_PATH_ANNOT + mouse + "/Lungs/")
        print("\n")
        print(f"{mouse} Metas")
        main(BASE_PATH_TEST + mouse + "/metas/", BASE_PATH_ANNOT + mouse + "/Metas/")
        print(100 * "-")
