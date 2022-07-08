import argparse
import multiprocessing
import os
import time

import cv2
import numpy as np
import pandas as pd
import scipy
import scipy.ndimage.filters as filters
import skimage
import skimage.io as io

##################################################################################
########################## DATA PREPARATION ######################################
##################################################################################


def list_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def list_files_path(path):
    return [path + f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def list_paths(path):
    """
    List all the paths in a directory
    """
    paths = []
    for root, dirs, files in os.walk(path):
        paths.extend(os.path.join(root, file) for file in files)
    return paths


def get_number(path):
    filename = os.path.basename(path)
    return int(filename.split("_")[0])


def enhance_contrast(image):
    image = skimage.exposure.equalize_adapthist(image / 255, clip_limit=0.03)
    return (image * 255).astype(np.uint8)


def process_img(im_path, img_indexes, labels_path, save_path):
    img = io.imread(im_path, plugin="tifffile").astype(np.uint8)
    for i, index in enumerate(img_indexes):
        if f"{str(index)}.tif" in labels_path:
            print(index)
            if index > 11136:  # 11136 = separator of two batches
                io.imsave(save_path + str(index) + ".tif", img[i])
            else:
                io.imsave(save_path + str(index) + ".tif", enhance_contrast(img[i]))


def img_to_slices():
    PATHS = ["Raw_data/"]

    tiff_path = []
    for path in PATHS:
        tiff_path += list_files_path(path)

    LAB_PATH = "Lungs/Labels/"
    l = list_files(LAB_PATH)

    LAB_PATH_metas = "Metastases/Labels/"
    l_metas = list_files(LAB_PATH_metas)

    for img_path in tiff_path:
        file_nb = get_number(img_path)
        i = 128 * file_nb
        img_indexes = range(i, i + 128)
        process_img(img_path, img_indexes, l, "Lungs/Images/")
        process_img(img_path, img_indexes, l_metas, "Metastases/Images/")


##################################################################################
########################## DATA AUGMENTATION #####################################
##################################################################################


def rotate_img(path, img_name):
    img = cv2.imread(path + img_name, cv2.IMREAD_UNCHANGED)
    # get image height, width
    (h, w) = img.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)
    angles = [90, 180, 270]

    scale = 1.0

    # Perform the counter clockwise rotation holding at the center
    for angle in angles:
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(img, M, (h, w))
        cv2.imwrite(path + str(angle) + "_" + img_name, rotated)


def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32(
        [
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size,
        ]
    )
    pts2 = pts1 + random_state.uniform(
        -alpha_affine, alpha_affine, size=pts1.shape
    ).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(
        image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101
    )

    dx = filters.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = filters.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = (
        np.reshape(y + dy, (-1, 1)),
        np.reshape(x + dx, (-1, 1)),
        np.reshape(z, (-1, 1)),
    )

    return scipy.ndimage.interpolation.map_coordinates(
        image, indices, order=1, mode="reflect"
    ).reshape(shape)


def elastic_transform_wrapped(img, mask, path_img, path_mask):
    im = io.imread(path_img + img, plugin="tifffile")
    im_mask = io.imread(path_mask + mask, plugin="tifffile")
    im_merge = np.concatenate((im[..., None], im_mask[..., None]), axis=2)

    # im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 3, im_merge.shape[1] * 0.09, im_merge.shape[1] * 0.09)
    im_merge_t = elastic_transform(
        im_merge,
        im_merge.shape[1] * 2,
        im_merge.shape[1] * 0.08,
        im_merge.shape[1] * 0.08,
    )  # soft transform

    im_t = im_merge_t[..., 0]
    im_mask_t = im_merge_t[..., 1]
    io.imsave(path_img + "t_" + img, im_t)
    io.imsave(path_mask + "t_" + mask, im_mask_t)


def process_DA(path_img, path_label):
    list_img = list_files(path_img)
    list_mask = list_files(path_label)

    for i, img in enumerate(list_img):
        print(img)
        mask = list_mask[i]
        rotate_img(path_img, img)
        rotate_img(path_label, mask)

    list_img = list_files(path_img)
    list_mask = list_files(path_label)

    for i, img in enumerate(list_img):
        print("transform " + img)
        mask = list_mask[i]
        elastic_transform_wrapped(img, mask, path_img, path_label)


def main_DA():
    img_folders = ["Lungs/Images/", "Metastases/Images/"]
    mask_folders = ["Lungs/Labels/", "Metastases/Labels/"]
    for i, img_path in enumerate(img_folders):
        mask_path = mask_folders[i]
        process_DA(img_path, mask_path)


##################################################################################
########################## Multi Processing ######################################
##################################################################################


def multi_process_fun(file_list, dirpath, function, num_workers):
    list_size = len(file_list)
    worker_amount = int(list_size / num_workers)

    processes = []
    for worker_num in range(num_workers):
        if worker_num == range(num_workers)[-1]:
            process = multiprocessing.Process(
                target=function, args=(dirpath, file_list[worker_amount * worker_num :])
            )
        else:
            process = multiprocessing.Process(
                target=function,
                args=(
                    dirpath,
                    file_list[
                        worker_amount * worker_num : worker_amount * worker_num
                        + worker_amount
                    ],
                ),
            )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


def wrapped_rotate(path, file_list):
    for file in file_list:
        print(file)
        rotate_img(path, file)


def wrapped_elastic_transformation(path, file_list):
    """Wrapped elastic transform for multiprocessing.

    Args:
        path ([(str,str)]): Cople of path ; first path for images, second path for masks
        file_list ([type]): Zip of img and mask files
    """
    path_img, path_mask = path
    for (img, mask) in file_list:
        elastic_transform_wrapped(img, mask, path_img, path_mask)


def process_DA_multi(path_img, path_label, n_workers):
    list_img = list_files(path_img)
    list_mask = list_files(path_label)

    multi_process_fun(list_img, path_img, wrapped_rotate, n_workers)
    multi_process_fun(list_mask, path_label, wrapped_rotate, n_workers)

    list_img = list_files(path_img)
    list_mask = list_files(path_label)

    multi_process_fun(
        list(zip(list_img, list_mask)),
        (path_img, path_label),
        wrapped_elastic_transformation,
        n_workers,
    )


def main_DA_multi(n_worker):
    img_folders = ["3classes/Images/"]  # "Lungs/Images/", "Metastases/Images/"]
    mask_folders = ["3classes/Labels/"]  # "Lungs/Labels/", "Metastases/Labels/"]
    for i, img_path in enumerate(img_folders):
        mask_path = mask_folders[i]
        process_DA_multi(img_path, mask_path, n_worker)


def get_args():
    """
    Argument parser.

    :return: Object containing all the parameters needed to train a model
    :rtype: Dict
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="number of thread used to run the script.",
    )
    return parser.parse_args()


##################################################################################
########################## 3 classes masks #######################################
##################################################################################


def get_healthy_mice(csv_path):
    """
    Get the healthy mice from the csv file
    """
    df = pd.read_csv(csv_path)
    return df.loc[df["Saine/Metas"] == "s"]["Id"].values


def create_new_mask(mask_lungs, mask_metas):
    assert np.shape(mask_lungs) == np.shape(mask_metas)
    return np.where((mask_metas == 1), 2, mask_lungs)


def get_number_3classes(name):
    return np.trunc(int(name.split(".")[0]) / 128)


def multi_classes_train():
    PATH_LUNGS = "Lungs/Labels/"
    PATH_METAS = "Metastases/Labels/"
    new_path_imgs = "3classes/Images/"
    new_path = "3classes/Labels/"
    PATH_IMGS = "Lungs/Images/"
    os.system(f"mkdir -p {new_path}")
    os.system(f"mkdir -p {new_path_imgs}")

    healthy_mice_list = get_healthy_mice("mouse_ID.csv")
    paths_lungs = list_paths(PATH_LUNGS)
    for lung_path in paths_lungs:
        name = lung_path.split("/")[-1]
        lung_mask = io.imread(lung_path) / 255
        print(100 * "-")
        print(name)
        try:
            meta_mask = io.imread(PATH_METAS + name) / 255
            new_mask = create_new_mask(lung_mask, meta_mask)
            os.system(f"cp {PATH_IMGS}{name} {new_path_imgs}")
            io.imsave(new_path + name, new_mask)
        except Exception as e:
            print(e)
            print(f"{name} meta not found.")
            nb = get_number_3classes(name)
            if nb in healthy_mice_list:
                print(f"{name} is healthy")
                os.system(f"cp {PATH_IMGS}{name} {new_path_imgs}")
                io.imsave(new_path + name, lung_mask)


def multi_classes_test():
    BASE_PATH = "Test/"
    name_list = ["m2Pc_c1_10Corr_1", "8_m2PL_day25", "28_2Pc_day50", "56_2PLc_day106"]
    for name in name_list:
        print(name)
        os.system(f"mkdir -p {BASE_PATH+name}/3classes/")
        paths_metas = list_paths(f"{BASE_PATH+name}/metas/")
        assert len(paths_metas) != 0, f"{BASE_PATH+name}/metas/"
        for meta_path in paths_metas:
            f_name = meta_path.split("/")[-1]
            try:
                meta_mask = io.imread(meta_path) / 255
                assert np.max(meta_mask) <= 1
                lung_mask = io.imread(f"{BASE_PATH}{name}/lungs/{f_name}") / 255
                assert np.max(lung_mask) <= 1
                new_mask = create_new_mask(lung_mask, meta_mask)
                io.imsave(f"{BASE_PATH+name}/3classes/" + f_name, new_mask)
            except Exception as e:
                print(f_name)


if __name__ == "__main__":
    start = time.time()
    args = get_args()
    img_to_slices()
    multi_classes_train()
    multi_classes_test()
    main_DA_multi(args.n_workers)
    end = time.time()
    print(f"Dataset generation took: {end - start}sec.")
