import os

import cc3d  # type: ignore


def write_in_csv(filename, mousename, day, vol_l, vol_m, vol_pm, mutation):
    """
    Create a csv file and fill it with
    :param filename:
    :type filename:
    :param mousename:
    :type mousename:
    :param day:
    :type day:
    :param vol_l:
    :type vol_l:
    :param vol_m:
    :type vol_m:
    :param vol_pm:
    :type vol_pm:
    :param mutation:
    :type mutation:
    :return:
    :rtype:
    """
    check_and_create_file(filename)
    with open(filename, "a") as f:
        f.write(f"{mousename},{day},{vol_l},{vol_m},{vol_pm},{mutation}" + "\n")


def write_meta_in_csv(filename, mousename, meta_id, vol, mutation):
    """
    Create a csv file and fill it in order to create graph number of meta
    in function of time.
    :param filename: csv filename
    :type filename: Str
    :param mousename: Name of the mouse file
    :type mousename: Str
    :param slice_nb: Id of the slice
    :type slice_nb: int
    :param meta_id:
    :type meta_id: Int
    :param vol: Meta volume
    :type vol: int
    :param mutation: Name of the mesured mutation
    :type mutation: Str
    """
    check_and_create_file_meta(filename)
    with open(filename, "a") as f:
        f.write(f"{mousename},{str(meta_id)},{str(vol)},{mutation}" + "\n")


def check_and_create_file_meta(path):
    if not os.path.isfile(path):
        with open(path, "a+") as f:
            f.write("name,meta_id,vol,mutation\n")


def check_and_create_file(path):
    if not os.path.isfile(path):
        with open(path, "a+") as f:
            f.write("name,day,vol_l,vol_m,vol_pm,mutation\n")


def nb_meta_volume_v2(stack, filename, mousename, mutation):
    labels_out, N = cc3d.connected_components(stack, return_N=True, connectivity=18)
    for id in range(1, N + 1):
        extracted_image = labels_out * (labels_out == id)
        write_meta_in_csv(
            filename,
            mousename,
            id,
            (extracted_image.sum() * 0.0047),
            mutation,
        )
