import os
import sys
import tempfile

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

import src.train as t
import src.utils.data as data
import src.utils.pprint as pprint
import src.utils.utils as utils

if os.uname()[1] == "iss":
    BASE_PATH = "/home/edgar/Documents/Datasets/deepmeta/Data/3classes_metas/"
else:
    BASE_PATH = "/home/elefevre/Datasets/deepmeta/3classesv2/3classesv2_full/"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,5"

SAVE_PATH = "data/Res_Unet3p_lovasz+focal_convsep_full.pth"
LOSS = np.inf
METRIC = [-1.0, -1.0, -1.0]


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    config = utils.get_args()
    dataloader = data.get_datasets(
        f'{BASE_PATH}Images/', f'{BASE_PATH}Labels/', config
    )
    setup(rank, world_size)
    model = utils.get_model(config).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    history_train, history_val, _ = t.train(ddp_model, dataloader, config)
    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(demo_basic, world_size)
