"""
reference
- https://github.com/pytorch/vision/blob/main/references/detection/utils.py
- https://github.com/facebookresearch/detr/blob/master/util/misc.py#L406

by lyuwenyu
"""

import os
import random

import numpy as np
import torch
import torch.distributed
import torch.distributed as tdist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler


def init_distributed():
    """
    distributed setup
    args:
        backend (str), ('nccl', 'gloo')
    """
    try:
        # launched by torchrun / torch.distributed.run (env:// rendezvous)
        tdist.init_process_group(
            init_method="env://",
        )
        torch.distributed.barrier()

        local_rank = get_local_rank()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        setup_print(get_rank() == 0)
        print("Initialized distributed mode...")

        return True

    except Exception as e:
        print(f"Not init distributed mode. - {e}")
        return False


def setup_print(is_main):
    """This function disables printing when not in master process"""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if force:
            args = (f"[Rank {get_rank()}]:",) + args
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_available_and_initialized():
    if not tdist.is_available():
        return False
    if not tdist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_available_and_initialized():
        return 0
    return tdist.get_rank()


def get_local_rank():
    if not is_dist_available_and_initialized():
        return 0
    return int(os.environ["LOCAL_RANK"])


def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return tdist.get_world_size()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def warp_model(
    model,
    find_unused_parameters=False,
    sync_bn=False,
):
    if is_dist_available_and_initialized():
        local_rank = get_local_rank()
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) if sync_bn else model
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters,
            broadcast_buffers=True,
        )
    return model


def seed_worker(worker_id):
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def warp_loader(loader, shuffle=False, seed=-1):
    if is_dist_available_and_initialized():
        rank = get_rank()
        sampler = DistributedSampler(
            loader.dataset, shuffle=shuffle, num_replicas=get_world_size(), rank=rank
        )
        generator = torch.Generator().manual_seed(seed + rank)
        loader = loader.__class__(
            loader.dataset,
            loader.batch_size,
            sampler=sampler,
            worker_init_fn=seed_worker,
            drop_last=loader.drop_last,
            collate_fn=loader.collate_fn,
            pin_memory=loader.pin_memory,
            num_workers=loader.num_workers,
            generator=generator,
        )
    return loader


def is_parallel(model) -> bool:
    # Returns True if model is of type DP or DDP
    return type(model) in (
        torch.nn.parallel.DataParallel,
        torch.nn.parallel.DistributedDataParallel,
    )


def de_parallel(model) -> nn.Module:
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def reduce_dict(data, avg=True):
    """
    Args
        data dict: input, {k: v, ...}
        avg bool: true
    """
    world_size = get_world_size()
    if world_size < 2:
        return data

    with torch.no_grad():
        keys, values = [], []
        for k in sorted(data.keys()):
            keys.append(k)
            values.append(data[k])

        values = torch.stack(values, dim=0).to(torch.float)
        tdist.all_reduce(values)

        if avg is True:
            values /= world_size

        _data = {k: v for k, v in zip(keys, values)}

    return _data


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    tdist.all_gather_object(data_list, data)
    return data_list


def set_seed(seed, gl_seed=0):
    """set random seed. `seed` < 0 keeps cudnn non-deterministic (faster)"""
    if seed >= 0 and gl_seed >= 0:
        seed += gl_seed
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
