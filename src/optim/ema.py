"""
reference:
https://github.com/ultralytics/yolov5/blob/master/utils/torch_utils.py#L404

by lyuwenyu
"""

import math
from copy import deepcopy

import torch
import torch.nn as nn

import src.misc.dist as dist
from src.core import register

__all__ = ["ModelEMA"]


@register
class ModelEMA(object):
    """Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmups: int = 2000, k=1):
        """`k`: number of base-batch steps per update (global batch / base batch), scales
        `updates` by k and the decay by sqrt(k) so that the EMA horizon is batch-size independent."""
        super().__init__()

        # Create EMA
        self.module = deepcopy(dist.de_parallel(model)).eval()  # FP32 EMA

        # if next(model.parameters()).device.type != 'cpu':
        #     self.module.half()  # FP16 EMA

        self.decay = decay
        self.warmups = warmups
        self.updates = 0  # number of EMA updates
        # self.filter_no_grad = filter_no_grad
        self.decay_fn = lambda x: (
            decay * (1 - math.exp(-x / warmups))
        )  # decay exponential ramp (to help early epochs)
        self.k = k

        # NOTE: the parameters keep `requires_grad=True` (the EMA is updated under no_grad and only
        # evaluated under no_grad). Freezing them would make nn.MultiheadAttention take its fused
        # inference fast path, which differs numerically (~1e-6) from the plain model used by
        # tools/eval.py and shifts the metrics by ~1e-4.

    def update(self, model: nn.Module):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1 * self.k
            d = self.decay_fn(self.updates) ** math.sqrt(self.k)

            new_params = dict(dist.de_parallel(model).named_parameters())
            new_buffers = dict(dist.de_parallel(model).named_buffers())
            for k, v in self.module.state_dict().items():
                if k in new_params:
                    if v.dtype.is_floating_point and new_params[k].requires_grad:
                        v *= d
                        v += (1 - d) * new_params[k].detach()
                elif k in new_buffers:
                    if v.dtype.is_floating_point:
                        v *= d
                        v += (1 - d) * new_buffers[k].detach()

    def to(self, *args, **kwargs):
        self.module = self.module.to(*args, **kwargs)
        return self

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        # Update EMA attributes
        self.copy_attr(self.module, model, include, exclude)

    @staticmethod
    def copy_attr(a, b, include=(), exclude=()):
        # Copy attributes from b to a, options to only include [...] and to exclude [...]
        for k, v in b.__dict__.items():
            if (len(include) and k not in include) or k.startswith("_") or k in exclude:
                continue
            else:
                setattr(a, k, v)

    def state_dict(
        self,
    ):
        return dict(module=self.module.state_dict(), updates=self.updates, warmups=self.warmups)

    def load_state_dict(self, state):
        self.module.load_state_dict(state["module"])
        if "updates" in state:
            self.updates = state["updates"]

    def forwad(
        self,
    ):
        raise RuntimeError("ema...")

    def extra_repr(self) -> str:
        return f"decay={self.decay}, warmups={self.warmups}"
