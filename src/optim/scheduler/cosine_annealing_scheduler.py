import math
import warnings

from torch.optim.lr_scheduler import LRScheduler


# ref: https://github.com/open-mmlab/mmengine/blob/85c83ba61689907fb1775713622b1b146d82277b/mmengine/optim/scheduler/param_scheduler.py#L567
class ContinuosCosineAnnealingLR(LRScheduler):
    r"""Cosine annealing of the lr of each parameter group (chainable form of
    `torch.optim.lr_scheduler.CosineAnnealingLR`), with the minimum lr given either
    as an absolute value (`eta_min`) or as a ratio of the base lr (`eta_min_ratio`).

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0 if `eta_min_ratio` is None.
        eta_min_ratio (float): Minimum learning rate as a ratio of the base lr.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, T_max, eta_min=None, eta_min_ratio=None, last_epoch=-1):
        if eta_min is None and eta_min_ratio is None:
            eta_min = 0.0
        assert (eta_min is None) ^ (eta_min_ratio is None), (
            "Either `eta_min` or `eta_min_ratio should be specified"
        )

        self.T_max = T_max
        self.eta_min = eta_min
        self.eta_min_ratio = eta_min_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):

        def _get_eta_min(base_value):
            if self.eta_min_ratio is None:
                return self.eta_min
            return base_value * self.eta_min_ratio

        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group["lr"]
                + (base_lr - _get_eta_min(base_lr)) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
            * (group["lr"] - _get_eta_min(base_lr))
            + _get_eta_min(base_lr)
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
        ]
