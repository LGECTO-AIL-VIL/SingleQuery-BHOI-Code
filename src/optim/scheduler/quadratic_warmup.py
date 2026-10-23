from torch.optim.lr_scheduler import LRScheduler

INF = int(1e9)


# ref: https://github.com/open-mmlab/mmpose/blob/main/mmpose/engine/schedulers/quadratic_warmup.py
# Modified for compitable to torch.optim.lr_scheduler
class QuadraticWarmupLR(LRScheduler):
    r"""Warm up the lr of each parameter group by quadratic formula:

    .. math::

        X_{t} = X_{t-1} + \frac{2t+1}{{(end-begin)}^{2}} \times X_{base}

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        begin (int): Step at which to start updating the lr. Defaults to 0.
        end (int): Step at which to stop updating the lr.
        last_epoch (int): The index of last step. Defaults to -1.
    """

    def __init__(self, optimizer, begin: int = 0, end: int = INF, last_epoch: int = -1):
        if end >= INF:
            raise ValueError(
                "``end`` must be less than infinity,"
                "Please set ``end`` parameter of "
                "``QuadraticWarmupScheduler`` as the "
                "number of warmup end."
            )
        self.total_iters = end - begin
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute lr using chainable form of the scheduler."""
        if self.last_epoch == 0:
            return [
                base_value * (2 * self.last_epoch + 1) / self.total_iters**2
                for base_value in self.base_lrs
            ]

        return [
            group["lr"] + base_value * (2 * self.last_epoch + 1) / self.total_iters**2
            for base_value, group in zip(self.base_lrs, self.optimizer.param_groups)
        ]
