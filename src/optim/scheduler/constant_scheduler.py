import warnings

from torch.optim.lr_scheduler import LRScheduler


# ref : torch.optim.lr_scheduler.ConstantLR
class ContinuosConstantLR(LRScheduler):
    """Multiply the lr by `factor` at step 0 and undo it at step `total_iters`."""

    def __init__(self, optimizer, factor=1.0, total_iters=5, last_epoch=-1):
        self.factor = factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [group["lr"] * self.factor for group in self.optimizer.param_groups]

        if self.last_epoch == self.total_iters:
            return [group["lr"] * (1.0 / self.factor) for group in self.optimizer.param_groups]

        return [group["lr"] for group in self.optimizer.param_groups]
