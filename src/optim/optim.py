import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from src.core import register

from .scheduler import ContinuosConstantLR, ContinuosCosineAnnealingLR, QuadraticWarmupLR

__all__ = [
    "AdamW",
    "SGD",
    "Adam",
    "MultiStepLR",
    "CosineAnnealingLR",
    "OneCycleLR",
    "ContinuosConstantLR",
    "QuadraticWarmupLR",
    "ContinuosCosineAnnealingLR",
    "SequentialLR",
]


SGD = register(optim.SGD)
Adam = register(optim.Adam)
AdamW = register(optim.AdamW)


MultiStepLR = register(lr_scheduler.MultiStepLR)
CosineAnnealingLR = register(lr_scheduler.CosineAnnealingLR)
OneCycleLR = register(lr_scheduler.OneCycleLR)
ContinuosConstantLR = register(ContinuosConstantLR)
QuadraticWarmupLR = register(QuadraticWarmupLR)
ContinuosCosineAnnealingLR = register(ContinuosCosineAnnealingLR)

# schedulers that can be nested in `SequentialLR`
_SCHEDULERS = {
    s.__name__: s
    for s in (
        MultiStepLR,
        CosineAnnealingLR,
        ContinuosConstantLR,
        QuadraticWarmupLR,
        ContinuosCosineAnnealingLR,
    )
}


def build_iter_from_epoch(scheduler, *args, epoch_length=None, **kwargs):
    """
    Build an iter-based lr_scheduler of the scheduler from an epoch-based config:
    epoch-valued arguments (milestones, T_max, total_iters, begin, end) are multiplied by `epoch_length`.
    """
    assert epoch_length is not None and epoch_length > 0, (
        f"`epoch_length` must be a positive integer, but got {epoch_length}."
    )

    # for MultiStepLR
    if "milestones" in kwargs:
        milestones = kwargs["milestones"]
        assert isinstance(milestones, list)
        kwargs["milestones"] = [int(milestone * epoch_length) for milestone in milestones]

    # for CosineAnnealingLR and ContinuosCosineAnnealingLR
    if "T_max" in kwargs:
        kwargs["T_max"] = kwargs["T_max"] * epoch_length

    # for ContinuosConstantLR
    if "total_iters" in kwargs:
        kwargs["total_iters"] = int(kwargs["total_iters"] * epoch_length)

    # for QuadraticWarmupLR
    if "begin" in kwargs:
        kwargs["begin"] = int(kwargs["begin"] * epoch_length)
    if "end" in kwargs:
        kwargs["end"] = int(kwargs["end"] * epoch_length)

    return scheduler(*args, **kwargs)


# ref: torch.optim.lr_scheduler.SequentialLR
@register
class SequentialLR(lr_scheduler.SequentialLR):
    """`SequentialLR` built from a list of scheduler configs (`type` + args).

    With `convert_to_iter_based`, milestones and the epoch-valued arguments of the schedulers
    are given in epochs and converted to iterations using `epoch_length`; the scheduler must
    then be stepped every iteration.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        schedulers: list,
        milestones,
        last_epoch=-1,
        convert_to_iter_based=False,
        epoch_length=None,
    ) -> None:
        assert len(schedulers) > 0, "schedulers is not defined"
        assert not convert_to_iter_based or epoch_length > 0, (
            "epoch_length should be defined to convert scheduler based on iteration."
        )

        self.iter_based = convert_to_iter_based
        if convert_to_iter_based:
            milestones_ = [int(milestone * epoch_length) for milestone in milestones]
        else:
            milestones_ = milestones

        schedulers_ = []
        for scheduler_config in schedulers:
            assert "type" in scheduler_config, "type for scheduler does not defined."
            scheduler_config = dict(scheduler_config)
            scheduler_class = _SCHEDULERS[scheduler_config.pop("type")]
            if convert_to_iter_based:
                scheduler = build_iter_from_epoch(
                    scheduler_class,
                    optimizer,
                    epoch_length=epoch_length,
                    last_epoch=last_epoch,
                    **scheduler_config,
                )
            else:
                scheduler = scheduler_class(optimizer, last_epoch=last_epoch, **scheduler_config)
            schedulers_.append(scheduler)

        super().__init__(optimizer, schedulers_, milestones_, last_epoch)
