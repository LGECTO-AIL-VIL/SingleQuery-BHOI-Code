import torch
import torch.utils.data as data

from src.core import register

__all__ = ["DataLoader", "default_collate_fn"]


@register
class DataLoader(data.DataLoader):
    __inject__ = ["dataset", "collate_fn"]

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ["dataset", "batch_size", "num_workers", "drop_last", "collate_fn"]:
            format_string += "\n"
            format_string += f"    {n}: {getattr(self, n)}"
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)


@register
def default_collate_fn(items):
    """Stack images into a batch tensor; targets stay a list of dicts."""
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]
