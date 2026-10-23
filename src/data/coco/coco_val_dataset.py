"""COCO validation set read from the official annotation files (e.g. instances_val2017.json)."""

import os
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
from PIL import Image, ImageOps
from pycocotools.coco import COCO

from src.core import register

__all__ = ["CocoValDetection"]


@register
class CocoValDetection:
    """Images of an official COCO annotation file; targets only hold `image_id` and `orig_size` ([W, H])."""

    __inject__ = ["transforms", "evaluators"]

    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        transforms: Optional[Callable] = None,
        evaluators: Optional[Iterable] = None,
        dataset_name: str = "COCO",
    ) -> None:
        self.coco = COCO(ann_file)
        self.img_ids = sorted(self.coco.getImgIds())
        self.img_folder = img_folder
        self._transforms = transforms
        self.evaluators = evaluators
        self.dataset_name = dataset_name

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img = Image.open(os.path.join(self.img_folder, img_info["file_name"])).convert("RGB")
        img = ImageOps.exif_transpose(img)
        target = {
            "image_id": img_id,
            "orig_size": torch.tensor(img.size, dtype=torch.float32),  # [W, H]
        }
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def get_evaluators(self) -> Iterable:
        return self.evaluators
