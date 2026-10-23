"""Training / evaluation transforms.

Torchvision v2 ops (photometric distort, resize, normalize, ...) operate on the
image only; the custom ops below (`RandomHorizontalFlip`, `RandomPerspective`,
`BoxConvert`) take and return an `(image, target)` pair and also update the
body-keypoint / hand-object (OiH) fields of `target`.
"""

import math
import random
from typing import Any, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
from PIL import Image

from src.core import GLOBAL_CONFIG, register

from .oih.oih_data_utils import oih_box_convert, oih_flip, oih_perspective

torchvision.disable_beta_transforms_warning()

__all__ = ["Compose"]

RandomPhotometricDistort = register(T.RandomPhotometricDistort)
Resize = register(T.Resize)
Normalize = register(T.Normalize)


@register
class Compose(T.Compose):
    def __init__(self, ops: List) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    op = dict(op)
                    name = op.pop("type")
                    transforms.append(getattr(GLOBAL_CONFIG[name]["_pymodule"], name)(**op))
                elif isinstance(op, nn.Module):
                    transforms.append(op)
                else:
                    raise ValueError(f"Unsupported transform spec: {op!r}")
        else:
            transforms = [EmptyTransform()]

        super().__init__(transforms=transforms)


@register
class EmptyTransform(T.Transform):
    def forward(self, *inputs: Any) -> Any:
        return inputs if len(inputs) > 1 else inputs[0]


@register
class ToImageTensor(T.ToImage):
    """PIL image -> uint8 image tensor."""


@register
class ConvertDtype(T.ToDtype):
    """Image tensor -> float32 in [0, 1]."""

    def __init__(self, scale: bool = True) -> None:
        super().__init__(torch.float32, scale)


@register
class RandomHorizontalFlip:
    """Flip image, boxes, body keypoints (left/right swapped) and OiH targets."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p
        self.mapping_kpts = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

    def __call__(self, img: Image.Image, target: dict) -> Tuple[Image.Image, dict]:
        if random.random() > self.p:
            return img, target

        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        w, _ = target["size"]

        target["boxes"][:, [0, 2]] = w - target["boxes"][:, [2, 0]]

        pose = target["2d_pose"][:, self.mapping_kpts]
        pose[..., 0] = torch.where(pose[..., 2] != 0, w - pose[..., 0], pose[..., 0])
        target["2d_pose"] = pose

        target = oih_flip(target, w)

        return img, target


@register
class RandomPerspective:
    """Random affine (rotation/scale/shear/translation) warp of image, boxes,
    keypoints and OiH targets, followed by resizing to `size`.
    """

    def __init__(
        self,
        p: float = 0.5,
        degrees: float = 10.0,
        translate: float = 0.1,
        scale: float = 0.1,
        shear: float = 10.0,
        perspective: float = 0.0,
        size: List[int] = [640, 640],
        norm: bool = True,
        ar_thr: float = 20.0,
        area_thr: float = 0.1,
    ) -> None:
        self.p = p
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.size = size
        self.norm = norm
        self.ar_thr = ar_thr
        self.area_thr = area_thr

    @staticmethod
    def box_candidates(
        box1: np.ndarray,
        box2: np.ndarray,
        wh_thr: int = 2,
        ar_thr: float = 20,
        area_thr: float = 0.1,
        eps: float = 1e-16,
    ) -> np.ndarray:
        """box1(4,n) before / box2(4,n) after augmentation -> boolean keep mask."""
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
        return (
            (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)
        )

    @staticmethod
    def box_norm(boxes: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        boxes[:, 0] = boxes[:, 0] / size[0]
        boxes[:, 1] = boxes[:, 1] / size[1]
        boxes[:, 2] = boxes[:, 2] / size[0]
        boxes[:, 3] = boxes[:, 3] / size[1]
        return boxes

    def __call__(self, im: Image.Image, targets: dict) -> Tuple[Image.Image, dict]:
        im = np.array(im)
        height, width = im.shape[:2]

        # Center
        C = np.eye(3)
        C[0, 2] = -im.shape[1] / 2
        C[1, 2] = -im.shape[0] / 2

        # Perspective
        P = np.eye(3)
        if random.random() <= self.p:
            P[2, 0] = random.uniform(-self.perspective, self.perspective)
            P[2, 1] = random.uniform(-self.perspective, self.perspective)

        # Rotation and scale
        R = np.eye(3)
        if random.random() > self.p:
            a = 0.0
            scale_factor = 1.0
        else:
            a = random.uniform(-self.degrees, self.degrees)
            scale_factor = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=scale_factor)

        # Shear
        S = np.eye(3)
        if random.random() <= self.p:
            S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
            S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)

        # Translation
        Tr = np.eye(3)
        if random.random() > self.p:
            Tr[0, 2] = 0.5 * width
            Tr[1, 2] = 0.5 * height
        else:
            Tr[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * width
            Tr[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * height

        # Combined matrix; order of operations (right to left) is important
        M = Tr @ S @ R @ P @ C
        if (M != np.eye(3)).any():
            if self.perspective:
                im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:
                im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

        if self.size[0] != im.shape[0] or self.size[1] != im.shape[1]:
            im = cv2.resize(im, (self.size[0], self.size[1]))
        im = Image.fromarray(im)
        targets["size"] = torch.as_tensor(im.size)

        n = len(targets["labels"])
        if n:
            # warp boxes (x1y1, x2y2, x1y2, x2y1 corners)
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets["boxes"][:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2).numpy()
            xy = xy @ M.T
            xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)

            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

            # filter candidates
            i = self.box_candidates(
                box1=targets["boxes"].numpy().T * scale_factor,
                box2=new.T,
                ar_thr=self.ar_thr,
                area_thr=self.area_thr,
            )
            targets["labels"] = targets["labels"][i]
            targets["boxes"] = torch.tensor(new[i], dtype=torch.float)
            targets["iscrowd"] = targets["iscrowd"][i]
            targets["area"] = targets["area"][i]
            targets["ids"] = targets["ids"][i]
            targets["2d_pose"] = targets["2d_pose"][i]
            targets["valid_pose"] = targets["valid_pose"][i]
            targets = oih_perspective(
                targets,
                i,
                M,
                (width, height),
                scale_factor,
                self.perspective,
                self.norm,
                self.box_candidates,
                self.box_norm,
                False,
            )

            if self.norm:
                targets["boxes"] = self.box_norm(targets["boxes"], (width, height))

            targets["norm_area"] = (targets["boxes"][:, 2] - targets["boxes"][:, 0]) * (
                targets["boxes"][:, 3] - targets["boxes"][:, 1]
            )

            # warp body keypoints of persons (label 0)
            person_mask = targets["labels"] == 0
            person_targets = targets["2d_pose"][person_mask]
            if len(person_targets) > 0:
                xy = person_targets.numpy().reshape(-1, 3)
                vis = xy[:, 2:].copy()
                xy[:, 2:] = 1
                xy = xy @ M.T
                xy = xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]
                out_mask = (
                    (xy[:, 0] < 0) | (xy[:, 1] < 0) | (xy[:, 0] > width) | (xy[:, 1] > height)
                )
                vis[out_mask] = 0
                keypoints = torch.tensor(np.concatenate((xy, vis), axis=-1), dtype=torch.float)
                targets["2d_pose"][person_mask] = keypoints.reshape(person_targets.shape[0], -1, 3)
                if self.norm:
                    targets["2d_pose"][..., 0] = targets["2d_pose"][..., 0] / width
                    targets["2d_pose"][..., 1] = targets["2d_pose"][..., 1] / height

        return im, targets


@register
class BoxConvert:
    def __init__(self, in_fmt: str = "xyxy", out_fmt: str = "cxcywh") -> None:
        self.in_fmt = in_fmt
        self.out_fmt = out_fmt

    def __call__(self, im: Image.Image, targets: dict) -> Tuple[Image.Image, dict]:
        if self.in_fmt != self.out_fmt and "boxes" in targets:
            targets["boxes"] = torchvision.ops.box_convert(
                targets["boxes"], in_fmt=self.in_fmt, out_fmt=self.out_fmt
            )
            if "oih_boxes" in targets:
                targets = oih_box_convert(targets, self.in_fmt, self.out_fmt)

        return im, targets
