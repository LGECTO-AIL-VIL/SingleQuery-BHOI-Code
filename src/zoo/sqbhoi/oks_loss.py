"""Object Keypoint Similarity (OKS) loss for COCO 17-keypoint poses."""

import numpy as np
import torch
import torch.nn as nn

__all__ = ["OKSLoss"]

COCO_SIGMAS = (
    np.array(
        [
            0.26,
            0.25,
            0.25,
            0.35,
            0.35,
            0.79,
            0.79,
            0.72,
            0.72,
            0.62,
            0.62,
            1.07,
            1.07,
            0.87,
            0.87,
            0.89,
            0.89,
        ],
        dtype=np.float32,
    )
    / 10.0
)


def oks_overlaps(kpt_preds, kpt_gts, kpt_valids, kpt_areas, sigmas):
    """Per-instance OKS. kpts: [N, 2K] (x, y normalised), valids: [N, K], areas: [N]."""
    sigmas = kpt_preds.new_tensor(sigmas)
    variances = (sigmas * 2) ** 2

    kpt_preds = kpt_preds.reshape(-1, kpt_preds.size(-1) // 2, 2)
    kpt_gts = kpt_gts.reshape(-1, kpt_gts.size(-1) // 2, 2)

    squared_distance = (kpt_preds[:, :, 0] - kpt_gts[:, :, 0]) ** 2 + (
        kpt_preds[:, :, 1] - kpt_gts[:, :, 1]
    ) ** 2
    squared_distance = squared_distance / (kpt_areas[:, None] * variances[None, :] * 2)
    squared_distance = torch.exp(-squared_distance) * kpt_valids
    return squared_distance.sum(dim=1) / (kpt_valids.sum(dim=1) + 1e-6)


class OKSLoss(nn.Module):
    """Returns one loss value per instance (`1 - oks` if `linear` else `-log(oks)`)."""

    def __init__(self, linear=False, num_keypoints=17, eps=1e-6, loss_weight=1.0):
        super().__init__()
        if num_keypoints != 17:
            raise ValueError(f"Unsupported keypoints number {num_keypoints}")
        self.linear = linear
        self.eps = eps
        self.loss_weight = loss_weight
        self.sigmas = COCO_SIGMAS

    def forward(self, pred, target, valid, area):
        oks = oks_overlaps(pred, target, valid, area, self.sigmas).clamp(min=self.eps)
        loss = 1 - oks if self.linear else -oks.log()
        return self.loss_weight * loss
