"""Training loss of SQ-BHOI (DETR-style set criterion)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.core import register
from src.misc.dist import get_world_size, is_dist_available_and_initialized

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from .oks_loss import OKSLoss

__all__ = ["SQBHOICriterion"]


@register
class SQBHOICriterion(nn.Module):
    """Hungarian-matched set criterion for person / hand-object-interaction training.

    Losses (each multiplied by `weight_dict[key]`; keys missing from `weight_dict` are dropped):
        loss_vfl                 varifocal classification loss
        loss_bbox, loss_giou     L1 / GIoU of the matched boxes
        loss_2d_kpts, loss_oks   L1 / OKS of the 17 keypoints of matched persons (label 0)
        loss_vis                 keypoint visibility BCE
        loss_oih_bbox            L1 of the right / left hand boxes of matched persons
        loss_oih_sim             focal loss of the "which query is my object" similarity logits
                                 (the last logit is the "no object" slot)
    The final decoder output gets the plain keys, intermediate decoder layers and the encoder
    proposals get `_aux_<i>`. Denoising queries get `_dn_<i>` and only use the box / class losses.

    Normalisers are averaged over processes: boxes -> all targets, keypoints / vis -> targets with
    `valid_pose`, oih -> hands with `valid_oih`. Targets carry a `cost` list ('cls', 'box', and
    'body' / 'oih' when keypoints / hand annotations exist) which selects the images used by the
    keypoint / oih losses.
    """

    __share__ = [
        "num_classes",
    ]
    __inject__ = [
        "matcher",
    ]

    losses = ("vfl", "boxes", "poses", "vis", "oih")
    dn_losses = ("vfl", "boxes")

    def __init__(self, matcher, weight_dict, num_classes=81, alpha=0.75, gamma=2.0):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.oks = OKSLoss(linear=True, num_keypoints=17)

    @staticmethod
    def _src_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _person_pairs(indices, targets, key):
        """Per image: (batch index, matched queries, matched targets) of label-0 targets
        in images whose `cost` contains `key`."""
        pairs = []
        for b, (t, (q, j)) in enumerate(zip(targets, indices)):
            if key in t["cost"]:
                keep = t["labels"].cpu()[j] == 0
                if keep.any():
                    pairs.append((b, q[keep], j[keep]))
        return pairs

    def loss_vfl(self, outputs, targets, indices, norms):
        idx = self._src_idx(indices)
        src_logits = outputs["pred_logits"]
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        if len(src_boxes) == 0 or len(target_boxes) == 0:
            ious = torch.tensor([]).to(src_logits).detach()
        else:
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()

        target_classes_o = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / norms["boxes"]
        return {"loss_vfl": loss}

    def loss_boxes(self, outputs, targets, indices, norms):
        pred_boxes = outputs["pred_boxes"]
        src_boxes = pred_boxes[self._src_idx(indices)]
        if len(src_boxes) == 0:
            return {"loss_bbox": 0.0 * pred_boxes.sum(), "loss_giou": 0.0 * pred_boxes.sum()}
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        return {
            "loss_bbox": loss_bbox.sum() / norms["boxes"],
            "loss_giou": loss_giou.sum() / norms["boxes"],
        }

    def loss_poses(self, outputs, targets, indices, norms):
        pred_poses = outputs["pred_poses"]
        pairs = self._person_pairs(indices, targets, "body")
        if not pairs:
            return {"loss_2d_kpts": 0.0 * pred_poses.sum(), "loss_oks": 0.0 * pred_poses.sum()}

        src = torch.cat([pred_poses[b, q] for b, q, _ in pairs])
        src = src.reshape(src.shape[0], -1, 2)
        target = torch.cat([targets[b]["2d_pose"][j] for b, _, j in pairs]).to(src)
        areas = torch.cat([targets[b]["norm_area"][j] for b, _, j in pairs]).to(src)
        conf = target[..., 2:3].clip(max=1)
        target = target[..., :2]

        loss_kpts = F.l1_loss(src, target, reduction="none") * conf
        loss_oks = self.oks(
            src.reshape(src.shape[0], -1),
            target.reshape(target.shape[0], -1),
            conf.squeeze(-1),
            areas,
        )
        return {
            "loss_2d_kpts": loss_kpts.sum() / norms["persons"],
            "loss_oks": loss_oks.sum() / norms["persons"],
        }

    def loss_vis(self, outputs, targets, indices, norms):
        pred_vis = outputs["pred_vis"]
        pairs = self._person_pairs(indices, targets, "body")
        if not pairs:
            return {"loss_vis": 0.0 * pred_vis.sum()}

        src = torch.cat([pred_vis[b, q] for b, q, _ in pairs])
        target = torch.cat([targets[b]["2d_pose"][j] for b, _, j in pairs]).to(src)
        target_vis = target[..., 2].clip(max=1)
        loss = F.binary_cross_entropy_with_logits(src, target_vis, reduction="none")
        return {"loss_vis": loss.sum() / norms["persons"]}

    def loss_oih(self, outputs, targets, indices, norms):
        """`pred_oih`: [B, Q, 2 (right, left), nq + 4] = nq similarity logits + hand box (cxcywh)."""
        pred_oih = outputs["pred_oih"]
        pairs = self._person_pairs(indices, targets, "oih")
        if not pairs:
            return {"loss_oih_bbox": 0.0 * pred_oih.sum(), "loss_oih_sim": 0.0 * pred_oih.sum()}

        nq = pred_oih.shape[-1] - 4
        src = torch.cat([pred_oih[b, q] for b, q, _ in pairs])
        valid = torch.cat([targets[b]["valid_oih"][j] for b, _, j in pairs]).to(src)
        target_boxes = torch.cat([targets[b]["oih_boxes"][j] for b, _, j in pairs])

        # Query matched to the object each hand holds (`oih_ids` refer to target `ids`);
        # -1 -> last "no object" slot.
        target_q = []
        for b, _, j in pairs:
            matched_q, matched_j = indices[b]
            matched_ids = targets[b]["ids"][matched_j]
            oih_ids = targets[b]["oih_ids"][j]
            hit = (matched_ids[None, None, :] == oih_ids[..., None]).float().argmax(-1)
            target_q.append(torch.where(oih_ids != -1, matched_q.to(hit.device)[hit], nq - 1))
        target_q = torch.cat(target_q).reshape(-1)

        src_sim = src[:, :, :-4].reshape(target_q.shape[0], nq)
        target_sim = F.one_hot(target_q, num_classes=nq).to(src_sim.dtype)
        loss_sim = torchvision.ops.sigmoid_focal_loss(
            src_sim, target_sim, alpha=0.25, gamma=2.0, reduction="none"
        )
        loss_sim = (loss_sim * valid.reshape(-1, 1)).sum()

        loss_box = F.l1_loss(src[:, :, -4:], target_boxes, reduction="none").sum(-1) * valid
        return {
            "loss_oih_bbox": loss_box.sum() / norms["oihs"],
            "loss_oih_sim": loss_sim / norms["oihs"],
        }

    def _layer_losses(self, outputs, targets, indices, norms, suffix="", names=None):
        out = {}
        for name in names or self.losses:
            for k, v in getattr(self, f"loss_{name}")(outputs, targets, indices, norms).items():
                if k in self.weight_dict:
                    out[k + suffix] = v * self.weight_dict[k]
        return out

    def _norms(self, targets, device):
        counts = torch.tensor(
            [
                sum(len(t["labels"]) for t in targets),
                sum(int(t["valid_pose"].sum()) for t in targets),
                sum(int(t["valid_oih"].sum()) for t in targets),
            ],
            dtype=torch.float,
            device=device,
        )
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(counts)
        counts = torch.clamp(counts / get_world_size(), min=1).tolist()
        return dict(boxes=counts[0], persons=counts[1], oihs=counts[2])

    def forward(self, outputs, targets):
        """Returns {loss name: weighted scalar tensor}; the total loss is the sum of all values."""
        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}
        indices = self.matcher(outputs_without_aux, targets)
        norms = self._norms(targets, outputs["pred_logits"].device)

        losses = self._layer_losses(outputs, targets, indices, norms)

        for i, aux_outputs in enumerate(outputs.get("aux_outputs", [])):
            aux_indices = self.matcher(aux_outputs, targets)
            losses.update(self._layer_losses(aux_outputs, targets, aux_indices, norms, f"_aux_{i}"))

        dn_meta = outputs.get("dn_meta")
        if dn_meta is not None and "dn_aux_outputs" in outputs:
            dn_indices = self.get_cdn_matched_indices(dn_meta, targets)
            dn_norms = dict(
                norms,
                boxes=norms["boxes"] * dn_meta["dn_num_group"],
                persons=norms["persons"] * dn_meta["dn_num_group"],
            )
            for i, aux_outputs in enumerate(outputs["dn_aux_outputs"]):
                losses.update(
                    self._layer_losses(
                        aux_outputs, targets, dn_indices, dn_norms, f"_dn_{i}", self.dn_losses
                    )
                )
        elif dn_meta is None:
            # no denoising group (e.g. batch without targets): keep the key set unchanged
            zero = torch.zeros((), device=outputs["pred_logits"].device)
            for i in range(len(outputs.get("aux_outputs", []))):
                for k in ("loss_vfl", "loss_bbox", "loss_giou"):
                    if k in self.weight_dict:
                        losses[f"{k}_dn_{i}"] = zero

        return losses

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """Denoising queries of each group are matched to the targets in order."""
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        dn_match_indices = []
        for i, t in enumerate(targets):
            num_gt = len(t["labels"])
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64).tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i].cpu(), gt_idx))
            else:
                dn_match_indices.append(
                    (torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64))
                )
        return dn_match_indices
