"""SQ-BHOI PostProcessor: detection, body pose and similarity-based OiH output."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.core import register

__all__ = ["SQBHOIOiHPostProcessor"]


@register
class SQBHOIOiHPostProcessor(nn.Module):
    __share__ = [
        "num_classes",
        "use_focal_loss",
        "num_top_queries",
        "task",
        "oih_config",
        "num_kpts",
    ]

    def __init__(
        self,
        num_classes=81,
        use_focal_loss=True,
        num_top_queries=300,
        task=None,
        oih_config=None,
        num_kpts=17,
    ):
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.task = task if task is not None else ["detection"]
        self.deploy_mode = False
        self.normalized = False
        self.num_kpts = num_kpts

        if oih_config is None:
            oih_config = {"oih_cls": 2, "oih_sim": True}
        assert oih_config.get("oih_sim", False), (
            "Only the similarity-based OiH head is supported; set `oih_sim: True` in oih_config."
        )
        self.oih_config = oih_config
        for k, v in oih_config.items():
            setattr(self, k, v)

    def extra_repr(self):
        return (
            f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, "
            f"num_top_queries={self.num_top_queries}"
        )

    def forward(self, outputs, orig_target_sizes):
        # Detection
        if "detection" in self.task:
            logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
            labels, boxes, scores, index = self._postprocess_detection(
                logits, boxes, orig_target_sizes
            )
        else:
            labels, boxes, scores, index = [], [], [], None

        # Body pose
        if "body" in self.task and outputs.get("pred_poses") is not None:
            poses = outputs["pred_poses"].clone()
            pred_vis = outputs.get("pred_vis")
            poses = self._postprocess_body(poses, index, pred_vis, orig_target_sizes)
        else:
            poses = []

        # OiH
        if "oih" in self.task and outputs.get("pred_oih") is not None:
            oihs = outputs["pred_oih"]
            oihs = self._postprocess_oih(
                oihs,
                index,
                orig_target_sizes,
                det_labels=outputs["pred_logits"],
                det_boxes=outputs["pred_boxes"],
            )
        else:
            oihs = []

        if self.deploy_mode:
            labels = None if labels == [] else labels
            boxes = None if boxes == [] else boxes
            scores = None if scores == [] else scores
            poses = None if poses == [] else poses
            oihs = None if oihs == [] else oihs
            return labels, boxes, scores, None, poses, None, oihs

        results = []
        for i in range(len(labels) if not isinstance(labels, list) or labels else 1):
            result = dict(
                labels=labels[i] if not isinstance(labels, list) else None,
                boxes=boxes[i] if not isinstance(boxes, list) else None,
                scores=scores[i] if not isinstance(scores, list) else None,
                poses=poses[i] if not isinstance(poses, list) and i < len(poses) else None,
                oihs=oihs[i] if not isinstance(oihs, list) and i < len(oihs) else None,
            )
            results.append(result)

        return results

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self

    @property
    def iou_types(self):
        return ("bbox",)

    def _postprocess_detection(self, logits, boxes, orig_target_sizes):
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        if not self.normalized:
            bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, axis=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            boxes = bbox_pred.gather(
                dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])
            )
        else:
            index = None
            scores, labels = logits.sigmoid().max(-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(
                    bbox_pred, dim=1, index=index.unsqueeze(-1).tile(1, 1, bbox_pred.shape[-1])
                )

        return labels, boxes, scores, index

    def _postprocess_body(self, poses, index, pred_vis, orig_target_sizes):
        if self.use_focal_loss and index is not None:
            poses = poses.gather(
                dim=1,
                index=index.unsqueeze(-1).repeat(1, 1, poses.shape[-1]),
            )
            if pred_vis is not None:
                pred_vis = pred_vis.gather(
                    dim=1,
                    index=index.unsqueeze(-1).repeat(1, 1, pred_vis.shape[-1]),
                )
        else:
            if poses.shape[1] > self.num_top_queries:
                poses = torch.gather(
                    poses,
                    dim=1,
                    index=index.unsqueeze(-1).tile(1, 1, poses.shape[-1]),
                )
                if pred_vis is not None:
                    pred_vis = pred_vis.gather(
                        dim=1,
                        index=index.unsqueeze(-1).repeat(1, 1, pred_vis.shape[-1]),
                    )

        poses = poses.reshape(-1, self.num_top_queries, self.num_kpts, 2)
        if not self.normalized:
            poses *= orig_target_sizes.unsqueeze(1).unsqueeze(1)
        if pred_vis is not None:
            vis = pred_vis.reshape(-1, self.num_top_queries, self.num_kpts, 1).sigmoid()
            poses = torch.cat([poses, vis], dim=-1)

        return poses

    def _postprocess_oih(self, oihs, index, orig_target_sizes, det_labels=None, det_boxes=None):
        oih_hand_boxes = oihs[..., -4:]
        for i in range(oih_hand_boxes.shape[2]):
            oih_hand_boxes[:, :, i] = torchvision.ops.box_convert(
                oih_hand_boxes[:, :, i], in_fmt="cxcywh", out_fmt="xyxy"
            )
            if not self.normalized:
                oih_hand_boxes[:, :, i] *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        oih_sim = oihs[..., :-4]
        oih_scores, oih_idxs = oih_sim.sigmoid().max(-1)
        bs, nq = oih_idxs.shape[:2]
        oih_labels = torch.zeros_like(oih_idxs)
        oih_labels[oih_idxs < nq] = 1  # on status
        oih_labels[oih_idxs == nq] = 0  # off status

        bbox_pred = torchvision.ops.box_convert(det_boxes, in_fmt="cxcywh", out_fmt="xyxy")
        if not self.normalized:
            bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        oih_idxs_nonoff = oih_idxs.clone()
        oih_idxs_nonoff[oih_idxs == nq] = 0
        oih_obj_boxes = torch.gather(
            bbox_pred.unsqueeze(2).repeat(1, 1, 2, 1),
            dim=1,
            index=oih_idxs_nonoff.unsqueeze(-1).repeat(1, 1, 1, 4),
        )
        oih_obj_boxes[oih_idxs == nq] = oih_hand_boxes[oih_idxs == nq]

        scores, labels = det_labels.sigmoid().max(-1)
        labels = torch.gather(
            labels.unsqueeze(-1).repeat(1, 1, 2).unsqueeze(-1),
            dim=1,
            index=oih_idxs_nonoff.unsqueeze(-1).repeat(1, 1, 1, 1),
        )

        oih_results = torch.cat(
            [
                oih_scores.unsqueeze(-1),
                oih_labels.unsqueeze(-1).float(),
                oih_hand_boxes,
                oih_obj_boxes,
                labels,
            ],
            dim=-1,
        )
        oih_results = oih_results.view(bs, nq, -1)

        oih_results = torch.gather(
            oih_results, dim=1, index=index.unsqueeze(-1).repeat(1, 1, oih_results.shape[-1])
        )

        return oih_results
