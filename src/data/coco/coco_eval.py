"""COCO box / keypoint evaluator (official val2017 annotations), distributed-safe.

The evaluator follows the validation-loop protocol of `src.solver.det_engine.evaluate`:
`reset()` -> `process(targets, results)` per batch -> `synchronize_between_processes()` -> `evaluate()`.
"""

import contextlib
import os
from typing import Dict, List, Sequence

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.core import register
from src.misc import dist

__all__ = ["CocoEvaluator", "CONTIGUOUS_TO_COCO_ID"]

# COCO 80-class contiguous_id -> dataset_id mapping (standard COCO)
CONTIGUOUS_TO_COCO_ID = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
    11: 13,
    12: 14,
    13: 15,
    14: 16,
    15: 17,
    16: 18,
    17: 19,
    18: 20,
    19: 21,
    20: 22,
    21: 23,
    22: 24,
    23: 25,
    24: 27,
    25: 28,
    26: 31,
    27: 32,
    28: 33,
    29: 34,
    30: 35,
    31: 36,
    32: 37,
    33: 38,
    34: 39,
    35: 40,
    36: 41,
    37: 42,
    38: 43,
    39: 44,
    40: 46,
    41: 47,
    42: 48,
    43: 49,
    44: 50,
    45: 51,
    46: 52,
    47: 53,
    48: 54,
    49: 55,
    50: 56,
    51: 57,
    52: 58,
    53: 59,
    54: 60,
    55: 61,
    56: 62,
    57: 63,
    58: 64,
    59: 65,
    60: 67,
    61: 70,
    62: 72,
    63: 73,
    64: 74,
    65: 75,
    66: 76,
    67: 77,
    68: 78,
    69: 79,
    70: 80,
    71: 81,
    72: 82,
    73: 84,
    74: 85,
    75: 86,
    76: 87,
    77: 88,
    78: 89,
    79: 90,
}

_STAT_NAMES = {
    "bbox": [
        "AP",
        "AP_50",
        "AP_75",
        "AP_S",
        "AP_M",
        "AP_L",
        "AR_1",
        "AR_10",
        "AR_100",
        "AR_S",
        "AR_M",
        "AR_L",
    ],
    "keypoints": ["AP", "AP_50", "AP_75", "AP_M", "AP_L", "AR", "AR_50", "AR_75", "AR_M", "AR_L"],
}


@register
class CocoEvaluator:
    """COCO AP of the detected boxes (`bbox`) and body keypoints (`keypoints`).

    Args:
        ann_files: `{iou_type: annotation json}`, e.g. `bbox` -> instances_val2017.json and
            `keypoints` -> person_keypoints_val2017.json.
        iou_types: subset of `ann_files` to evaluate (default: all of them).
    """

    def __init__(self, ann_files: Dict[str, str], iou_types: Sequence[str] = None) -> None:
        self.iou_types = list(iou_types) if iou_types is not None else list(ann_files)
        assert all(t in _STAT_NAMES for t in self.iou_types), self.iou_types
        self.coco_gt = {}
        for iou_type in self.iou_types:
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                self.coco_gt[iou_type] = COCO(ann_files[iou_type])
        self.reset()

    def reset(self) -> None:
        # {iou_type: {image_id: [coco result dict, ...]}}
        self.predictions = {t: {} for t in self.iou_types}

    def process(self, targets: List[Dict], results: List[Dict]) -> None:
        for target, result in zip(targets, results):
            if result["labels"] is None:
                continue
            image_id = int(target["image_id"])
            boxes = result["boxes"].cpu()  # xyxy, original image size
            scores = result["scores"].cpu()
            labels = result["labels"].cpu()

            if "bbox" in self.iou_types:
                dets = []
                for j in range(len(scores)):
                    coco_id = CONTIGUOUS_TO_COCO_ID.get(int(labels[j].item()))
                    if coco_id is None:  # non-COCO class (e.g. "object")
                        continue
                    x1, y1, x2, y2 = boxes[j].tolist()
                    dets.append(
                        {
                            "image_id": image_id,
                            "category_id": coco_id,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(scores[j]),
                        }
                    )
                self.predictions["bbox"][image_id] = dets

            if "keypoints" in self.iou_types and result.get("poses") is not None:
                poses = result["poses"].cpu()  # (nq, 17, 3): x, y, visibility
                dets = []
                for j in range(len(scores)):
                    if int(labels[j].item()) != 0:  # person only
                        continue
                    x1, y1, x2, y2 = boxes[j].tolist()
                    kpts = []
                    for x, y, v in poses[j].tolist():
                        kpts.extend([x, y, 2 if v > 0.5 else 1])
                    dets.append(
                        {
                            "image_id": image_id,
                            "category_id": 1,
                            "keypoints": kpts,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(scores[j]),
                        }
                    )
                self.predictions["keypoints"][image_id] = dets

    def synchronize_between_processes(self) -> None:
        """Merge the predictions of all ranks (padded duplicates carry identical entries)."""
        for iou_type in self.iou_types:
            merged = {}
            for part in dist.all_gather(self.predictions[iou_type]):
                merged.update(part)
            self.predictions[iou_type] = merged

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for iou_type in self.iou_types:
            names = _STAT_NAMES[iou_type]
            preds = self.predictions[iou_type]
            results = [d for image_id in sorted(preds) for d in preds[image_id]]
            if not results:
                stats[iou_type] = {f"{iou_type}_{n}": 0.0 for n in names}
                continue

            coco_gt = self.coco_gt[iou_type]
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                coco_dt = coco_gt.loadRes(results)
            coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
            coco_eval.evaluate()
            coco_eval.accumulate()
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()
            stats[iou_type] = {f"{iou_type}_{n}": float(v) for n, v in zip(names, coco_eval.stats)}
        return stats
