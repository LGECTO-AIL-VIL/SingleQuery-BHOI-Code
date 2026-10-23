"""COCO(-WholeBody) dataset with body keypoints and hand-object (OiH) annotations.

The annotation file keeps the segments of image `i` in `annotations[i]["segments_info"]`
(class id, xywh box, area, 17 body keypoints and
the `{right,left}_hand_*` OiH fields).
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.core import get_metadata, register

from ..oih.oih_data_utils import oih_data_loader

__all__ = ["CocoOiHDetection"]


@register
class CocoOiHDetection:
    __inject__ = ["transforms", "evaluators"]
    __share__ = ["num_kpts", "category_ver", "oih_config"]

    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        transforms=None,
        evaluators=None,
        dataset_name: str = "COCO",
        num_kpts: int = 17,
        category_ver: str = "1.9",
        oih_config: Optional[Dict] = None,
    ) -> None:
        with open(ann_file) as f:
            coco = json.load(f)

        # align "images" and "annotations" (both sorted by image id)
        images = sorted(coco["images"], key=lambda x: x["id"])
        self.annotations = sorted(coco["annotations"], key=lambda x: x["image_id"])
        for img, ann in zip(images, self.annotations):
            assert img["file_name"][:-4] == ann["file_name"][:-4], (img, ann)

        self.img_folder = img_folder
        self._transforms = transforms
        self.evaluators = evaluators
        self.dataset_name = dataset_name
        self.num_kpts = num_kpts
        self.label_mapper = get_metadata(category_ver).thing_dataset_id_to_contiguous_id
        oih_config = oih_config or {}
        self.oih_data_loader = oih_data_loader(
            oih_config.get("oih_cls", 2), oih_config.get("oih_contact", False)
        )
        self.null_kpts = [0.0] * (num_kpts * 3)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, dict]:
        img, target = self.load(self.annotations[idx])
        target["epoch"] = self.epoch

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        target["dataset"] = self.dataset_name
        return img, target

    def load(self, ann_info: dict) -> Tuple[Image.Image, dict]:
        """Read the image and build the (un-augmented) target of one annotation entry."""
        img_path = os.path.join(
            self.img_folder, os.path.splitext(ann_info["file_name"])[0] + ".jpg"
        )
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # keep only segments of known categories
        segments: List[dict] = []
        for seg in ann_info["segments_info"]:
            seg["category_id"] = int(seg["category_id"])
            if seg["category_id"] in self.label_mapper:
                segments.append(seg)

        labels = torch.tensor(
            [self.label_mapper[s["category_id"]] for s in segments], dtype=torch.int64
        )
        boxes = torch.tensor([s["bbox"] for s in segments])  # xywh
        if len(boxes) > 0:
            boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]  # xyxy
        else:
            boxes = torch.zeros((0, 4))
        ids = torch.tensor([s["id"] for s in segments])
        area = torch.tensor(
            [s["area"] if "area" in s else s["bbox"][2] * s["bbox"][3] for s in segments]
        )

        target = {
            "image_id": ann_info["image_id"] if "image_id" in ann_info else ann_info["id"],
            "file_name": ann_info["file_name"],
            "size": torch.as_tensor([w, h]),
            "orig_size": torch.as_tensor([w, h]),
            "iscrowd": torch.tensor([s["iscrowd"] for s in segments if "iscrowd" in s]),
            "area": area,
            "face_area": area.clone(),
            "norm_area": area / (w * h),
            "boxes": boxes,
            "labels": labels,
            "ids": ids,
        }
        cost = ["cls", "box"]

        # body keypoints (a segment with a "keypoints" field is a valid pose target) and OiH
        pose_2d, valid_person = [], []
        oih, oih_cont, valid_oih, oih_boxes, oih_obj_boxes, oih_vis, oih_ids = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for seg in segments:
            if "keypoints" in seg:
                pose_2d.append(
                    np.array(seg["keypoints"])
                    .reshape(self.num_kpts, 3)
                    .astype(np.float32)
                    .reshape(-1)
                )
                valid_person.append(True)
            else:
                pose_2d.append(self.null_kpts)
                valid_person.append(False)
            ins, box, obj_box, cont, valid, vis, oih_id = self.oih_data_loader.load(
                seg, person_id=1
            )
            oih.append(ins)
            oih_boxes.append(box)
            oih_obj_boxes.append(obj_box)
            oih_cont.append(cont)
            valid_oih.append(valid)
            oih_vis.append(vis)
            oih_ids.append(oih_id)

        pose_2d = torch.tensor(np.array(pose_2d), dtype=torch.float)
        pose_2d = pose_2d.reshape(pose_2d.shape[0], self.num_kpts, 3)
        if pose_2d.sum() != 0:
            cost.append("body")
        target["2d_pose"] = pose_2d
        target["valid_pose"] = torch.tensor(valid_person).bool()
        target["oih_boxes"] = torch.tensor(np.array(oih_boxes), dtype=torch.float)
        target["oih_obj_boxes"] = torch.tensor(np.array(oih_obj_boxes), dtype=torch.float)
        target["oih"] = torch.tensor(np.array(oih), dtype=torch.float)
        target["oih_cont"] = torch.tensor(np.array(oih_cont), dtype=torch.float)
        target["valid_oih"] = torch.tensor(valid_oih).bool()
        target["oih_vis"] = torch.tensor(np.array(oih_vis), dtype=torch.float)
        target["oih_ids"] = torch.tensor(oih_ids, dtype=torch.int)
        if target["valid_oih"].sum() != 0:
            cost.append("oih")
        target["cost"] = cost

        return img, target

    def get_evaluators(self):
        return self.evaluators

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
