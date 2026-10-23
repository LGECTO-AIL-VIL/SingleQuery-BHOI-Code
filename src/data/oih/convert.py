from typing import Dict, Tuple, Union

import numpy as np
import torch


def cpu_tolist(tensor: torch.Tensor) -> Union[torch.Tensor, object]:
    # return (lambda x: (x.cpu().tolist() if isinstance(x, torch.Tensor) else x))(tensor)
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().tolist()
    return tensor


def convert_gt_to_coco_dict(gt_data: Dict) -> Tuple[Dict, Dict]:
    """
    gt_data: list of dicts, each containing:
        - image_id (str)
        - handbox (Tensor or list) -> shape: [N, 4] in xyxy format
        - handvalid (Tensor or list of bools) -> shape: [N]

    image_wh_map: optional dict of image_name -> (width, height)
    img_size: image [w, h]
    """
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "rhand"},
            {"id": 2, "name": "lhand"},
        ],
    }
    image_id_map = {}
    img_id_counter = 1
    ann_id = 1

    for item in gt_data:
        filename = item["image_id"]
        boxes = item["hand_box"]
        valids = item["hand_valid"]

        extra_fields = {
            k: cpu_tolist(item[k])
            for k in item
            if k not in {"image_id", "hand_box", "hand_valid", "img_size", "n_persons"}
        }
        N = len(boxes)
        normalized_fields = {}
        for k, v in extra_fields.items():
            if isinstance(v, int) or len(v) == N or (N == 1 and len(np.array(v).shape) == 1):
                normalized_fields[k] = v
            elif len(np.array(v).shape) == 1 or len(v) == 1:
                normalized_fields[k] = [v] * N  # broadcast scalar to all boxes
            else:
                raise ValueError(f"Field '{k}' length mismatch: expected {N}, got {len(v)}")

        # Ensure list
        # tensor -> list
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().tolist()
        if isinstance(valids, torch.Tensor):
            valids = valids.cpu().tolist()
        # numpy -> list
        if hasattr(boxes, "tolist"):
            boxes = boxes.tolist()
        if hasattr(valids, "tolist"):
            valids = valids.tolist()

        # Assign numeric image_id
        # Avoid duplicated GT when DDP
        if filename not in image_id_map:
            image_id_map[filename] = img_id_counter
            width, height = item["img_size"]

            coco["images"].append(
                {
                    "id": img_id_counter,
                    "file_name": filename,
                    "width": width,
                    "height": height,
                }
            )
            img_id_counter += 1
            ###
            img_id = image_id_map[filename]

            # Add valid boxes
            for i, (box_rl, valid_rl) in enumerate(zip(boxes, valids)):
                # right, left
                for r, (box, valid) in enumerate(zip(box_rl, valid_rl)):
                    if not valid:
                        continue
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    # x1, y1, w, h = box
                    ann = {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": r + 1,  # right is 1, left is 2
                        "bbox": [x1, y1, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }

                    # Attach extra fields
                    for k, v in normalized_fields.items():
                        if isinstance(v, int):
                            ann[k] = v
                        else:
                            ann[k] = v[i][r]

                    coco["annotations"].append(ann)
                    ann_id += 1

    return coco, image_id_map
