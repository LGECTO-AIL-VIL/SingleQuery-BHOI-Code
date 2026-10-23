from typing import Any, List, Tuple

import numpy as np
import torch
import torchvision


class oih_data_loader:
    def __init__(self, oih_cls: int, oih_contact: bool):
        self.oih_cls = oih_cls
        self.oih_contact = oih_contact
        self.oih_contact_cls = {
            "no_contact": 0,
            "object_contact": 1,
            "self_contact": 2,
            "other_person_contact": 3,
        }
        if self.oih_cls == 2:
            self.oih_hand_state_cls = {
                "held": 1.0,
                "used": 1.0,
                "touched": 1.0,
                "hold": 1.0,
                "use": 1.0,
                "touch": 1.0,
                "null": 0.0,
            }
        elif self.oih_cls == 3:
            self.oih_hand_state_cls = {
                "held": 1.0,
                "used": 1.0,
                "hold": 1.0,
                "use": 1.0,
                "touched": 2.0,
                "touch": 2.0,
                "null": 0.0,
            }
        # no_contact -> off 0.0, valid
        # (object) & (not null) -> on 1.0, valid
        # (self, other_person contact) & (null -> gt always none) -> on 1.0, valid
        self.null_oihs = [0.0, 0.0, 1.0, 1.0]

    def load(
        self,
        ann: dict,
        person_id: int = 0,
        right_wrist_idx: int = 10,
        left_wrist_idx: int = 9,
    ) -> Tuple[List, List, List, List, List, List]:
        # ['None', 'container_,_held', 'container_,_touched', 'neither_,_held', 'neither_,_touched',
        #  'tool_,_held', 'tool_,_touched', 'tool_,_used']
        oih_ins = []
        oih_cont_ins = []
        valid_oih_ins = []
        oih_box = []
        oih_obj_box = []
        oih_ids = []

        def _check_oih_anno(pos: str) -> None:
            hand_contact_str = "".join([pos, "_hand_contact"])
            hand_oih_box_str = "".join([pos, "_hand_oih_box"])
            hand_state_str = "".join([pos, "_hand_state"])
            hand_obj_box_str = "".join([pos, "_hand_obj_box"])
            hand_id_str = "".join([pos, "_obj_id"])

            if hand_contact_str in ann and ann[hand_contact_str] != "inconclusive":
                # TODO : support multilabel contact class(Contact Hands have multilabel)
                if isinstance(ann[hand_contact_str], list):
                    ann[hand_contact_str] = ann[hand_contact_str][0]
                if ann[hand_contact_str] == "None" or ann[hand_contact_str] is None:
                    oih_box.append(self.null_oihs)
                    oih_obj_box.append(self.null_oihs)
                    oih_ins.append(self.oih_hand_state_cls["null"])
                    oih_cont_ins.append(self.oih_contact_cls["no_contact"])
                    valid_oih_ins.append(False)
                    oih_ids.append(-1)
                    return
                oih_cont = self.oih_contact_cls[ann[hand_contact_str]]
                oih_cont_ins.append(oih_cont)
                oih_box.append(ann[hand_oih_box_str])
                is_valid = True
                if oih_cont == 0:  # no contact
                    oih_ins.append(self.oih_hand_state_cls["null"])  # off state
                    oih_obj_box.append(self.null_oihs)
                    oih_ids.append(-1)
                elif oih_cont == 1:  # object contact
                    if "_" in ann[hand_state_str]:
                        hand_state_key = ann[hand_state_str].split("_")[-1]
                    else:
                        hand_state_key = ann[hand_state_str]
                    hand_state = self.oih_hand_state_cls.get(hand_state_key, None)

                    if hand_state is not None:  # object contact with valid hand state
                        oih_ins.append(hand_state)
                        # contact_hands doesn't have obj box
                        if ann[hand_obj_box_str] == "None":
                            oih_obj_box.append(self.null_oihs)
                        else:
                            oih_obj_box.append(ann[hand_obj_box_str])
                    else:  # object contact with "None" hand state
                        oih_ins.append(self.oih_hand_state_cls["null"])
                        oih_obj_box.append(self.null_oihs)
                        is_valid = False
                    oih_ids.append(ann.get(hand_id_str, -1))
                else:  # self or other person contact
                    box_state = ann[hand_obj_box_str]
                    if box_state == "None":
                        oih_obj_box.append(self.null_oihs)
                        oih_ins.append(self.oih_hand_state_cls["null"])  # off state
                        is_valid = False
                    else:
                        oih_ins.append(self.oih_hand_state_cls["held"])
                        oih_obj_box.append(ann[hand_obj_box_str])
                    oih_ids.append(ann.get(hand_id_str, -1))
                valid_oih_ins.append(is_valid)
            else:
                oih_box.append(self.null_oihs)
                oih_obj_box.append(self.null_oihs)
                oih_ins.append(self.oih_hand_state_cls["null"])
                oih_cont_ins.append(self.oih_contact_cls["no_contact"])
                valid_oih_ins.append(False)
                oih_ids.append(-1)

        _check_oih_anno("right")
        _check_oih_anno("left")

        body_key_points = ann.get("keypoints")
        if body_key_points is not None and ann["category_id"] == person_id:
            body_key_points = np.asarray(body_key_points).reshape(-1, 3)
            oih_vis = [
                body_key_points[right_wrist_idx][-1].astype(np.float32),
                body_key_points[left_wrist_idx][-1].astype(np.float32),
            ]
        else:
            oih_vis = np.array([0.0, 0.0], dtype=np.float32)

        return oih_ins, oih_box, oih_obj_box, oih_cont_ins, valid_oih_ins, oih_vis, oih_ids


def oih_flip(target: dict, width: int) -> dict:
    if "oih" in target:
        for oih in target["oih"]:
            new_oih = oih[0].clone()
            oih[0] = oih[1]
            oih[1] = new_oih

    if "oih_cont" in target:
        for oih_cont in target["oih_cont"]:
            new_oih_cont = oih_cont[0].clone()
            oih_cont[0] = oih_cont[1]
            oih_cont[1] = new_oih_cont

    if "oih_ids" in target:
        for oih_id in target["oih_ids"]:
            new_oih_id = oih_id[0].clone()
            oih_id[0] = oih_id[1]
            oih_id[1] = new_oih_id

    if "valid_oih" in target:
        for oih_val in target["valid_oih"]:
            new_oih_val = oih_val[0].clone()
            oih_val[0] = oih_val[1]
            oih_val[1] = new_oih_val

    if "oih_boxes" in target:
        for boxes in target["oih_boxes"]:
            new_x1 = width - boxes[0][2]
            new_x2 = width - boxes[0][0]
            boxes[0][0] = new_x1
            boxes[0][2] = new_x2
            new_x1_ = width - boxes[1][2]
            new_x2_ = width - boxes[1][0]
            boxes[1][0] = new_x1_
            boxes[1][2] = new_x2_
            new_boxes1 = boxes[0].clone()
            boxes[0] = boxes[1]
            boxes[1] = new_boxes1

    if "oih_obj_boxes" in target:
        for boxes in target["oih_obj_boxes"]:
            new_x1 = width - boxes[0][2]
            new_x2 = width - boxes[0][0]
            boxes[0][0] = new_x1
            boxes[0][2] = new_x2
            new_x1_ = width - boxes[1][2]
            new_x2_ = width - boxes[1][0]
            boxes[1][0] = new_x1_
            boxes[1][2] = new_x2_
            new_boxes1 = boxes[0].clone()
            boxes[0] = boxes[1]
            boxes[1] = new_boxes1

    return target


def oih_perspective(
    targets: dict,
    i: list,
    M: np.array,
    size: Tuple[int, int],
    scale_factor: float,
    perspective: float,
    norm: bool,
    box_candidates: Any,
    box_norm: Any,
    use_segments: bool = False,
) -> dict:
    width, height = size

    targets["oih_boxes"] = targets["oih_boxes"][i]
    targets["oih"] = targets["oih"][i]
    targets["oih_cont"] = targets["oih_cont"][i]
    targets["valid_oih"] = targets["valid_oih"][i]
    targets["oih_ids"] = targets["oih_ids"][i]

    # filter oih ids
    oih_ids = targets["oih_ids"]
    for k in range(oih_ids.shape[0]):
        for j in range(oih_ids.shape[1]):
            val = oih_ids[k, j]
            if val != -1 and val not in targets["ids"]:
                oih_ids[k, j] = -1
    targets["oih_ids"] = oih_ids

    # warp oih boxes
    new_n = len(targets["labels"])
    oih_boxes = targets["oih_boxes"].reshape(new_n * 2, 4)
    new_oih_boxes = np.zeros((new_n * 2, 4))
    xy = np.ones((new_n * 2 * 4, 3))
    xy[:, :2] = (
        oih_boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(new_n * 2 * 4, 2).numpy()
    )  # x1y1, x2y2, x1y2, x2y1
    xy = xy @ M.T  # transform
    xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(
        new_n * 2, 8
    )  # perspective rescale or affine

    # create new boxes
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    new_oih_boxes = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, new_n * 2).T

    # clip
    new_oih_boxes[:, [0, 2]] = new_oih_boxes[:, [0, 2]].clip(0, width)
    new_oih_boxes[:, [1, 3]] = new_oih_boxes[:, [1, 3]].clip(0, height)

    # filter candidates
    oih_i = box_candidates(
        box1=oih_boxes.numpy().T * scale_factor,
        box2=new_oih_boxes.T,
        area_thr=0.01 if use_segments else 0.10,
    )
    oih_i = np.reshape(oih_i, (-1, 2))
    if norm:
        new_oih_boxes = box_norm(new_oih_boxes, (width, height))
    targets["oih_boxes"] = torch.tensor(new_oih_boxes.reshape(new_n, 2, 4), dtype=torch.float)
    targets["valid_oih"] = targets["valid_oih"] & oih_i

    # warp oih obj boxes
    targets["oih_obj_boxes"] = targets["oih_obj_boxes"][i]
    new_n = len(targets["labels"])
    oih_obj_boxes = targets["oih_obj_boxes"].reshape(new_n * 2, 4)
    new_oih_obj_boxes = np.zeros((new_n * 2, 4))
    xy = np.ones((new_n * 2 * 4, 3))
    xy[:, :2] = (
        oih_obj_boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(new_n * 2 * 4, 2).numpy()
    )  # x1y1, x2y2, x1y2, x2y1
    xy = xy @ M.T  # transform
    xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(
        new_n * 2, 8
    )  # perspective rescale or affine

    # create new boxes
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    new_oih_obj_boxes = (
        np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, new_n * 2).T
    )

    # clip
    new_oih_obj_boxes[:, [0, 2]] = new_oih_obj_boxes[:, [0, 2]].clip(0, width)
    new_oih_obj_boxes[:, [1, 3]] = new_oih_obj_boxes[:, [1, 3]].clip(0, height)
    if norm:
        new_oih_obj_boxes = box_norm(new_oih_obj_boxes, (width, height))
    targets["oih_obj_boxes"] = torch.tensor(
        new_oih_obj_boxes.reshape(new_n, 2, 4), dtype=torch.float
    )

    return targets


def oih_box_convert(targets: dict, in_fmt: str, out_fmt: str) -> dict:
    if "oih_boxes" in targets:
        oih_boxes = torchvision.ops.box_convert(
            targets["oih_boxes"].reshape(-1, 4), in_fmt=in_fmt, out_fmt=out_fmt
        )
        targets["oih_boxes"] = oih_boxes.reshape(-1, 2, 4)
    if "oih_obj_boxes" in targets:
        oih_obj_boxes = torchvision.ops.box_convert(
            targets["oih_obj_boxes"].reshape(-1, 4), in_fmt=in_fmt, out_fmt=out_fmt
        )
        targets["oih_obj_boxes"] = oih_obj_boxes.reshape(-1, 2, 4)
    return targets
