import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou

from src.core import register
from src.misc import dist

from .convert import convert_gt_to_coco_dict, cpu_tolist
from .oih_utils import OiHStruct

__all__ = ["OiHEvaluator"]


@register
class OiHEvaluator(object):
    """
    hand box is oracle or e2e mode
    1. Hard localize TP = Hand Object TP | Hand TP | person TP
    2. Medium localize TP = Hand TP | person TP
    3. soft localize TP = person AP
    4.

    input box format = xyxy
    """

    __share__ = ["category_ver", "oih_config"]

    def __init__(
        self,
        category_ver: str = "1.0",
        oih_config: Optional[Dict] = None,
        person_box_oracle: bool = True,
        use_valid_gt_only: bool = False,
        eval_hand_obj: bool = False,
        ignore_small: bool = False,
    ) -> None:
        self.metadata = None
        assert oih_config is not None, 'need "oih_config"'
        self.oih_config = oih_config
        for k, v in oih_config.items():
            setattr(self, k, v)
        self.all_gt_items = []
        self.pred_results = []
        self.ignore_small = ignore_small
        # self.person_cls = self.metadata.contiguous_id_to_dataset_id[
        #     self.metadata.stuff_classes.index("person")
        # ]
        self.person_cls = 0  # TODO read from metadata
        self.box_confidence_threshold = 0.7
        self.min_box_topK = 10
        self.iou_threshold = 0.5
        self.soft_correct = 0
        self.soft_tot = 0

        self.person_box_oracle = person_box_oracle
        self._match = _hungarian_match
        self.person_iou_thr = 0.5
        # oracle: use matched person(between gt and pred)
        # e2e: use all pred person
        self.eval_mode = "oracle" if person_box_oracle else "e2e"
        # Only use those with at least one valid GT(reduce FP)
        self.valid_gt_only = use_valid_gt_only

        # eval hand obj box(hard tp)
        # only work on/off now(Contact Hand doesn't have object)
        self.eval_hand_obj = eval_hand_obj

    def reset(self) -> None:
        self.pred_results = []
        self.all_gt_items = []
        self.soft_tot = 0
        self.soft_correct = 0

    def process(self, inputs: List, outputs: List) -> None:
        preds = {target["image_id"]: [output, target] for target, output in zip(inputs, outputs)}
        self.update(preds)

    def evaluate(self) -> Dict:
        dict_coco_gt, image_id_map = convert_gt_to_coco_dict(self.all_gt_items)
        for i in range(len(self.pred_results)):
            new_id = image_id_map.get(self.pred_results[i]["image_id"])
            assert new_id, "OIH EVAL: mismatched between pred img and gt"
            self.pred_results[i]["image_id"] = new_id

        coco_gt = COCO()
        coco_gt.dataset = dict_coco_gt
        coco_gt.createIndex()
        # Make 'info' key in dataset(pycocotools copy 'info' from dataset at pycocotools >=2.0.10)
        coco_gt.dataset.setdefault("info", {})
        # there isn't any matched box -> return zeros
        if len(self.pred_results) == 0:
            return {
                "oih": {
                    "Acc|TP_Hard(person+hand+cls+obj. IoU0.5)": 0.0,
                    "Acc|TP_Medium(person+hand+cls. IoU0.5)": 0.0,
                    "Acc|TP_Soft(IoU0.5)": 0.0,
                    "correct": 0.0,  # medium
                    "correct_hard": 0.0,
                    "correct_medium(O)": 0,
                    "ACC | TP_Medium(O)": 0.0,
                }
            }
        ## soft, state_correct_with_obj(medium2)
        # BC: remove duplicated when DDP
        soft_tot = 0
        soft_correct = 0
        state_correct_with_obj = 0
        # for ins in coco_gt.anns.values():
        for img_id in coco_gt.getImgIds():
            ann_ids = coco_gt.getAnnIds(img_id)
            if len(ann_ids) == 0:
                # no anns in img
                continue
            ins = coco_gt.loadAnns(ann_ids[0])[0]
            soft_tot += ins.get("soft_tot", 0)
            soft_correct += ins.get("soft_correct", 0)
            if self.eval_hand_obj:
                state_correct_with_obj += ins.get("state_correct_with_obj", 0)

        coco_dt = coco_gt.loadRes(self.pred_results)
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.params.areaRngLbl = ["all"]
        evaluator.params.areaRng = [
            evaluator.params.areaRng[0],
        ]
        evaluator.evaluate()
        mid_acc, mid_correct, hard_acc, hard_correct = self.compute_state_cls_accuracy(
            evaluator, iou_threshold=self.iou_threshold
        )
        evaluator.accumulate()
        evaluator.summarize()
        soft_acc = soft_correct / soft_tot if soft_tot > 0 else 0.0
        print(f"Acc|TP soft {soft_acc:.4f}")

        result = {
            "oih": {
                "Acc|TP_Hard(person+hand+cls+obj. IoU0.5)": hard_acc,
                "Acc|TP_Medium(person+hand+cls. IoU0.5)": mid_acc,
                "Acc|TP_Soft(IoU0.5)": soft_acc,
                "correct": mid_correct,  # medium
                "correct_hard": hard_correct,
                "correct_medium(O)": state_correct_with_obj,
                "ACC | TP_Medium(O)": state_correct_with_obj / soft_tot if soft_tot > 0 else 0.0,
            }
        }

        # Report by person count per image
        person_count_report = self._report_by_person_count(coco_gt, coco_dt)
        result["oih_by_person_count"] = person_count_report

        return result

    def _report_by_person_count(self, coco_gt: COCO, coco_dt: COCO) -> Dict:
        """Report OiH metrics grouped by person count per image.
        Groups: 1 person, 2~4 persons, >=5 persons.
        """
        # Build image_id -> person count mapping from all_gt_items (detection GT based)
        # same image_id may appear multiple times (DDP duplicates), use first occurrence only
        img_person_count = {}
        for gt_item in self.all_gt_items:
            img_id = gt_item["image_id"]
            if img_id in img_person_count:
                continue
            img_person_count[img_id] = gt_item["n_persons"]

        # Map original image_id -> coco int image_id
        orig_to_coco_id = {}
        for img_info in coco_gt.dataset["images"]:
            orig_to_coco_id[img_info["file_name"]] = img_info["id"]

        # Group coco image ids by person count
        groups = {"1": [], "2~4": [], ">=5": []}
        for orig_id, count in img_person_count.items():
            coco_id = orig_to_coco_id.get(orig_id)
            if coco_id is None:
                continue
            if count == 1:
                groups["1"].append(coco_id)
            elif count <= 4:
                groups["2~4"].append(coco_id)
            else:
                groups[">=5"].append(coco_id)

        report = {}
        for group_name, img_ids in groups.items():
            if len(img_ids) == 0:
                report[group_name] = {
                    "n_images": 0,
                    "Acc|TP_Hard": 0.0,
                    "Acc|TP_Medium": 0.0,
                    "Acc|TP_Soft": 0.0,
                    "ACC|TP_Medium(O)": 0.0,
                }
                continue

            # Soft accuracy & state_correct_with_obj for this group
            g_soft_tot = 0
            g_soft_correct = 0
            g_state_correct_with_obj = 0
            for img_id in img_ids:
                ann_ids = coco_gt.getAnnIds(img_id)
                if len(ann_ids) == 0:
                    continue
                ins = coco_gt.loadAnns(ann_ids[0])[0]
                g_soft_tot += ins.get("soft_tot", 0)
                g_soft_correct += ins.get("soft_correct", 0)
                if self.eval_hand_obj:
                    g_state_correct_with_obj += ins.get("state_correct_with_obj", 0)

            # Medium/Hard accuracy: run COCOeval on subset
            img_id_set = set(img_ids)
            group_preds = [p for p in self.pred_results if p["image_id"] in img_id_set]

            g_mid_acc = 0.0
            g_hard_acc = 0.0
            if len(group_preds) > 0:
                try:
                    coco_dt_group = coco_gt.loadRes(group_preds)
                    eval_group = COCOeval(coco_gt, coco_dt_group, iouType="bbox")
                    eval_group.params.imgIds = img_ids
                    eval_group.params.areaRngLbl = ["all"]
                    eval_group.params.areaRng = [eval_group.params.areaRng[0]]
                    eval_group.evaluate()
                    g_mid_acc, _, g_hard_acc, _ = self.compute_state_cls_accuracy(
                        eval_group, iou_threshold=self.iou_threshold
                    )
                except Exception:
                    pass

            g_soft_acc = g_soft_correct / g_soft_tot if g_soft_tot > 0 else 0.0
            g_medium_o = g_state_correct_with_obj / g_soft_tot if g_soft_tot > 0 else 0.0
            report[group_name] = {
                "n_images": len(img_ids),
                "Acc|TP_Hard": g_hard_acc,
                "Acc|TP_Medium": g_mid_acc,
                "Acc|TP_Soft": g_soft_acc,
                "ACC|TP_Medium(O)": g_medium_o,
            }
            print(
                f"[Person count: {group_name}] n_imgs={len(img_ids)}, "
                f"Hard={g_hard_acc:.4f}, Medium={g_mid_acc:.4f}, "
                f"Soft={g_soft_acc:.4f}, Medium(O)={g_medium_o:.4f}"
            )

        return report

    def update(self, predictions: Dict) -> None:
        for i, (img_id, [pred, target]) in enumerate(predictions.items()):
            oih_pred = OiHStruct(pred["oihs"], **self.oih_config)
            # save target
            gt_cls_mask = target["labels"] == self.person_cls
            # size mask
            gt_size_mask = gt_cls_mask
            if self.ignore_small:
                if len(target["oih_boxes"]) != 0:
                    hand_box_sizes = torch.stack(
                        [
                            torch.stack(
                                [(x[3] - x[1]) * (x[2] - x[0]), (y[3] - y[1]) * (y[2] - y[0])],
                                dim=-1,
                            )
                            for x, y in target["oih_boxes"]
                        ]
                    )

                    area_ratio = 640 / target["size"][0] * 640 / target["size"][1]
                    gt_size_mask = (hand_box_sizes * area_ratio > 32 * 32).sum(-1, dtype=torch.bool)
            gt_cls_mask = gt_cls_mask & gt_size_mask
            # ~ size mask
            gt_hand_box = target["oih_boxes"][gt_cls_mask]
            gt_hand_obj_box = target["oih_obj_boxes"][gt_cls_mask]  # xyxy

            self.all_gt_items.append(
                {
                    "image_id": target["image_id"],
                    "img_size": target["size"],
                    "hand_box": gt_hand_box,  # xyxy.
                    "hand_valid": target["valid_oih"][gt_cls_mask],
                    # "contact_cls": target["oih_cont"][gt_cls_mask],
                    "state_cls": target["oih"][gt_cls_mask],
                    "hand_obj_box": gt_hand_obj_box,
                    "n_persons": int((target["labels"] == self.person_cls).sum()),
                }
            )

            # match person
            pred_cls_mask = pred["labels"] == self.person_cls
            gt_boxes = target["boxes"][gt_cls_mask]  # xyxy
            dt_boxes = pred["boxes"][pred_cls_mask]  # xyxy
            if len(gt_boxes) and len(dt_boxes):
                pairs, un_p, _un_g = self._match(gt_boxes, dt_boxes, self.person_iou_thr)
            else:
                pairs, un_p, _un_g = [], list(range(len(dt_boxes))), list(range(len(gt_boxes)))
            fn = 0

            if len(pairs):
                pairs_arr = np.asarray(pairs)
                dt_p = pairs_arr[:, -1]
                gt_p = pairs_arr[:, 0]
                mask = target["valid_oih"][gt_cls_mask][gt_p]  # OIH valid mask
                valid_gt_only_mask = mask.any(axis=-1)  # have least 1 valid idx
                self._add_pred_boxes(
                    "pred_results",
                    img_id,
                    pred["scores"],
                    oih_pred,
                    pred_cls_mask,
                    dt_p if not self.valid_gt_only else dt_p[valid_gt_only_mask.to("cpu").numpy()],
                )
                # calculate acc | TP soft(person)
                # valid gt hands only
                soft_tot = target["valid_oih"][gt_cls_mask][gt_p].sum()
                rt, lt = target["oih"][gt_cls_mask][gt_p].unbind(-1)
                rp = oih_pred["right_hand_cls"][pred_cls_mask][dt_p][:, -1]
                lp = oih_pred["left_hand_cls"][pred_cls_mask][dt_p][:, -1]

                soft_correct = ((rt == rp) & mask[:, 0]).sum()
                soft_correct += ((lt == lp) & mask[:, 1]).sum()

                on_states_r = (rp == 1) & (rt == rp) & mask[:, 0]
                r_gt_obj_box = target["oih_obj_boxes"][gt_cls_mask][gt_p][on_states_r, 0]
                r_pred_obj_box = oih_pred["right_hand_obj_box"][pred_cls_mask][dt_p][on_states_r]

                on_states_l = (lp == 1) & (lt == lp) & mask[:, 1]
                l_gt_obj_box = target["oih_obj_boxes"][gt_cls_mask][gt_p][on_states_l, 1]
                l_pred_obj_box = oih_pred["left_hand_obj_box"][pred_cls_mask][dt_p][on_states_l]

                r_ious = ((rp == 0) & (rt == rp) & mask[:, 0]).sum()  # off state
                if len(r_gt_obj_box) != 0 and len(r_pred_obj_box) != 0:
                    r_ious += (box_iou(r_pred_obj_box, r_gt_obj_box).diagonal() > 0.5).sum()

                l_ious = ((lp == 0) & (lt == lp) & mask[:, 1]).sum()  # off state
                if len(l_gt_obj_box) != 0 and len(l_pred_obj_box) != 0:
                    l_ious = (box_iou(l_pred_obj_box, l_gt_obj_box).diagonal() > 0.5).sum()

                state_correct = r_ious + l_ious
                # add data
                self.all_gt_items[-1]["soft_tot"] = int((soft_tot + fn).cpu())
                self.all_gt_items[-1]["soft_correct"] = int(soft_correct.cpu())
                self.all_gt_items[-1]["state_correct_with_obj"] = int(state_correct.cpu())

            if self.eval_mode != "oracle":
                self._add_pred_boxes(
                    "pred_results",
                    img_id,
                    pred["scores"],
                    oih_pred,
                    pred_cls_mask,
                    np.asarray(un_p),
                )

    def _add_pred_boxes(
        self,
        attr_name: str,
        img_id: str,
        scores: Union[List, torch.Tensor, np.ndarray],
        oih_pred: OiHStruct,
        pred_cls_mask: Union[List, torch.Tensor, np.ndarray],
        idx: Union[List, np.ndarray],
    ) -> None:
        r_hand_boxes = convert_to_xywh(oih_pred["right_hand_box"][pred_cls_mask][idx])
        l_hand_boxes = convert_to_xywh(oih_pred["left_hand_box"][pred_cls_mask][idx])
        dest = getattr(self, attr_name)

        (r_hand_boxes, l_hand_boxes, scores, r_oih, l_oih) = map(
            cpu_tolist,
            [
                r_hand_boxes,
                l_hand_boxes,
                scores[pred_cls_mask],
                oih_pred["right_hand_cls"][pred_cls_mask],
                oih_pred["left_hand_cls"][pred_cls_mask],
            ],
        )
        (right_hand_obj_box, left_hand_obj_box) = map(
            cpu_tolist,
            [
                oih_pred["right_hand_obj_box"][pred_cls_mask],
                oih_pred["left_hand_obj_box"][pred_cls_mask],
            ],
        )

        r_vis = None
        l_vis = None
        if self.oih_config.get("oih_vis", False):
            r_vis = oih_pred["right_hand_vis"][pred_cls_mask][idx].sigmoid()
            l_vis = oih_pred["left_hand_vis"][pred_cls_mask][idx].sigmoid()
        preds = []
        for i, (rhand_box, lhand_box, score) in enumerate(zip(r_hand_boxes, l_hand_boxes, scores)):
            if score < self.box_confidence_threshold and i > self.min_box_topK:
                break
            if r_vis is None or r_vis[i] > 0.5:
                preds.append(
                    {
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": rhand_box,
                        "score": score,
                        "state_cls": r_oih[i],
                        "obj_box": right_hand_obj_box[i],
                    }
                )
            if l_vis is None or l_vis[i] > 0.5:
                preds.append(
                    {
                        "image_id": img_id,
                        "category_id": 2,
                        "bbox": lhand_box,
                        "score": score,
                        "state_cls": l_oih[i],
                        "obj_box": left_hand_obj_box[i],
                    }
                )
        if len(preds) != 0:
            dest.append(preds)

    def synchronize_between_processes(self) -> None:
        all_gther = dist.all_gather(self.all_gt_items)
        self.all_gt_items = np.concatenate(all_gther, 0).tolist()
        all_gther = dist.all_gather(self.pred_results)
        self.pred_results = [x for block in all_gther for x in block]  # images[predicts]

        # Remove duplicated in pred
        seen = set()
        new_pred_results = []
        for i in range(len(self.pred_results)):
            if len(self.pred_results[i]) == 0:
                continue
            img_id = self.pred_results[i][0]["image_id"]
            if img_id in seen:
                continue
            else:
                seen.add(img_id)
                for j in range(len(self.pred_results[i])):
                    new_pred_results.append(self.pred_results[i][j])
        self.pred_results = new_pred_results
        # ~ Remove duplicated in pred
        # if isinstance(self.soft_correct, torch.Tensor):
        #     self.soft_correct = self.soft_correct.cpu()
        # if isinstance(self.soft_tot, torch.Tensor):
        #     self.soft_tot = self.soft_tot.cpu()
        # self.soft_correct = np.asarray(dist.all_gather(self.soft_correct)).sum()
        # self.soft_tot = np.asarray(dist.all_gather(self.soft_tot)).sum()

    def accumulate(self) -> None:
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self) -> None:
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def compute_state_cls_accuracy(self, coco_eval: COCOeval, iou_threshold: float = 0.5) -> Tuple:
        start_time = time.time()
        # IoU threshold index
        assert iou_threshold in coco_eval.params.iouThrs, (
            f"OiH Acc compute iou support {coco_eval.params.iouThrs}"
        )
        iou_thr_index = np.where(np.isclose(coco_eval.params.iouThrs, iou_threshold))[0][0]

        medium_correct = 0
        hard_correct = 0
        total = 0

        pred_by_image = {}
        for p in self.pred_results:
            pred_by_image.setdefault(p["image_id"], []).append(p)

        for eval_dict in coco_eval.evalImgs:
            if eval_dict is None:
                continue

            image_id = eval_dict["image_id"]
            dt_matches = eval_dict["dtMatches"][iou_thr_index]
            dt_ignore = eval_dict["dtIgnore"][iou_thr_index]

            gtMatches = eval_dict["gtMatches"]
            gtIgnore = eval_dict["gtIgnore"]
            fn = int(np.sum((gtMatches[iou_thr_index, :] == 0) & (gtIgnore == 0)))
            # print(f"Unmatched between pred and GT: {fn}. Add to FN")
            total = total + fn  # penalty for unmatched gt

            preds = pred_by_image.get(image_id, [])

            # TODO too complex
            for det_idx, gt_id in enumerate(dt_matches):
                if gt_id == 0 or dt_ignore[det_idx] == 1:
                    continue  # not matched or ignored

                if det_idx >= len(preds):
                    continue

                pred_state = int(preds[det_idx].get("state_cls")[-1])
                gt_state = int(coco_eval.cocoGt.anns[gt_id].get("state_cls"))

                if pred_state == gt_state:
                    medium_correct += 1
                    # on state.(obj box iou check)
                    if pred_state == 1 and self.eval_hand_obj:
                        iou = box_iou(
                            torch.as_tensor(preds[det_idx].get("obj_box")).reshape(1, -1),
                            torch.as_tensor(
                                coco_eval.cocoGt.anns[gt_id].get("hand_obj_box")
                            ).reshape(1, -1),
                        )[0]
                        if iou >= iou_threshold:
                            hard_correct += 1
                    elif self.eval_hand_obj:
                        # off state
                        hard_correct += 1
                total += 1
            # ~TODO
        print(f"OiH Compute time {time.time() - start_time}")
        mid_acc = medium_correct / total if total > 0 else 0.0
        hard_acc = hard_correct / total if total > 0 else 0.0
        print(f"State classification accuracy (IoU >= {iou_threshold}): {mid_acc:.4f}. ")
        print(
            f"joint(hand box + hand cls + obj box) accuracy: {hard_acc:.4f}."
            f"oih_tot_matched: {total}, "
            f"oih_medium_correct_matched: {medium_correct}, "
            f"oih_hard correct_matched: {hard_correct}"
        )
        return mid_acc, medium_correct, hard_acc, hard_correct


def convert_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def _hungarian_match(gt: np.ndarray, dt: np.ndarray, thr: float) -> Tuple:
    """
    gt and dt box format : xyxy
    """
    if len(gt) == 0 or len(dt) == 0:
        return [], list(range(len(dt))), list(range(len(gt)))
    iou = box_iou(torch.as_tensor(gt), torch.as_tensor(dt)).cpu().numpy()
    cost = 1.0 - iou
    cost[iou < thr] = 1e6
    gi, pj = linear_sum_assignment(cost)
    keep = cost[gi, pj] < 1e6
    gi, pj = gi[keep], pj[keep]
    used_g = np.zeros(len(gt), bool)
    used_p = np.zeros(len(dt), bool)
    for g, p in zip(gi, pj):
        used_g[g] = True
        used_p[p] = True
    un_p = [i for i in range(len(dt)) if not used_p[i]]
    un_g = [i for i in range(len(gt)) if not used_g[i]]
    return list(zip(gi.tolist(), pj.tolist())), un_p, un_g
