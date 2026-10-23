"""
SQ-BHOI evaluation: detection (COCO AP), body pose (COCO keypoints AP) and
OiH (ContactHands accuracy). Uses the same validation datasets / evaluators as `tools/train.py`.

Usage (from any directory):
    python3 tools/eval.py --config configs/sqbhoi_r50vd.yml --checkpoint weights/checkpoint.pth

Dataset paths default to <repo>/datasets/{mscoco,ContactHands}; override with
--coco-val-img, --coco-val-ann, --coco-kpt-ann, --contacthands-img, --contacthands-gt.
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core import YAMLConfig
from src.data import CocoEvaluator, CocoValDetection, ContactHands, OiHEvaluator, default_collate_fn

# ============================================================================
# Model Loading
# ============================================================================


def load_model(config_path, checkpoint_path):
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    model = cfg.model.to("cuda")
    postprocessor = cfg.postprocessor.to("cuda")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["ema"]["module"])
    model.eval()
    postprocessor.eval()
    return model, postprocessor


# ============================================================================
# Evaluation
# ============================================================================


def eval_transform():
    return T.Compose(
        [
            T.Resize(size=[640, 640]),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


@torch.no_grad()
def run_evaluators(model, postprocessor, dataset, batch_size, desc):
    """Run `dataset` through the model and return the merged results of its evaluators."""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=default_collate_fn,
    )
    evaluators = dataset.get_evaluators()
    for evaluator in evaluators:
        evaluator.reset()

    for samples, targets in tqdm(dataloader, desc=desc):
        samples = samples.to("cuda")
        targets = [
            {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in t.items()} for t in targets
        ]

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs, orig_target_sizes)

        for evaluator in evaluators:
            evaluator.process(targets, results)

    results = {}
    for evaluator in evaluators:
        evaluator.synchronize_between_processes()
        results.update(evaluator.evaluate())
    return results


def evaluate_coco(model, postprocessor, img_folder, ann_files, batch_size=8):
    """COCO detection (`bbox`) and/or keypoint (`keypoints`) AP; `ann_files` = {iou_type: json}."""
    print("\n" + "=" * 60)
    print("COCO " + " + ".join(ann_files) + " Evaluation")
    print("=" * 60)

    dataset = CocoValDetection(
        img_folder,
        next(iter(ann_files.values())),
        transforms=eval_transform(),
        evaluators=[CocoEvaluator(ann_files)],
    )
    results = run_evaluators(model, postprocessor, dataset, batch_size, "COCO")

    stats = {}
    if "bbox" in results:
        r = results["bbox"]
        stats.update(
            {
                "det_AP": r["bbox_AP"],
                "det_AP50": r["bbox_AP_50"],
                "det_AP75": r["bbox_AP_75"],
                "det_AP_S": r["bbox_AP_S"],
                "det_AP_M": r["bbox_AP_M"],
                "det_AP_L": r["bbox_AP_L"],
            }
        )
    if "keypoints" in results:
        r = results["keypoints"]
        stats.update(
            {
                "kpt_AP": r["keypoints_AP"],
                "kpt_AP50": r["keypoints_AP_50"],
                "kpt_AP75": r["keypoints_AP_75"],
                "kpt_AP_M": r["keypoints_AP_M"],
                "kpt_AP_L": r["keypoints_AP_L"],
            }
        )
    return stats


def evaluate_oih_contacthands(model, postprocessor, img_folder, gt_folder, batch_size=8):
    """Evaluate OiH on ContactHands."""
    print("\n" + "=" * 60)
    print("OiH ContactHands Evaluation")
    print("=" * 60)

    oih_config = postprocessor.oih_config

    dataset = ContactHands(
        mode="val",
        img_folder=img_folder,
        gt_folder=gt_folder,
        transforms=eval_transform(),
        oih_config=oih_config,
        evaluators=[
            OiHEvaluator(
                oih_config=oih_config,
                person_box_oracle=True,
                eval_hand_obj=oih_config.get("oih_obj_box", False),
            )
        ],
    )
    print(f"ContactHands dataset: {len(dataset)} images")

    result = run_evaluators(model, postprocessor, dataset, batch_size, "OiH")

    oih_stats = result.get("oih", {})
    print("\nResults:")
    for k, v in oih_stats.items():
        print(f"  {k}: {v}")

    return {
        "oih_TP_Medium": oih_stats.get("Acc|TP_Medium(person+hand+cls. IoU0.5)", 0),
        "oih_TP_Soft": oih_stats.get("Acc|TP_Soft(IoU0.5)", 0),
        "oih_TP_Hard": oih_stats.get("Acc|TP_Hard(person+hand+cls+obj. IoU0.5)", 0),
        "oih_correct_medium": oih_stats.get("correct", 0),
        "oih_correct_hard": oih_stats.get("correct_hard", 0),
    }


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SQ-BHOI (COCO det / keypoints, ContactHands OiH)."
    )
    parser.add_argument(
        "--config", type=str, default=None, help="config YAML (default: configs/sqbhoi_r50vd.yml)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint to evaluate (default: weights/checkpoint.pth)",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="batch size (default: 8)")
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        default=["det", "kpt", "oih"],
        choices=["det", "kpt", "oih"],
        help="evaluations to run (default: det kpt oih)",
    )

    # Dataset paths
    parser.add_argument(
        "--coco-val-img",
        type=str,
        default=None,
        help="COCO val2017 image folder (default: datasets/mscoco/images/val2017)",
    )
    parser.add_argument(
        "--coco-val-ann",
        type=str,
        default=None,
        help="COCO instances_val2017.json (default: datasets/mscoco/annotations/instances_val2017.json)",
    )
    parser.add_argument(
        "--coco-kpt-ann",
        type=str,
        default=None,
        help="COCO person_keypoints_val2017.json "
        "(default: datasets/mscoco/annotations/person_keypoints_val2017.json)",
    )
    parser.add_argument(
        "--contacthands-img",
        type=str,
        default=None,
        help="ContactHands image folder (default: datasets/ContactHands/JPEGImages)",
    )
    parser.add_argument(
        "--contacthands-gt",
        type=str,
        default=None,
        help="ContactHands GT folder (default: datasets/ContactHands/eval_Annotations)",
    )

    args = parser.parse_args()

    # Defaults are relative to the repo root, so the script works from any directory.
    def root(path):
        return os.path.join(REPO_ROOT, path)

    args.config = args.config or root("configs/sqbhoi_r50vd.yml")
    args.checkpoint = args.checkpoint or root("weights/checkpoint.pth")
    args.coco_val_img = args.coco_val_img or root("datasets/mscoco/images/val2017")
    args.coco_val_ann = args.coco_val_ann or root(
        "datasets/mscoco/annotations/instances_val2017.json"
    )
    args.coco_kpt_ann = args.coco_kpt_ann or root(
        "datasets/mscoco/annotations/person_keypoints_val2017.json"
    )
    args.contacthands_img = args.contacthands_img or root("datasets/ContactHands/JPEGImages")
    args.contacthands_gt = args.contacthands_gt or root("datasets/ContactHands/eval_Annotations")

    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Evaluations: {args.eval}")

    # Load model
    model, postprocessor = load_model(args.config, args.checkpoint)
    print("Model loaded successfully.\n")

    all_stats = {}

    # COCO detection / keypoint evaluation (one pass over val2017)
    ann_files = {}
    for name, iou_type, path in (
        ("det", "bbox", args.coco_val_ann),
        ("kpt", "keypoints", args.coco_kpt_ann),
    ):
        if name not in args.eval:
            continue
        if os.path.exists(path):
            ann_files[iou_type] = path
        else:
            print(f"[SKIP] {name} eval: {path} not found")
    if ann_files:
        all_stats.update(
            evaluate_coco(
                model,
                postprocessor,
                args.coco_val_img,
                ann_files,
                batch_size=args.batch_size,
            )
        )

    # OiH evaluation
    if "oih" in args.eval:
        if os.path.exists(args.contacthands_gt):
            stats = evaluate_oih_contacthands(
                model,
                postprocessor,
                args.contacthands_img,
                args.contacthands_gt,
                batch_size=args.batch_size,
            )
            all_stats.update(stats)
        else:
            print(f"[SKIP] OiH eval: {args.contacthands_gt} not found")

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for k, v in all_stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
