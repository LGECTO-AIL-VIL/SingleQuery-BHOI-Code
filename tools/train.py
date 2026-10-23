"""by lyuwenyu

Train / validate SQ-BHOI (from any directory): python3 tools/train.py -c configs/sqbhoi_r50vd.yml
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import yaml

import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS


def parse_overrides(items):
    """`['a.b=1', 'epoches=2']` -> `{'a': {'b': 1}, 'epoches': 2}`; values are parsed as yaml."""
    overrides = {}
    for item in items or []:
        key, _, value = item.partition("=")
        node = overrides
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return overrides


def main(
    args,
) -> None:
    """main"""
    dist.init_distributed()

    # command line arguments override the config
    kwargs = parse_overrides(args.update)
    if args.proceed:
        kwargs["proceed"] = True
    if args.test_only:
        kwargs["test_only"] = True
    if args.resume:
        kwargs["resume"] = args.resume
    if args.tuning:
        kwargs["tuning"] = args.tuning
    if args.amp:
        kwargs["use_amp"] = True
    if args.seed is not None:
        kwargs["seed"] = args.seed

    cfg = YAMLConfig(args.config, **kwargs)
    dist.set_seed(cfg.seed)

    # `task` is a list of tasks: 'detection', 'body', 'oih'
    task = cfg.yaml_cfg["task"]
    task = [task] if isinstance(task, str) else list(task)
    solver = TASKS["multitask" if len(task) > 1 else task[0]](cfg, task)

    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train / validate SQ-BHOI. "
        "Multi-GPU: torchrun --nproc_per_node=N tools/train.py -c <config>"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="config YAML (e.g. configs/sqbhoi_r50vd.yml)",
    )
    parser.add_argument(
        "--resume", "-r", type=str, help="resume training from a checkpoint (model, optimizer, ...)"
    )
    parser.add_argument(
        "--proceed",
        action="store_true",
        default=False,
        help="resume from <output_dir>/checkpoint/checkpoint.pth if it exists",
    )
    parser.add_argument(
        "--tuning", "-t", type=str, help="initialize the model weights from a checkpoint"
    )
    parser.add_argument(
        "--test-only", action="store_true", default=False, help="only run the validation"
    )
    parser.add_argument(
        "--amp", action="store_true", default=False, help="use automatic mixed precision"
    )
    parser.add_argument("--seed", type=int, help="random seed (default: from the config)")
    parser.add_argument(
        "--update",
        "-u",
        nargs="+",
        metavar="KEY=VALUE",
        help="override config entries, e.g. -u epoches=1 train_dataloader.batch_size=4",
    )
    args = parser.parse_args()

    main(args)
