"""Train / evaluate loops.

Derived from RT-DETR (https://github.com/lyuwenyu/RT-DETR), adapted to multi-task
(detection, body pose, object-in-hand) training and several validation sets.
"""

import math
import sys
from typing import Dict, Iterable, List, Optional, Union

import torch

from src.misc import InfoLogger, MetricLogger, SmoothedValue
from src.misc.dist import reduce_dict


def _to_device(targets: List[Dict], device: torch.device) -> List[Dict]:
    return [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    device: torch.device,
    epoch: int,
    logger: InfoLogger,
    max_norm: float = 0,
    print_freq: int = 10,
    ema=None,
    scaler=None,
) -> Dict:
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    for idx in range(len(optimizer.param_groups)):
        metric_logger.add_meter(f"lr_{idx}", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    iter_based = getattr(scheduler, "iter_based", False)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = _to_device(targets, device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type=device.type, cache_enabled=True):
                outputs = model(samples, targets)
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()

        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values()).detach()

        if not math.isfinite(loss_value):
            logger.info(f"Loss is {loss_value}, stopping training")
            logger.info(f"{loss_dict_reduced}")
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(
            **{f"lr_{idx}": group["lr"] for idx, group in enumerate(optimizer.param_groups)}
        )

        if iter_based:
            scheduler.step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: %s", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    postprocessor: torch.nn.Module,
    data_loaders: Union[Iterable, List[Iterable]],
    device: torch.device,
    logger: InfoLogger,
) -> Dict:
    """Run every validation loader with the evaluators of its dataset.

    Returns `{iou_type or task: {metric: value}}`; the keys of datasets whose `dataset_name`
    is not `COCO` get a `_<dataset_name>` suffix (e.g. `oih_ContactHands`).
    """
    model.eval()

    if not isinstance(data_loaders, list):
        data_loaders = [data_loaders]

    stats = {}
    for data_loader in data_loaders:
        dataset = data_loader.dataset
        evaluators = dataset.get_evaluators()
        for evaluator in evaluators:
            evaluator.reset()

        metric_logger = MetricLogger(delimiter="  ", logger=logger)
        for samples, targets in metric_logger.log_every(data_loader, 10, "Test:"):
            samples = samples.to(device)
            targets = _to_device(targets, device)

            outputs = model(samples)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessor(outputs, orig_target_sizes)

            for evaluator in evaluators:
                evaluator.process(targets, results)

        # gather the predictions from all processes and accumulate
        for evaluator in evaluators:
            evaluator.synchronize_between_processes()

        dataset_name = getattr(dataset, "dataset_name", "COCO")
        for evaluator in evaluators:
            res = evaluator.evaluate()
            if dataset_name != "COCO":
                res = {f"{k}_{dataset_name}": v for k, v in res.items()}
            stats.update(res)

    logger.info("Evaluate_Result: %s", stats)

    return stats
