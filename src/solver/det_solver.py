"""
by lyuwenyu
"""

import datetime
import json
import time

import numpy as np

from src.misc import dist

from .det_engine import evaluate, train_one_epoch
from .solver import BaseSolver

# task -> list of (key in the evaluation stats, metric used to select the best checkpoint).
# A metric starting with '-' is lower-is-better. For datasets other than COCO the key of the
# stats carries a `_<dataset_name>` suffix (see `det_engine.evaluate`).
TASK_METRICS = {
    "detection": [("bbox", "bbox_AP")],
    "body": [("keypoints", "keypoints_AP")],
    "oih": [("oih_ContactHands", "correct")],
}


def _set_epoch(dataloader, epoch: int) -> None:
    if hasattr(dataloader, "set_epoch"):
        dataloader.set_epoch(epoch)


class DetSolver(BaseSolver):
    def fit(
        self,
    ) -> None:
        self.logger.info("Start training")
        self.train()

        args = self.cfg

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"number of params:{n_parameters}")

        task_metric = {
            t_name: metric for task in self.task for t_name, metric in TASK_METRICS.get(task, [])
        }

        if self.best_stat is None:
            self.best_stat = {}

        checkpoint_dir = self.output_dir / "checkpoint"

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            _set_epoch(self.train_dataloader, epoch)
            for dataloader in self.val_dataloader:
                _set_epoch(dataloader, epoch)

            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.lr_scheduler,
                self.device,
                epoch,
                self.logger,
                args.clip_max_norm,
                print_freq=args.log_step,
                ema=self.ema,
                scaler=self.scaler,
            )

            # epoch-based schedulers; iter-based ones are stepped inside `train_one_epoch`
            if self.lr_scheduler is not None and not getattr(
                self.lr_scheduler, "iter_based", False
            ):
                self.lr_scheduler.step()

            # save checkpoint before evaluation
            checkpoint_dir.mkdir(exist_ok=True)
            dist.save_on_master(self.state_dict(epoch), checkpoint_dir / "checkpoint.pth")

            module = self.ema.module if self.ema else self.model

            test_stats = evaluate(
                module, self.postprocessor, self.val_dataloader, self.device, self.logger
            )

            if dist.is_main_process():
                for k, v in train_stats.items():
                    self.summary_writer.add_scalar(f"Train/{k}", v, epoch)

                for k, v in test_stats.items():
                    # tensorboard needs flat scalars; nested dicts (e.g. oih_by_person_count) are flattened
                    scalars = {}
                    for name, value in v.items():
                        if isinstance(value, dict):
                            scalars.update({f"{name}/{n}": x for n, x in value.items()})
                        else:
                            scalars[name] = value
                    self.summary_writer.add_scalars(f"Test/{k}", scalars, epoch)

                self._update_best_stat(task_metric, test_stats, epoch)
                self.logger.info("best_stat: %s", self.best_stat)

            # save checkpoint with best_stat
            for t_name, stat in self.best_stat.items():
                if stat["updated"]:
                    path = checkpoint_dir / f"best_checkpoint_{t_name}_{task_metric[t_name]}.pth"
                    dist.save_on_master(self.state_dict(epoch), path)
                    stat["updated"] = False

            dist.save_on_master(self.state_dict(epoch), checkpoint_dir / "checkpoint.pth")

            if dist.is_main_process():
                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    **{f"test_{k}": v for k, v in test_stats.items()},
                    "epoch": epoch,
                    "n_parameters": n_parameters,
                }

                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info(f"Training time {total_time_str}")

    def _update_best_stat(self, task_metric: dict, test_stats: dict, epoch: int) -> None:
        for t_name, metric in task_metric.items():
            sign = 1
            if metric.startswith("-"):
                sign, metric = -1, metric[1:]

            if t_name not in test_stats or metric not in test_stats[t_name]:
                continue

            value = test_stats[t_name][metric]
            best = self.best_stat.get(t_name)
            if best is None or sign * value > sign * best[metric] or np.isnan(best[metric]):
                self.best_stat[t_name] = {"epoch": epoch, metric: value, "updated": True}

    def val(
        self,
    ) -> None:
        self.eval()

        for dataloader in self.val_dataloader:
            _set_epoch(dataloader, -1)

        module = self.ema.module if self.ema else self.model
        evaluate(module, self.postprocessor, self.val_dataloader, self.device, self.logger)
