"""by lyuwenyu"""

import copy
import os
import re

import torch
import torch.nn as nn

from .catalog import get_metadata
from .config import BaseConfig
from .yaml_utils import create, load_config, merge_config, merge_dict, save_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIRS = ("datasets", "assets", "weights")


def resolve_repo_paths(node):
    """Make config paths under `datasets/`, `assets/` or `weights/` (optionally with a leading `./`)
    relative to the repo root instead of the current working directory."""
    if isinstance(node, dict):
        return {k: resolve_repo_paths(v) for k, v in node.items()}
    if isinstance(node, list):
        return [resolve_repo_paths(v) for v in node]
    if isinstance(node, str):
        rel = node[2:] if node.startswith("./") else node
        if rel.split("/", 1)[0] in REPO_DIRS and "/" in rel:
            return os.path.join(REPO_ROOT, rel)
    return node


class YAMLConfig(BaseConfig):
    def __init__(self, cfg_path: str, **kwargs) -> None:
        super().__init__()

        # paths written in config files are repo-root relative; command line overrides stay cwd relative
        cfg = resolve_repo_paths(load_config(cfg_path))
        merge_dict(cfg, kwargs)

        # pprint(cfg)

        self.yaml_cfg = cfg

        self.seed = cfg.get("seed", -1)
        self.log_step = cfg.get("log_step", 100)
        self.checkpoint_step = cfg.get("checkpoint_step", 1)
        self.epoches = cfg.get("epoches", -1)
        self.sync_bn = cfg.get("sync_bn", False)
        self.output_dir = cfg.get("output_dir", None)

        self.use_ema = cfg.get("use_ema", False)
        self.use_amp = cfg.get("use_amp", False)
        self.autocast = cfg.get("autocast", dict())
        self.find_unused_parameters = cfg.get("find_unused_parameters", None)
        self.clip_max_norm = cfg.get("clip_max_norm", 0.0)

        # `proceed`: resume from `<output_dir>/checkpoint/checkpoint.pth` if it exists,
        # otherwise start from the `resume` / `tuning` entries of the config.
        self.resume = cfg.get("resume", "")
        self.tuning = cfg.get("tuning", "")
        self.proceed = cfg.get("proceed", False) and not cfg.get("test_only", False)
        if self.proceed:
            last_ckpt_path = os.path.join(self.output_dir, "checkpoint/checkpoint.pth")
            if os.path.exists(last_ckpt_path):
                print(f"Proceed: resume from {last_ckpt_path}")
                self.resume = last_ckpt_path
                self.tuning = ""
            else:
                print(f"Proceed: {last_ckpt_path} does not exist, start from the config.")

        assert not (self.tuning and self.resume), (
            "Only support from_scratch or resume or tuning at one time"
        )

        self.metadata = get_metadata(cfg.get("category_ver", "1.9"))
        self.img_norm_mean, self.img_norm_std = self.get_img_normalize_args()

    @property
    def model(
        self,
    ) -> torch.nn.Module:
        if self._model is None and "model" in self.yaml_cfg:
            merge_config(self.yaml_cfg)
            self._model = create(self.yaml_cfg["model"])
        return self._model

    @property
    def postprocessor(
        self,
    ) -> torch.nn.Module:
        if self._postprocessor is None and "postprocessor" in self.yaml_cfg:
            merge_config(self.yaml_cfg)
            self._postprocessor = create(self.yaml_cfg["postprocessor"])
        return self._postprocessor

    @property
    def criterion(
        self,
    ):
        if self._criterion is None and "criterion" in self.yaml_cfg:
            merge_config(self.yaml_cfg)
            self._criterion = create(self.yaml_cfg["criterion"])
        return self._criterion

    @property
    def optimizer(
        self,
    ):
        if self._optimizer is None and "optimizer" in self.yaml_cfg:
            merge_config(self.yaml_cfg)
            params = self.get_optim_params(self.yaml_cfg["optimizer"], self.model)
            self._optimizer = create("optimizer", params=params)

        return self._optimizer

    @property
    def lr_scheduler(
        self,
    ):
        if self._lr_scheduler is None and "lr_scheduler" in self.yaml_cfg:
            merge_config(self.yaml_cfg)
            self._lr_scheduler = create(
                "lr_scheduler", optimizer=self.optimizer, epoch_length=len(self.train_dataloader)
            )
            self.logger.info("Initial lr: %s", self._lr_scheduler.get_last_lr())

        return self._lr_scheduler

    @property
    def train_dataloader(
        self,
    ):
        if self._train_dataloader is None and "train_dataloader" in self.yaml_cfg:
            merge_config(self.yaml_cfg)
            self._train_dataloader = create("train_dataloader")
            self._train_dataloader.shuffle = self.yaml_cfg["train_dataloader"].get("shuffle", False)

        return self._train_dataloader

    @train_dataloader.setter
    def train_dataloader(self, loader):
        self._train_dataloader = loader

    @property
    def val_dataloader(
        self,
    ):
        """A dataloader, or a list of dataloaders if `val_dataloader` is a list in the config."""
        if self._val_dataloader is None and "val_dataloader" in self.yaml_cfg:
            merge_config(self.yaml_cfg)
            self._val_dataloader = create("val_dataloader")
            if isinstance(self._val_dataloader, list):
                for loader, loader_cfg in zip(
                    self._val_dataloader, self.yaml_cfg["val_dataloader"]
                ):
                    loader.shuffle = loader_cfg.get("shuffle", False)
            else:
                self._val_dataloader.shuffle = self.yaml_cfg["val_dataloader"].get("shuffle", False)

        return self._val_dataloader

    @val_dataloader.setter
    def val_dataloader(self, loader):
        self._val_dataloader = loader

    @property
    def ema(
        self,
    ):
        if self._ema is None and self.yaml_cfg.get("use_ema", False):
            merge_config(self.yaml_cfg)
            self._ema = create("ema", model=self.model)

        return self._ema

    @property
    def scaler(
        self,
    ):
        if self._scaler is None and self.yaml_cfg.get("use_amp", False):
            merge_config(self.yaml_cfg)
            self._scaler = create("scaler")

        return self._scaler

    @property
    def logger(
        self,
    ):
        if self._logger is None:
            merge_config(self.yaml_cfg)
            self._logger = create("logger", output_dir=self.output_dir)
        return self._logger

    @property
    def summary_writer(
        self,
    ):
        if self._summary_writer is None:
            merge_config(self.yaml_cfg)
            self._summary_writer = create("summary", logger=self.logger, output_dir=self.output_dir)
        return self._summary_writer

    @staticmethod
    def get_optim_params(cfg: dict, model: nn.Module):
        """
        E.g.:
            ^(?=.*a)(?=.*b).*$         means including a and b
            ^((?!b.)*a((?!b).)*$       means including a but not b
            ^((?!b|c).)*a((?!b|c).)*$  means including a but not (b | c)
        """
        assert "type" in cfg, ""
        cfg = copy.deepcopy(cfg)

        if "params" not in cfg:
            return model.parameters()

        assert isinstance(cfg["params"], list), ""

        param_groups = []
        visited = []
        for pg in cfg["params"]:
            pattern = pg["params"]
            params = {
                k: v
                for k, v in model.named_parameters()
                if v.requires_grad and len(re.findall(pattern, k)) > 0
            }
            pg["params"] = params.values()
            param_groups.append(pg)
            visited.extend(list(params.keys()))

        names = [k for k, v in model.named_parameters() if v.requires_grad]

        if len(visited) < len(names):
            unseen = set(names) - set(visited)
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
            param_groups.append({"params": params.values()})
            visited.extend(list(params.keys()))

        assert len(visited) == len(names), ""

        return param_groups

    def save_config(self) -> None:
        save_config(os.path.join(self.output_dir, "config.yml"), self.yaml_cfg)

    def get_img_normalize_args(self):
        """mean/std of the `Normalize` op of the (first) training dataset."""
        mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

        dataset = self.yaml_cfg.get("train_dataloader", {}).get("dataset")
        while dataset is not None and "transforms" not in dataset:
            if "dataset" in dataset:
                dataset = dataset["dataset"]
            elif "datasets" in dataset:
                dataset = dataset["datasets"][0]
            else:
                dataset = None

        if dataset is not None:
            for op in dataset["transforms"]["ops"]:
                if op["type"] == "Normalize":
                    mean, std = op["mean"], op["std"]

        return mean, std
