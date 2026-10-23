"""by lyuwenyu"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from src.core import BaseConfig
from src.misc import dist


class BaseSolver(object):
    def __init__(self, cfg: BaseConfig, task: List[str]) -> None:

        self.cfg = cfg
        self.logger = cfg.logger
        self.task = task

        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None
        self.ema = None
        self.best_stat = None

    def setup(
        self,
    ):
        """Avoid instantiating unnecessary classes"""
        cfg = self.cfg
        device = cfg.device
        self.device = device
        self.last_epoch = cfg.last_epoch

        self.model = dist.warp_model(cfg.model.to(device), cfg.find_unused_parameters, cfg.sync_bn)
        self.criterion = cfg.criterion.to(device) if cfg.criterion is not None else None
        self.postprocessor = cfg.postprocessor

        # NOTE (lvwenyu): should load_tuning_state before ema instance building
        if self.cfg.tuning:
            self.logger.info(f"Tuning checkpoint from {self.cfg.tuning}")
            self.load_tuning_state(self.cfg.tuning)

        self.scaler = cfg.scaler
        self.ema = cfg.ema.to(device) if cfg.ema is not None else None

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.summary_writer = cfg.summary_writer

    def _warp_val_dataloaders(
        self,
    ) -> None:
        """`cfg.val_dataloader` is a loader or a list of loaders; `self.val_dataloader` is always a list."""
        val_dataloaders = self.cfg.val_dataloader
        if not isinstance(val_dataloaders, list):
            val_dataloaders = [val_dataloaders]

        self.val_dataloader = [
            dist.warp_loader(loader, shuffle=loader.shuffle, seed=self.cfg.seed)
            for loader in val_dataloaders
        ]
        self.cfg.val_dataloader = self.val_dataloader

    def train(
        self,
    ) -> None:
        self.setup()

        if dist.is_main_process():
            self.cfg.save_config()

        self.train_dataloader = dist.warp_loader(
            self.cfg.train_dataloader, shuffle=self.cfg.train_dataloader.shuffle, seed=self.cfg.seed
        )
        # NOTE the lr scheduler is built from `len(cfg.train_dataloader)` (per-process iterations)
        self.cfg.train_dataloader = self.train_dataloader
        self._warp_val_dataloaders()

        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler

        # NOTE instantiating order
        if self.cfg.resume:
            self.logger.info(f"Resume checkpoint from {self.cfg.resume}")
            self.resume(self.cfg.resume)

    def eval(
        self,
    ) -> None:
        self.setup()
        self._warp_val_dataloaders()

        if self.cfg.resume:
            self.logger.info(f"resume from {self.cfg.resume}")
            self.resume(self.cfg.resume)

    def state_dict(self, last_epoch: int) -> Dict:
        """state dict"""
        state = {}
        state["model"] = dist.de_parallel(self.model).state_dict()
        state["date"] = datetime.now().isoformat()
        state["last_epoch"] = last_epoch

        if self.optimizer is not None:
            state["optimizer"] = self.optimizer.state_dict()

        if self.lr_scheduler is not None:
            state["lr_scheduler"] = self.lr_scheduler.state_dict()

        if self.ema is not None:
            state["ema"] = self.ema.state_dict()

        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()

        if self.best_stat is not None:
            state["best_stat"] = self.best_stat

        return state

    def load_state_dict(self, state: Dict) -> None:
        """load state dict"""
        if "last_epoch" in state:
            self.last_epoch = state["last_epoch"]
            self.logger.info("Loading last_epoch")

        if "model" in state:
            dist.de_parallel(self.model).load_state_dict(state["model"])
            self.logger.info("Loading model.state_dict")

        if self.ema is not None and "ema" in state:
            self.ema.load_state_dict(state["ema"])
            self.logger.info("Loading ema.state_dict")

        if self.optimizer is not None and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
            self.logger.info("Loading optimizer.state_dict")

        if self.lr_scheduler is not None and "lr_scheduler" in state:
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])
            self.logger.info("Loading lr_scheduler.state_dict")

        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])
            self.logger.info("Loading scaler.state_dict")

        if "best_stat" in state:
            self.best_stat = state["best_stat"]
            for t_name in self.best_stat:
                self.best_stat[t_name]["updated"] = False
            self.logger.info("Loading best_stat")

    def resume(self, path: str) -> None:
        """load resume"""
        # for cuda:0 memory
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(state)

    def load_tuning_state(self, path: str) -> None:
        """only load model for tuning and skip missed/dismatched keys"""
        if "http" in path:
            state = torch.hub.load_state_dict_from_url(path, map_location="cpu", weights_only=False)
        else:
            state = torch.load(path, map_location="cpu", weights_only=False)

        module = dist.de_parallel(self.model)

        # TODO hard code
        params = state["ema"]["module"] if "ema" in state else state["model"]
        stat, infos = self._matched_state(module.state_dict(), params)

        results = module.load_state_dict(stat, strict=False)
        infos["missed"] = results.missing_keys
        self.logger.info(f"Load model.state_dict, {infos}")

    @staticmethod
    def _matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]) -> Tuple:
        missed_list = []
        unmatched_list = []
        matched_state = {}
        for k, v in state.items():
            if k in params:
                if v.shape == params[k].shape:
                    matched_state[k] = params[k]
                else:
                    unmatched_list.append(k)
            else:
                missed_list.append(k)

        return matched_state, {"missed": missed_list, "unmatched": unmatched_list}

    def fit(
        self,
    ) -> None:
        raise NotImplementedError("")

    def val(
        self,
    ) -> None:
        raise NotImplementedError("")
