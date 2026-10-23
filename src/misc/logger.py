"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/misc.py
Mostly copy-paste from torchvision references.
"""

import datetime
import importlib
import logging
import os
import time
from collections import defaultdict, deque

import torch
import torch.distributed as tdist

from src.core import register

from .dist import is_dist_available_and_initialized, is_main_process


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_available_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        tdist.barrier()
        tdist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t", logger=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.print = print if logger is None else logger.info

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    self.print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    self.print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


@register
class InfoLogger(object):
    """
    use logging to record log, only work on the main process
    """

    def __init__(self, output_dir):
        self.is_main_process = is_main_process()

        self.logger = self.get_logger(output_dir)
        self.infologger_ftns = {"info", "warning", "debug", "error"}

    def __getattr__(self, name):
        if not self.is_main_process:  # info only print on GPU 0.

            def wrapper(info, *args, **kwargs):
                pass

            return wrapper
        if name in self.infologger_ftns:
            return getattr(self.logger, name)
        raise AttributeError(name)

    @staticmethod
    def get_logger(output_dir, level=logging.INFO):
        logger = logging.getLogger("Training_Logger")
        logger.setLevel(level)
        if logger.handlers:
            return logger

        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(levelname)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        stdout_stream = logging.StreamHandler()
        stdout_stream.setFormatter(formatter)

        os.makedirs(output_dir, exist_ok=True)
        file_stream = logging.FileHandler(os.path.join(output_dir, "training_log.log"))
        file_stream.setFormatter(formatter)

        logger.addHandler(stdout_stream)
        logger.addHandler(file_stream)

        return logger


class DummyLogger:
    def nop(*args, **kw):
        pass

    def __getattr__(self, _):
        return self.nop


@register
class SummaryWriter(object):
    """tensorboard summary writer; a no-op on non-main processes or if disabled / not installed"""

    def __init__(self, logger, output_dir, enabled=True) -> None:
        self.writer = self.get_summary_writer(logger, output_dir, enabled)

    def __del__(self):
        self.writer.close()

    def __getattr__(self, name):
        if name == "writer":
            raise AttributeError(name)
        return getattr(self.writer, name)

    @staticmethod
    def get_summary_writer(logger, output_dir, enabled):
        writer = DummyLogger()
        if enabled and is_main_process():
            summary_dir = os.path.join(output_dir, "summaries")
            os.makedirs(summary_dir, exist_ok=True)

            for module in ["tensorboardX", "torch.utils.tensorboard"]:
                try:
                    writer = importlib.import_module(module).SummaryWriter(summary_dir)
                    logger.info(f"Create the tensorboard summary in directory {summary_dir}.")
                    break
                except ImportError:
                    continue
            else:
                logger.warning(
                    "Summary (tensorboard) is enabled but tensorboard is not installed. "
                    "Install tensorboard or set `summary.enabled: False`."
                )

        return writer
