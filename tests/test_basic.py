"""CPU-only smoke tests; no dataset needed. Run with `pytest tests`."""

import os
import sys

import pytest
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.core import YAMLConfig
from src.optim import ModelEMA
from src.optim.scheduler import ContinuosCosineAnnealingLR, QuadraticWarmupLR

CONFIG = os.path.join(REPO_ROOT, "configs", "sqbhoi_r50vd.yml")
CHECKPOINT = os.path.join(REPO_ROOT, "weights", "checkpoint.pth")


@pytest.fixture(scope="module")
def cfg():
    # pretrained=False: do not download ImageNet weights in tests
    return YAMLConfig(CONFIG, PResNet={"pretrained": False})


@pytest.fixture(scope="module")
def model(cfg):
    torch.manual_seed(0)
    return cfg.model.eval()


def test_config_builds(cfg, model):
    assert type(model).__name__ == "SQBHOI"
    assert type(cfg.postprocessor).__name__ == "SQBHOIOiHPostProcessor"
    criterion = cfg.criterion
    assert type(criterion).__name__ == "SQBHOICriterion"
    assert criterion.weight_dict["loss_vfl"] > 0
    assert cfg.epoches == 72


def test_forward_shapes(cfg, model):
    with torch.no_grad():
        out = model(torch.randn(1, 3, 640, 640))
    assert out["pred_logits"].shape == (1, 300, 81)
    assert out["pred_boxes"].shape == (1, 300, 4)
    assert out["pred_poses"].shape == (1, 300, 34)
    assert out["pred_vis"].shape == (1, 300, 17)
    assert out["pred_oih"].shape[:3] == (1, 300, 2)
    assert all(torch.isfinite(v).all() for v in out.values() if torch.is_tensor(v))

    with torch.no_grad():
        res = cfg.postprocessor.eval()(out, torch.tensor([[480, 640]]))
    assert len(res) == 1
    assert res[0]["boxes"].shape[-1] == 4
    assert res[0]["poses"].shape[-2:] == (17, 3)


def test_ema_tracks_model():
    net = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    ema = ModelEMA(net, decay=0.9, warmups=1, k=1)
    before = ema.module[0].weight.clone()
    with torch.no_grad():
        net[0].weight.add_(1.0)
    ema.update(net)
    after = ema.module[0].weight
    assert ema.updates == 1
    assert not torch.equal(before, after)
    # EMA moves towards the model but never past it
    assert torch.all(after > before) and torch.all(after < net[0].weight)


def test_schedulers():
    p = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    warm = QuadraticWarmupLR(opt, begin=0, end=5)
    lrs = []
    for _ in range(5):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        warm.step()
    assert lrs == sorted(lrs) and lrs[0] < lrs[-1]
    # lr_t = (t + 1)^2 / end^2 * base_lr: base lr is reached at the last warmup step
    assert lrs[0] == pytest.approx(1 / 25)
    assert lrs[-1] == pytest.approx(1.0)

    opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    cos = ContinuosCosineAnnealingLR(opt, T_max=10, eta_min_ratio=0.05)
    for _ in range(10):
        opt.step()
        cos.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.05)


@pytest.mark.skipif(not os.path.exists(CHECKPOINT), reason="weights/checkpoint.pth missing")
def test_checkpoint_strict_load(model):
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["ema"]["module"], strict=True)
