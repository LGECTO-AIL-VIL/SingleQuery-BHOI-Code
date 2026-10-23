"""
Export SQ-BHOI to ONNX.

Usage (from any directory):
    python3 tools/export_onnx.py --resume weights/checkpoint.pth --file-name weights/checkpoint.onnx --check --simplify
"""

import argparse
import inspect
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn

from src.core import YAMLConfig


class DeployModel(nn.Module):
    """Wraps model + postprocessor (deploy mode) and drops the `None`
    placeholders returned by SQBHOIOiHPostProcessor in deploy mode so the
    graph only has real tensor outputs. `has_body`/`has_oih` control whether
    the pose/oih outputs are included at all -- torch.onnx.export can't
    handle a `None` output, so tasks that are disabled must be omitted
    entirely rather than passed through as None."""

    def __init__(self, cfg, has_body=True, has_oih=True) -> None:
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        self.has_body = has_body
        self.has_oih = has_oih

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        labels, boxes, scores, _, poses, _, oihs = self.postprocessor(outputs, orig_target_sizes)
        result = [labels, boxes, scores]
        if self.has_body:
            result.append(poses)
        if self.has_oih:
            result.append(oihs)
        return tuple(result)


def main(args):
    if not args.resume:
        raise AttributeError("only support resume to load model.state_dict by now.")
    cfg = YAMLConfig(args.config, resume=args.resume)

    checkpoint = torch.load(args.resume, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    task = cfg.yaml_cfg.get("task", ["detection"])
    has_body = "body" in task
    has_oih = "oih" in task
    model = DeployModel(cfg, has_body=has_body, has_oih=has_oih)
    model.eval()
    # Parameters (e.g. oih_bias) still have requires_grad=True after eval();
    # torch.jit tracing refuses to embed such tensors as constants, so detach
    # all params for the export pass.
    for p in model.parameters():
        p.requires_grad_(False)
    # `dec_oih_bias`/`enc_oih_bias` are plain nn.Parameter objects stored in a
    # python list (not registered via nn.ParameterList), so they are missed by
    # model.parameters() above. Detach them explicitly.
    transformer = getattr(model.model, "decoder", None)  # SQBHOIOiHTransformer
    if transformer is not None:
        if getattr(transformer, "enc_oih_bias", None) is not None:
            transformer.enc_oih_bias.requires_grad_(False)
        inner_decoder = getattr(transformer, "decoder", None)
        if inner_decoder is not None and getattr(inner_decoder, "oih_bias", None):
            for b in inner_decoder.oih_bias:
                b.requires_grad_(False)

    data = torch.rand(1, 3, *args.size)
    size = torch.tensor([[args.size[1], args.size[0]]], dtype=torch.float32)

    output_names = ["labels", "boxes", "scores"]
    if has_body:
        output_names.append("poses")
    if has_oih:
        output_names.append("oihs")
    dynamic_axes = {name: {0: "N"} for name in ["images", "orig_target_sizes"] + output_names}

    # torch>=2.9 defaults to the dynamo exporter (forces opset>=18 and writes external
    # .data files); use the TorchScript exporter so opset 16 / single file is kept.
    torchscript_kwargs = (
        {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    )

    with torch.no_grad():
        torch.onnx.export(
            model,
            (data, size),
            args.file_name,
            input_names=["images", "orig_target_sizes"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=16,
            verbose=False,
            **torchscript_kwargs,
        )
    print(f"Exported ONNX model to {args.file_name} (outputs: {output_names})")

    if args.check:
        import onnx

        onnx.checker.check_model(onnx.load(args.file_name))
        print("Check export onnx model done...")

    if args.simplify:
        import onnx
        import onnxsim

        input_shapes = {"images": data.shape, "orig_target_sizes": size.shape}
        onnx_model_simplify, check = onnxsim.simplify(
            args.file_name, input_shapes=input_shapes, dynamic_input_shape=True
        )
        onnx.save(onnx_model_simplify, args.file_name)
        print(f"Simplify onnx model {check}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SQ-BHOI to ONNX.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="config YAML (default: configs/sqbhoi_r50vd.yml)",
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
        default=None,
        help="checkpoint (default: weights/checkpoint.pth)",
    )
    parser.add_argument(
        "--file-name",
        "-f",
        type=str,
        default=None,
        help="output ONNX file (default: weights/checkpoint.onnx)",
    )
    parser.add_argument(
        "--size", type=int, nargs=2, default=[640, 640], help="input size H W (default: 640 640)"
    )
    parser.add_argument(
        "--check", action="store_true", default=False, help="validate the exported model with onnx"
    )
    parser.add_argument(
        "--simplify", action="store_true", default=False, help="simplify the model with onnxsim"
    )

    args = parser.parse_args()
    args.config = args.config or os.path.join(REPO_ROOT, "configs/sqbhoi_r50vd.yml")
    args.resume = args.resume or os.path.join(REPO_ROOT, "weights/checkpoint.pth")
    args.file_name = args.file_name or os.path.join(REPO_ROOT, "weights/checkpoint.onnx")
    main(args)
