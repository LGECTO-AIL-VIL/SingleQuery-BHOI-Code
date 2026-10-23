# Single-Query Person-Centric Bimanual Hand-Object Interaction Detection

[![arXiv](https://img.shields.io/badge/arXiv-2609.12155-b31b1b.svg)](https://arxiv.org/abs/2609.12155)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://lgecto-ail-vil.github.io/SingleQuery-BHOI/)
[![ECCV 2026](https://img.shields.io/badge/ECCV-2026-green.svg)](#citation)

Official implementation of **"Single-Query Person-Centric Bimanual Hand-Object Interaction Detection"** (ECCV 2026).

Jonghyun Kim, Junho Roh, Yubin Yoon, Hyotae Lee, Jongkuk Park, Taehwan Hwang, Jaechul Kim, Jungho Lee

[Paper (arXiv)](https://arxiv.org/abs/2609.12155) | [Project page](https://lgecto-ail-vil.github.io/SingleQuery-BHOI/)


## Overview

SQ-BHOI (SingleQuery-BHOI) uses one query per person to predict the human box, body pose, hand boxes and hand states, and the interaction targets of both hands. It uses part-aware deformable attention and a hand-to-query relationship matrix with a learnable "off" token. The code is built on a DETR-style detector (RT-DETR).

**Results (ResNet-50vd, COCO val2017 / ContactHands):**

| Task | Metric | Value |
|:-----|:-------|:-----:|
| Detection | AP | 48.5 |
| Body Pose | AP | 63.1 |
| Object-in-Hand (Soft) | Accuracy | 81.2 |
| Object-in-Hand (Medium) | Accuracy | 43.9 |
| Object-in-Hand (Hard) | Accuracy | 17.9 |

These numbers are reproduced by `tools/eval.py` with the released checkpoint.


## Installation

Tested with Python 3.12 and PyTorch 2.12. The minimum versions in `requirements.txt` (`torch>=2.0`, `torchvision>=0.15`) have not been tested.

```bash
pip install -r requirements.txt          # training / evaluation / demo
pip install -r requirements-export.txt   # additionally: ONNX export and ONNX Runtime inference
pip install -r requirements-dev.txt      # additionally: pytest, ruff
```

`requirements-export.txt` installs `onnxruntime-gpu`; use `onnxruntime` instead on CPU-only machines.


## Data Preparation

Expected layout (read by `configs/include/dataset.yml`):

```
datasets/
├── mscoco/
│   ├── images/{train2017,val2017}/
│   └── annotations/
│       ├── instances_val2017.json
│       └── person_keypoints_val2017.json
├── ContactHands/                     # evaluation only
│   ├── JPEGImages/
│   └── eval_Annotations/
└── singlequery_bhoi_train.json       # our training annotation
```

- **COCO** images and annotations are downloaded from [cocodataset.org](https://cocodataset.org) by the script below, or symlinked from an existing copy with `COCO_DIR`.
- **ContactHands** is not hosted here. Download it from the official release and point `CONTACTHANDS_DIR` at it.
- **`singlequery_bhoi_train.json`** is our training annotation (person boxes, 17 keypoints, hand and hand-object annotations for COCO train2017). It is downloaded from the [SingleQuery-BHOI-Dataset](https://github.com/LGECTO-AIL-VIL/SingleQuery-BHOI-Dataset) repository (`annotations/singlequery_bhoi_train.zip`) and verified by sha256.

```bash
CONTACTHANDS_DIR=/path/to/ContactHands bash scripts/prepare_data.sh
# optional: COCO_DIR=/path/to/existing/coco   (symlinked instead of downloading)
# optional: --no-train-images                 (skip train2017.zip)
```


## Model Weights

```bash
bash scripts/download_weights.sh   # from the GitHub Release v1.0.0 of this repository
```

This downloads `weights/checkpoint.pth` and verifies its sha256
(`770ffecd2264c8e1e50c214791c4aeb36d0a9b6d9852d336ec8ac627c06d33d5`).
See [Data Licenses and Attribution](#data-licenses-and-attribution) for the terms that apply to the weights.


## Evaluation

```bash
# COCO detection AP, COCO keypoint AP and ContactHands OiH (default: all three)
python tools/eval.py --config configs/sqbhoi_r50vd.yml --checkpoint weights/checkpoint.pth --eval det kpt oih

# a subset
python tools/eval.py --eval det
python tools/eval.py --eval oih
```

Dataset locations can be overridden with `--coco-val-img`, `--coco-val-ann`, `--coco-kpt-ann`, `--contacthands-img` and `--contacthands-gt`. `--batch-size` sets the evaluation batch size (default 8).


## Training

Training uses `datasets/singlequery_bhoi_train.json` and COCO train2017 only. The paper's model was trained with additional data, so numbers obtained with this recipe may differ from the paper. The ImageNet-pretrained backbone weights are downloaded on first use.

```bash
# single GPU
python tools/train.py -c configs/sqbhoi_r50vd.yml

# multi-GPU (reference setting: 8 GPUs)
torchrun --nproc_per_node=8 tools/train.py -c configs/sqbhoi_r50vd.yml
```

The default config is the reference setting: global batch 128 (16 images x 8 GPUs), 72 epochs. With a different number of GPUs, keep `batch_size x #GPUs = 16 x k` and rescale `ema.k`, `optimizer.lr`, the backbone lr, `optimizer.betas`, `optimizer.eps` and `optimizer.weight_decay` with the rules and the worked 2-GPU example in the header of [configs/sqbhoi_r50vd.yml](configs/sqbhoi_r50vd.yml). Config entries can be overridden on the command line with `-u key=value`.

Other options:

- `--amp`: automatic mixed precision.
- `-r <checkpoint>`: resume training; `--proceed`: resume from `<output_dir>/checkpoint/checkpoint.pth` if it exists.
- `-t <checkpoint>`: initialize the model weights from a checkpoint.
- `--test-only`: run validation only; `--seed N`: set the random seed.

Outputs (checkpoints, logs) are written to `output/sqbhoi_r50vd/` (`output_dir` in the config).


## Demo

```bash
python tools/demo.py --image demo/000000409475.jpg --output result.png
```

Defaults: `--config configs/sqbhoi_r50vd.yml`, `--checkpoint weights/checkpoint.pth`. Other flags: `--det-threshold`, `--kpt-threshold`, `--oih-threshold`, `--show-detection`, `--size H W`. The images in `demo/` are COCO val2017 images; see [demo/ATTRIBUTION.txt](demo/ATTRIBUTION.txt).


## ONNX Export and Inference

```bash
# weights/checkpoint.pth -> weights/checkpoint.onnx
python tools/export_onnx.py --check --simplify

python tools/onnx_live_inference.py --image demo/000000409475.jpg            # single image
python tools/onnx_live_inference.py --image-dir demo --output-dir demo_out   # folder of images
python tools/onnx_live_inference.py --camera 0                               # webcam
```

Export options: `-c/--config`, `-r/--resume`, `-f/--file-name`, `--size H W`. Inference options: `--onnx`, `--cpu`, `--headless`, `--save-video`, `--max-frames`, `--det-threshold`, `--kpt-threshold`, `--oih-threshold`, `--show-detection`.


## Tests

CPU-only smoke tests that need no dataset:

```bash
pytest tests
```


## Project Structure

```
SingleQuery-BHOI-Code/
├── assets/                  # pose_represent_coco.npy
├── configs/
│   ├── sqbhoi_r50vd.yml     # train / eval / demo config
│   └── include/             # dataset, runtime, optimizer, model configs
├── demo/                    # example images + ATTRIBUTION.txt
├── scripts/
│   ├── prepare_data.sh      # datasets/ setup
│   └── download_weights.sh  # weights/checkpoint.pth + sha256 check
├── src/
│   ├── core/                # config and registry
│   ├── data/                # dataloader, transforms, coco/ and oih/ (ContactHands) datasets and evaluators
│   ├── misc/                # distributed utilities, logger
│   ├── nn/backbone/         # PResNet backbone
│   ├── optim/               # optimizer, schedulers, EMA, AMP
│   ├── solver/              # train / eval loops
│   └── zoo/sqbhoi/          # model, decoder, criterion, matcher, postprocessor
├── tests/                   # test_basic.py
├── tools/                   # train.py, eval.py, demo.py, export_onnx.py, onnx_live_inference.py
├── weights/                 # model checkpoints (git-ignored)
└── datasets/                # data (git-ignored)
```


## Data Licenses and Attribution

Please verify the upstream terms yourself before using any data or weights; the notes below are a summary, not legal advice.

- **Code**: Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).
- **COCO**: annotations are licensed CC BY 4.0; images are subject to the Flickr terms of their authors (see the [COCO terms of use](https://cocodataset.org/#termsofuse)). This repository does not redistribute COCO images except the three unmodified demo images in `demo/` (see [demo/ATTRIBUTION.txt](demo/ATTRIBUTION.txt)); download the rest yourself.
- **Training annotation `singlequery_bhoi_train.json`**: derived from COCO, COCO-WholeBody (research / non-commercial use only, SenseTime) and ContactHands (Narasimhaswamy et al., NeurIPS 2020; the code repository is MIT-licensed, please check the dataset terms). It is therefore released for **non-commercial research use only** under CC BY-NC 4.0.
- **Model weights**: the same non-commercial research notice applies to the released checkpoint, which was trained on the data above.


## Acknowledgements

This codebase is built upon [RT-DETR](https://github.com/lyuwenyu/RT-DETR) (lyuwenyu) and borrows from DETR, torchvision references, MMEngine and MMPose; see [NOTICE](NOTICE).


## Citation

```bibtex
@inproceedings{kim2026singlequery,
  title         = {Single-Query Person-Centric Bimanual Hand-Object Interaction Detection},
  author        = {Kim, Jonghyun and Roh, Junho and Yoon, Yubin and Lee, Hyotae and Park, Jongkuk and Hwang, Taehwan and Kim, Jaechul and Lee, Jungho},
  booktitle     = {European Conference on Computer Vision (ECCV)},
  year          = {2026},
  eprint        = {2609.12155},
  archivePrefix = {arXiv}
}
```
