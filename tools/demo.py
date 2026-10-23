"""
SQ-BHOI inference & visualization demo (detection, pose, hand-object interaction).

Usage (from any directory):
    python3 tools/demo.py --image path/to/image.jpg --checkpoint weights/checkpoint.pth --output output.png
"""

import argparse
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
import torchvision.transforms.v2 as T
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.core import YAMLConfig

# COCO class names (81 classes, index 0 = background for focal loss)
COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# COCO skeleton connections for body keypoints
COCO_SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # head
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),  # arms
    (5, 11),
    (6, 12),
    (11, 12),  # torso
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),  # legs
]

# Color palette for keypoints
KPT_COLORS = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
]

# Skeleton limb colors
LIMB_COLORS = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
]

# Detection box colors (cycling palette)
DET_COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]

# OiH colors
HAND_COLOR_RIGHT = (255, 100, 100)  # red-ish for right hand
HAND_COLOR_LEFT = (100, 100, 255)  # blue-ish for left hand
OBJ_COLOR = (100, 255, 100)  # green for held object
LINE_COLOR_RIGHT = (255, 50, 50)
LINE_COLOR_LEFT = (50, 50, 255)


def load_model(config_path, checkpoint_path):
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    model = cfg.model.to("cuda")
    postprocessor = cfg.postprocessor.to("cuda")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["ema"]["module"])
    model.eval()
    postprocessor.eval()
    return model, postprocessor


def preprocess_image(image_path, size=(640, 640)):
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    orig_w, orig_h = img.size

    transform = T.Compose(
        [
            T.Resize(size=list(size)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    img_tensor = transform(img).unsqueeze(0).to("cuda")
    # orig_target_sizes format is [W, H] in this codebase
    orig_size = torch.tensor([[orig_w, orig_h]], dtype=torch.float32).to("cuda")
    return img, img_tensor, orig_size


def draw_detection(draw, box, label_idx, score):
    """Draw detection box with label."""
    x1, y1, x2, y2 = box
    color = DET_COLORS[label_idx % len(DET_COLORS)]
    label_name = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else f"cls{label_idx}"

    # Draw box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    # Draw label background
    text = f"{label_name} {score:.2f}"
    font = _get_font(12)
    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # Position label above box, or below if no space
    ly = y1 - th - 6
    if ly < 0:
        ly = y2 + 2
    draw.rectangle([x1, ly, x1 + tw + 6, ly + th + 4], fill=color)
    draw.text((x1 + 3, ly + 1), text, fill=(255, 255, 255), font=font)


def draw_skeleton(draw, keypoints, vis_threshold=0.3):
    """Draw body skeleton with keypoints."""
    kpts = keypoints  # (17, 3): x, y, vis

    # Draw limbs
    for idx, (i, j) in enumerate(COCO_SKELETON):
        if kpts[i, 2] > vis_threshold and kpts[j, 2] > vis_threshold:
            x1, y1 = int(kpts[i, 0]), int(kpts[i, 1])
            x2, y2 = int(kpts[j, 0]), int(kpts[j, 1])
            color = LIMB_COLORS[idx % len(LIMB_COLORS)]
            draw.line([(x1, y1), (x2, y2)], fill=color, width=2)

    # Draw keypoints
    for i in range(len(kpts)):
        if kpts[i, 2] > vis_threshold:
            x, y = int(kpts[i, 0]), int(kpts[i, 1])
            color = KPT_COLORS[i % len(KPT_COLORS)]
            r = 4
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(255, 255, 255), width=1)


def draw_oih(draw, oih_data, det_score_threshold=0.3):
    """Draw OiH hand-object interaction.

    oih_data per query: (2 * 11) = [
        rh_score, rh_label, rh_box(x1y1x2y2), rh_obj_box(x1y1x2y2), rh_obj_label,
        lh_score, lh_label, lh_box(x1y1x2y2), lh_obj_box(x1y1x2y2), lh_obj_label,
    ]
    """
    # Right hand (first 11 values)
    rh_score = oih_data[0]
    rh_label = int(oih_data[1])  # 1=contact, 0=no contact
    rh_box = oih_data[2:6]
    rh_obj_box = oih_data[6:10]

    # Left hand (next 11 values)
    lh_score = oih_data[11]
    lh_label = int(oih_data[12])
    lh_box = oih_data[13:17]
    lh_obj_box = oih_data[17:21]

    # Draw right hand
    if rh_score > det_score_threshold:
        _draw_hand_interaction(
            draw, rh_box, rh_obj_box, rh_label, rh_score, HAND_COLOR_RIGHT, LINE_COLOR_RIGHT, "R"
        )

    # Draw left hand
    if lh_score > det_score_threshold:
        _draw_hand_interaction(
            draw, lh_box, lh_obj_box, lh_label, lh_score, HAND_COLOR_LEFT, LINE_COLOR_LEFT, "L"
        )


def _draw_hand_interaction(
    draw, hand_box, obj_box, label, score, hand_color, line_color, hand_side
):
    """Draw a single hand-object interaction."""
    hx1, hy1, hx2, hy2 = [int(v) for v in hand_box]

    # Draw hand box with rounded feel (double outline)
    draw.rectangle([hx1, hy1, hx2, hy2], outline=hand_color, width=3)

    # Label
    status = "contact" if label == 1 else "free"
    font = _get_font(10)
    text = f"{hand_side}:{status} {score:.2f}"
    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    ly = hy2 + 2
    draw.rectangle([hx1, ly, hx1 + tw + 4, ly + th + 2], fill=hand_color + (200,))
    draw.text((hx1 + 2, ly), text, fill=(255, 255, 255), font=font)

    # If contact, draw object box and connection line
    if label == 1:
        ox1, oy1, ox2, oy2 = [int(v) for v in obj_box]
        draw.rectangle([ox1, oy1, ox2, oy2], outline=OBJ_COLOR, width=2)

        # Draw connection line from hand center to object center
        hcx, hcy = (hx1 + hx2) // 2, (hy1 + hy2) // 2
        ocx, ocy = (ox1 + ox2) // 2, (oy1 + oy2) // 2
        # Draw dashed-style arrow line
        _draw_arrow(draw, hcx, hcy, ocx, ocy, line_color, width=2)


def _draw_arrow(draw, x1, y1, x2, y2, color, width=2):
    """Draw a line with an arrowhead."""
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    # Arrowhead
    angle = math.atan2(y2 - y1, x2 - x1)
    arrow_len = 8
    for da in [0.5, -0.5]:
        ax = x2 - arrow_len * math.cos(angle + da)
        ay = y2 - arrow_len * math.sin(angle + da)
        draw.line([(x2, y2), (int(ax), int(ay))], fill=color, width=width)


def _get_font(size=12):
    """Get a font, fallback to default if unavailable."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


def visualize(
    image,
    results,
    det_threshold=0.5,
    kpt_threshold=0.3,
    oih_threshold=0.3,
    show_detection=False,
):
    """Create visualization with detection, pose, and OiH results.

    Hand-object-interaction (OiH) boxes are drawn by default; raw detection
    boxes/labels are only drawn when `show_detection=True`.
    """
    img = image.copy().convert("RGBA")

    labels = results.get("labels")
    boxes = results.get("boxes")
    scores = results.get("scores")
    poses = results.get("poses")
    oihs = results.get("oihs")

    if labels is None:
        return img.convert("RGB")

    num_queries = len(labels)

    # Main draw on image
    draw = ImageDraw.Draw(img)

    # 1. Draw body poses first (below boxes)
    if poses is not None:
        for i in range(num_queries):
            if scores[i] < det_threshold:
                continue
            if int(labels[i]) == 0:  # person
                kpts = poses[i].cpu().numpy()
                draw_skeleton(draw, kpts, vis_threshold=kpt_threshold)

    # 2. Draw OiH interactions
    if oihs is not None:
        for i in range(num_queries):
            if scores[i] < det_threshold:
                continue
            if int(labels[i]) == 0:  # OiH for person
                oih_data = oihs[i].cpu().numpy()
                draw_oih(draw, oih_data, det_score_threshold=oih_threshold)

    # 3. Draw detection boxes on top (optional; off by default)
    if show_detection:
        for i in range(num_queries):
            if scores[i] < det_threshold:
                continue
            draw_detection(draw, boxes[i].tolist(), int(labels[i]), float(scores[i]))

    return img.convert("RGB")


def main():
    parser = argparse.ArgumentParser(
        description="SQ-BHOI inference demo: draws pose and hand-object interactions."
    )
    parser.add_argument("--image", type=str, required=True, help="input image")
    parser.add_argument(
        "--config", type=str, default=None, help="config YAML (default: configs/sqbhoi_r50vd.yml)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="checkpoint (default: weights/checkpoint.pth)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="demo_output.png",
        help="output image (default: demo_output.png)",
    )
    parser.add_argument(
        "--det-threshold", type=float, default=0.5, help="detection score threshold (default: 0.5)"
    )
    parser.add_argument(
        "--kpt-threshold",
        type=float,
        default=0.3,
        help="keypoint visibility threshold (default: 0.3)",
    )
    parser.add_argument(
        "--oih-threshold",
        type=float,
        default=0.3,
        help="hand-object score threshold (default: 0.3)",
    )
    parser.add_argument(
        "--show-detection",
        action="store_true",
        help="also draw object detection boxes and labels",
    )
    parser.add_argument(
        "--size", type=int, nargs=2, default=[640, 640], help="input size H W (default: 640 640)"
    )
    args = parser.parse_args()
    args.config = args.config or os.path.join(REPO_ROOT, "configs/sqbhoi_r50vd.yml")
    args.checkpoint = args.checkpoint or os.path.join(REPO_ROOT, "weights/checkpoint.pth")

    print(f"Loading model from {args.config}...")
    model, postprocessor = load_model(args.config, args.checkpoint)

    print(f"Processing {args.image}...")
    orig_img, img_tensor, orig_size = preprocess_image(args.image, tuple(args.size))

    with torch.no_grad():
        outputs = model(img_tensor)
        results = postprocessor(outputs, orig_size)

    result = results[0]
    # Move tensors to CPU
    for k, v in result.items():
        if v is not None and isinstance(v, torch.Tensor):
            result[k] = v.cpu()

    n_det = (
        (result["scores"] > args.det_threshold).sum().item() if result["scores"] is not None else 0
    )
    print(f"Detected {n_det} objects (threshold={args.det_threshold})")

    vis_img = visualize(
        orig_img,
        result,
        det_threshold=args.det_threshold,
        kpt_threshold=args.kpt_threshold,
        oih_threshold=args.oih_threshold,
        show_detection=args.show_detection,
    )

    vis_img.save(args.output)
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
