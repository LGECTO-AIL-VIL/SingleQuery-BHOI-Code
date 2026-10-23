"""
ONNX Runtime inference & visualization for SQ-BHOI.

Modes:
  1. Single image         : --image demo/000000409475.jpg
  2. Folder of images     : --image-dir demo
  3. Live webcam          : --camera 0

Usage (from any directory):
    python3 tools/export_onnx.py --check          # once: writes weights/checkpoint.onnx
    python3 tools/onnx_live_inference.py --image-dir demo --output-dir demo_out
    python3 tools/onnx_live_inference.py --camera 0
"""

import argparse
import ctypes
import glob
import importlib.util
import os
import sys
import time

import cv2
import numpy as np
import torch

# onnxruntime's CUDAExecutionProvider dlopen()s cuDNN/cuBLAS/etc. and only
# searches LD_LIBRARY_PATH. When those libs come from pip packages
# (nvidia-cudnn-cu12, ...) the loader can't find them ("libcudnn.so.9: cannot
# open shared object file"), so pre-load them before importing onnxruntime.
_nvidia = importlib.util.find_spec("nvidia")
for _base in _nvidia.submodule_search_locations if _nvidia else []:
    for _pkg in ("cudnn", "cublas", "cuda_runtime", "cufft", "curand", "cusparse", "nvjitlink"):
        _lib_dir = os.path.join(_base, _pkg, "lib")
        for _so in sorted(os.listdir(_lib_dir)) if os.path.isdir(_lib_dir) else []:
            if ".so" in _so:
                try:
                    ctypes.CDLL(os.path.join(_lib_dir, _so))
                except OSError:
                    pass

import onnxruntime as ort
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from demo import visualize  # tools/demo.py


def build_session(onnx_path, use_cuda=True):
    providers = []
    if use_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    sess = ort.InferenceSession(onnx_path, providers=providers)

    active_providers = sess.get_providers()
    print(f"[ONNX Runtime] available providers : {ort.get_available_providers()}")
    print(f"[ONNX Runtime] requested providers  : {providers}")
    print(f"[ONNX Runtime] ACTIVE provider      : {active_providers[0]}")
    if use_cuda and active_providers[0] != "CUDAExecutionProvider":
        print(
            "[ONNX Runtime] WARNING: CUDA was requested but is NOT active; "
            "falling back to CPU. Check CUDA/cuDNN installation."
        )
    return sess


def preprocess(img_rgb, size=(640, 640)):
    """img_rgb: HxWx3 uint8 RGB numpy array. Returns (input_tensor(1,3,H,W) float32
    normalized to [-1,1], orig_size[[W,H]] float32)."""
    orig_h, orig_w = img_rgb.shape[:2]
    resized = cv2.resize(img_rgb, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    data = resized.astype(np.float32) / 255.0
    data = (data - 0.5) / 0.5
    data = data.transpose(2, 0, 1)[None]  # (1,3,H,W)
    orig_size = np.array([[orig_w, orig_h]], dtype=np.float32)
    return data, orig_size


def run_onnx(sess, data, orig_size):
    outputs = sess.run(None, {"images": data, "orig_target_sizes": orig_size})
    names = [o.name for o in sess.get_outputs()]
    out = dict(zip(names, outputs))

    result = {}
    for key in ("labels", "boxes", "scores", "poses", "oihs"):
        val = out.get(key)
        result[key] = torch.from_numpy(val[0]) if val is not None else None
    return result


def process_frame(
    sess, frame_bgr, size, det_thr, kpt_thr, oih_thr, scale=1.0, show_detection=False
):
    """frame_bgr: HxWx3 uint8 BGR (as returned by cv2). Returns visualized BGR frame."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    data, orig_size = preprocess(frame_rgb, size)

    t0 = time.time()
    result = run_onnx(sess, data, orig_size)
    infer_ms = (time.time() - t0) * 1000

    pil_img = Image.fromarray(frame_rgb)
    vis_img = visualize(
        pil_img,
        result,
        det_threshold=det_thr,
        kpt_threshold=kpt_thr,
        oih_threshold=oih_thr,
        show_detection=show_detection,
    )
    vis_bgr = cv2.cvtColor(np.array(vis_img), cv2.COLOR_RGB2BGR)

    if scale != 1.0:
        vis_bgr = cv2.resize(vis_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    fps = 1000.0 / infer_ms if infer_ms > 0 else 0.0
    provider = sess.get_providers()[0].replace("ExecutionProvider", "")
    cv2.putText(
        vis_bgr,
        f"{fps:.1f} FPS ({infer_ms:.1f} ms) [{provider}]",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    return vis_bgr


def run_on_images(
    sess,
    image_paths,
    output_dir,
    size,
    det_thr,
    kpt_thr,
    oih_thr,
    scale=1.0,
    show_detection=False,
):
    os.makedirs(output_dir, exist_ok=True)
    for path in image_paths:
        frame_bgr = cv2.imread(path)
        if frame_bgr is None:
            print(f"Skip unreadable image: {path}")
            continue
        vis_bgr = process_frame(
            sess,
            frame_bgr,
            size,
            det_thr,
            kpt_thr,
            oih_thr,
            scale=scale,
            show_detection=show_detection,
        )
        out_path = os.path.join(output_dir, os.path.basename(path))
        cv2.imwrite(out_path, vis_bgr)
        print(f"Saved {out_path}")


def run_on_camera(
    sess,
    camera_index,
    size,
    det_thr,
    kpt_thr,
    oih_thr,
    window_name="SQ-BHOI Live",
    headless=False,
    save_video=None,
    max_frames=None,
    scale=1.0,
    show_detection=False,
):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    # Auto-fallback to headless mode when no GUI/display is available (e.g.
    # running on a remote server without X11), otherwise cv2.imshow() raises.
    if not headless and not os.environ.get("DISPLAY"):
        print("No DISPLAY found; running in headless mode (use --save-video to record).")
        headless = True

    writer = None
    frame_count = 0
    print('Press "q" to quit.' if not headless else "Headless mode: press Ctrl+C to stop.")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            vis_bgr = process_frame(
                sess,
                frame_bgr,
                size,
                det_thr,
                kpt_thr,
                oih_thr,
                scale=scale,
                show_detection=show_detection,
            )

            if save_video:
                if writer is None:
                    h, w = vis_bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(save_video, fourcc, 20.0, (w, h))
                writer.write(vis_bgr)

            if not headless:
                cv2.imshow(window_name, vis_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Saved video to {save_video}")
        if not headless:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="SQ-BHOI inference with ONNX Runtime (images or webcam)."
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default=None,
        help="exported ONNX model (default: weights/checkpoint.onnx)",
    )
    parser.add_argument("--image", type=str, default=None, help="single input image")
    parser.add_argument("--image-dir", type=str, default=None, help="folder of input images")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="demo_out",
        help="folder for visualized images (default: demo_out)",
    )
    parser.add_argument(
        "--camera", type=int, default=None, help="camera device index for live inference"
    )
    parser.add_argument("--headless", action="store_true", help="do not open a display window")
    parser.add_argument(
        "--save-video", type=str, default=None, help="save live output to this video file"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="stop live inference after N frames"
    )
    parser.add_argument(
        "--size", type=int, nargs=2, default=[640, 640], help="input size H W (default: 640 640)"
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="output display scale factor (default: 1.0)"
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
    parser.add_argument("--cpu", action="store_true", help="force the CPU execution provider")
    parser.add_argument(
        "--show-detection", action="store_true", help="also draw object detection boxes and labels"
    )
    args = parser.parse_args()
    args.onnx = args.onnx or os.path.join(REPO_ROOT, "weights/checkpoint.onnx")

    sess = build_session(args.onnx, use_cuda=not args.cpu)
    size = tuple(args.size)

    if args.camera is not None:
        run_on_camera(
            sess,
            args.camera,
            size,
            args.det_threshold,
            args.kpt_threshold,
            args.oih_threshold,
            headless=args.headless,
            save_video=args.save_video,
            max_frames=args.max_frames,
            scale=args.scale,
            show_detection=args.show_detection,
        )
        return

    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        image_paths = sorted(
            glob.glob(os.path.join(args.image_dir, "*.jpg"))
            + glob.glob(os.path.join(args.image_dir, "*.jpeg"))
            + glob.glob(os.path.join(args.image_dir, "*.png"))
        )
    else:
        parser.error("One of --image, --image-dir, or --camera must be given.")
        return

    run_on_images(
        sess,
        image_paths,
        args.output_dir,
        size,
        args.det_threshold,
        args.kpt_threshold,
        args.oih_threshold,
        scale=args.scale,
        show_detection=args.show_detection,
    )


if __name__ == "__main__":
    main()
