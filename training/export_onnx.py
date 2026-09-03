"""Export a trained YOLO checkpoint (best.pt) to ONNX and verify the
resulting model has an output layout the runtime detector can actually
parse.

src/argus/detectors/onnx_yolo.py supports two output contracts -- legacy
YOLOv8 (raw, NMS-pending predictions shaped `(1, 4 + num_classes,
num_anchors)` or its transpose) and YOLO26's end-to-end/NMS-free layout
(`(1, max_det, 6)`, each row already decoded as `[x1, y1, x2, y2, confidence,
class_id]`) -- and exposes `detect_layout()` to tell them apart from a raw
output tensor. This script reuses that *exact* helper (rather than
re-deriving the shape rule here) so verification can never silently drift out
of sync with what the runtime does: it runs a real onnxruntime session
against the freshly exported model with a random input, feeds the raw output
through the same `detect_layout()` the runtime calls, and fails loudly if
neither contract matches.

Usage:
    python training/export_onnx.py --weights runs/train/argus_yolov8n/weights/best.pt
    python training/export_onnx.py --weights runs/train/yolo26s_v1/weights/best.pt --imgsz 512
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from argus.detectors.onnx_yolo import detect_layout

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_OUT_PATH = DEFAULT_MODELS_DIR / "argus.onnx"

#: Used only when num_classes can't be derived from either --nc or the
#: exported model's own `names` (see `resolve_num_classes`).
DEFAULT_NUM_CLASSES = 5  # error extrusion, spaghetti, stringing, warping, zits


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained best.pt")
    parser.add_argument("--imgsz", type=int, default=640, help="Must match the imgsz used for training (default: 640)")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help=f"Final ONNX destination (default: {DEFAULT_OUT_PATH})")
    parser.add_argument("--no-simplify", action="store_true", help="Disable onnx-simplifier (default: simplify=True)")
    parser.add_argument(
        "--nc",
        type=int,
        default=None,
        help=(
            "Number of classes the model was trained with. Defaults to reading "
            f"it off the loaded checkpoint's own `names`, falling back to "
            f"{DEFAULT_NUM_CLASSES} only if that can't be determined."
        ),
    )
    return parser.parse_args(argv)


def export_to_onnx(weights: Path, imgsz: int, opset: int, simplify: bool) -> tuple[Path, Optional[dict[int, str]]]:
    """Run the Ultralytics ONNX export and return `(exported_path, names)`.

    `names` is the `{class_id: class_name}` mapping baked into the checkpoint
    -- Ultralytics exposes this as `model.names` on any loaded `YOLO` model,
    YOLOv8 or YOLO26 alike -- or `None` if the loaded model has no such
    attribute. Returning it lets the caller derive `num_classes` from the
    actual model being exported instead of assuming a hardcoded constant.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = getattr(model, "names", None)
    exported_path = model.export(format="onnx", opset=opset, simplify=simplify, dynamic=False, imgsz=imgsz)
    return Path(exported_path), names


def resolve_num_classes(cli_nc: Optional[int], model_names: Optional[dict[int, str]]) -> tuple[int, str]:
    """Pick `num_classes` for verification, preferring (in order): an
    explicit `--nc`, the exported model's own `names`, and only then the
    hardcoded default. Returns `(num_classes, source)` where `source`
    describes where the value came from, for the printed summary."""
    if cli_nc is not None:
        return cli_nc, f"--nc={cli_nc}"
    if model_names:
        names_list = [model_names[i] for i in sorted(model_names)]
        return len(model_names), f"model names {names_list}"
    return DEFAULT_NUM_CLASSES, f"default ({DEFAULT_NUM_CLASSES}; could not read class names off the exported model)"


def classify_output_shape(raw: np.ndarray, num_classes: int) -> tuple[tuple[int, ...], str]:
    """Given a raw ONNX output array, return `(shape, layout)` where `layout`
    is exactly what `OnnxYoloDetector` (src/argus/detectors/onnx_yolo.py)
    would resolve to for this same array and `num_classes` -- it's computed
    by that module's own `detect_layout()`, not a reimplementation of the
    rule. Raises `AssertionError` (wrapping `detect_layout`'s `ValueError`)
    if `raw` matches neither the YOLOv8 layout nor the YOLO26 end-to-end
    layout. Pure/no I/O, so it's testable with synthetic arrays."""
    shape = tuple(raw.shape)
    try:
        layout = detect_layout(raw, num_classes)
    except ValueError as exc:
        raise AssertionError(
            f"ONNX output shape {shape} matches neither layout the runtime detector "
            f"(src/argus/detectors/onnx_yolo.py) supports for num_classes={num_classes}: "
            f"YOLOv8 '(1, {4 + num_classes}, N)' (or its transpose '(1, N, {4 + num_classes})') "
            f"or YOLO26 end-to-end '(1, N, 6)'. {exc}"
        ) from exc
    return shape, layout


def verify_onnx(onnx_path: Path, imgsz: int, num_classes: int) -> tuple[tuple[int, ...], str]:
    """Load `onnx_path` with onnxruntime, run one random input through it,
    and return `(shape, layout)` via `classify_output_shape`. Raises
    `AssertionError` if the output doesn't match a layout the runtime
    detector can parse."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name

    rng = np.random.default_rng(seed=1337)
    dummy = rng.random((1, 3, imgsz, imgsz), dtype=np.float32)

    outputs = session.run(None, {input_name: dummy})
    raw = outputs[0]

    return classify_output_shape(raw, num_classes)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    print(f"[export] Exporting '{args.weights}' -> ONNX (opset={args.opset}, simplify={not args.no_simplify}, "
          f"dynamic=False, imgsz={args.imgsz}) ...")
    exported_path, model_names = export_to_onnx(args.weights, args.imgsz, args.opset, simplify=not args.no_simplify)
    print(f"[export] Ultralytics wrote: {exported_path}")

    num_classes, nc_source = resolve_num_classes(args.nc, model_names)
    print(f"[export] num_classes={num_classes} (source: {nc_source})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported_path, args.out)
    print(f"[export] Copied to: {args.out}")

    print("[export] Verifying exported model with onnxruntime ...")
    shape, layout = verify_onnx(args.out, args.imgsz, num_classes)

    print()
    print("=" * 72)
    print("ONNX EXPORT VERIFICATION")
    print("=" * 72)
    print(f"Model path:      {args.out}")
    print(f"Input size:      {args.imgsz}x{args.imgsz}  (must match detector.input_size in config, currently used by the runtime)")
    print(f"Output shape:    {shape}")
    print(f"Num classes:     {num_classes}  (source: {nc_source})")
    print(f"Detected layout: {layout!r}")
    if layout == "yolov8":
        print(
            f"  -> YOLOv8-style raw predictions (1, {4 + num_classes}, N) [or its transpose]; "
            "the runtime detector will run its own NMS on this output."
        )
    else:
        print(
            "  -> YOLO26 end-to-end/NMS-free output (1, N, 6) = [x1, y1, x2, y2, conf, cls]; "
            "the runtime detector skips NMS for this layout (the model already deduped it)."
        )
    print(f"If this model is ever loaded with detector.layout != 'auto', set it to: {layout!r}")
    print("=" * 72)


if __name__ == "__main__":
    main()
