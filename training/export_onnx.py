"""Exports a trained YOLO checkpoint (best.pt) to ONNX and verifies the output layout is one the runtime detector can parse: legacy YOLOv8 raw predictions ``(1, 4+num_classes, N)`` (or its transpose) or YOLO26's end-to-end/NMS-free ``(1, N, 6)`` rows. Reuses the runtime's own ``detect_layout()`` (src/argus/detectors/onnx_yolo.py) rather than re-deriving the shape rule, so verification can't drift out of sync with the runtime.

Usage: python training/export_onnx.py --weights runs/train/argus_yolov8n/weights/best.pt [--imgsz 640]
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


# argparse.Namespace parse_args(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: argparse.Namespace - parsed --weights, --imgsz (default 640), --opset (default 12),
#          --out (default models/argus.onnx), --no-simplify, and --nc (default None, resolved
#          later by resolve_num_classes)
# Description: Defines and parses the CLI for exporting a trained checkpoint to ONNX.
# Side Effects: None (argparse may print usage/help and call sys.exit on bad input, but no
#               filesystem or network activity)
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


# tuple[Path, Optional[dict[int, str]]] export_to_onnx(Path weights, int imgsz, int opset, bool simplify)
# Inputs: Path weights - path to the trained best.pt checkpoint
#         int imgsz - export image size, must match the imgsz used for training
#         int opset - ONNX opset version to export with
#         bool simplify - whether to run onnx-simplifier on the exported graph
# Outputs: tuple[Path, Optional[dict[int, str]]] - path Ultralytics wrote the ONNX file to, and
#          the {class_id: class_name} mapping baked into the checkpoint (model.names), or None
#          if the loaded model has no such attribute
# Description: Runs the Ultralytics ONNX export for the given checkpoint and returns both the
#              output path and the model's class-name mapping so the caller can derive
#              num_classes from the actual model rather than a hardcoded constant.
# Side Effects: Imports ultralytics.YOLO lazily; loads the checkpoint into memory; writes an
#               .onnx file to disk (path chosen by Ultralytics, alongside the weights file).
def export_to_onnx(weights: Path, imgsz: int, opset: int, simplify: bool) -> tuple[Path, Optional[dict[int, str]]]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = getattr(model, "names", None)
    exported_path = model.export(format="onnx", opset=opset, simplify=simplify, dynamic=False, imgsz=imgsz)
    return Path(exported_path), names


# tuple[int, str] resolve_num_classes(Optional[int] cli_nc, Optional[dict[int, str]] model_names)
# Inputs: Optional[int] cli_nc - explicit --nc value from the CLI, or None
#         Optional[dict[int, str]] model_names - the exported model's {class_id: class_name}
#         mapping, or None if unavailable
# Outputs: tuple[int, str] - (num_classes, source) where source describes where the value came
#          from (--nc, the model's own names, or the hardcoded DEFAULT_NUM_CLASSES) for the
#          printed summary
# Description: Picks num_classes for verification, preferring an explicit --nc, then the
#              exported model's own names, and only then the hardcoded default.
# Side Effects: None
def resolve_num_classes(cli_nc: Optional[int], model_names: Optional[dict[int, str]]) -> tuple[int, str]:
    if cli_nc is not None:
        return cli_nc, f"--nc={cli_nc}"
    if model_names:
        names_list = [model_names[i] for i in sorted(model_names)]
        return len(model_names), f"model names {names_list}"
    return DEFAULT_NUM_CLASSES, f"default ({DEFAULT_NUM_CLASSES}; could not read class names off the exported model)"


# tuple[tuple[int, ...], str] classify_output_shape(np.ndarray raw, int num_classes)
# Inputs: np.ndarray raw - raw ONNX model output array from an inference run
#         int num_classes - number of classes the model was trained with
# Outputs: tuple[tuple[int, ...], str] - (shape, layout) where layout is "yolov8" or the
#          YOLO26 end-to-end layout name, exactly as src/argus/detectors/onnx_yolo.py's
#          detect_layout() would resolve it for this array
# Description: Classifies a raw ONNX output array's layout by delegating to the runtime
#              detector's own detect_layout() (not a reimplementation), so export-time
#              verification can never silently drift out of sync with what the runtime does.
#              Pure/no I/O, so it's testable with synthetic arrays.
# Side Effects: Raises AssertionError (wrapping detect_layout's ValueError) if raw matches
#               neither the YOLOv8 layout nor the YOLO26 end-to-end layout. No I/O.
def classify_output_shape(raw: np.ndarray, num_classes: int) -> tuple[tuple[int, ...], str]:
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


# tuple[tuple[int, ...], str] verify_onnx(Path onnx_path, int imgsz, int num_classes)
# Inputs: Path onnx_path - path to the exported ONNX model file
#         int imgsz - input image size the model expects (square input)
#         int num_classes - number of classes the model was trained with
# Outputs: tuple[tuple[int, ...], str] - (shape, layout) of the raw output, via
#          classify_output_shape
# Description: Loads onnx_path with onnxruntime (CPU provider), runs one deterministic random
#              input through it, and classifies the resulting output layout.
# Side Effects: Loads the ONNX file into an onnxruntime InferenceSession (CPU); runs one
#               forward pass. RNG is a locally-created np.random.default_rng(seed=1337), so it
#               does not touch global RNG state. Raises AssertionError if the output doesn't
#               match a layout the runtime detector can parse.
def verify_onnx(onnx_path: Path, imgsz: int, num_classes: int) -> tuple[tuple[int, ...], str]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name

    rng = np.random.default_rng(seed=1337)
    dummy = rng.random((1, 3, imgsz, imgsz), dtype=np.float32)

    outputs = session.run(None, {input_name: dummy})
    raw = outputs[0]

    return classify_output_shape(raw, num_classes)


# None main(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: None
# Description: CLI entry point. Exports the given checkpoint to ONNX, resolves num_classes,
#              copies the exported file to the final --out destination, verifies the output
#              layout with onnxruntime, and prints a verification summary including which
#              detector.layout setting to use if not left on "auto".
# Side Effects: Raises FileNotFoundError if --weights doesn't exist; writes an ONNX file to
#               disk via Ultralytics export; creates --out's parent directory and copies the
#               exported file there (shutil.copy2); runs a verification inference pass via
#               onnxruntime; prints export progress and a verification report to stdout.
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
