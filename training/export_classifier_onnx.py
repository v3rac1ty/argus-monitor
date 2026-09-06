"""Exports the trained classification checkpoint (best.pt) to ONNX and verifies it two ways: (1) a raw onnxruntime pass confirms output shape is exactly ``(1, num_classes)``; (2) the project's own ``ClassifierDetector`` classifies real test images end to end, proving the exported artifact and runtime detector agree.

Ultralytics indexes classification outputs alphabetically by training folder name, not any human-readable config order, and ``ClassifierDetector`` trusts ``cfg.class_names`` to already match it -- so this script builds ``class_names`` from the checkpoint's own ``model.names``, never from config.example.yaml, so the sanity check can't be mislabeled the same way.

Usage: python training/export_classifier_onnx.py --weights runs/train/cls_v1/weights/best.pt --imgsz 512
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_OUT_PATH = DEFAULT_MODELS_DIR / "argus_cls.onnx"
DEFAULT_TEST_DATA_DIR = REPO_ROOT / "datasets" / "argus_cls" / "test"

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

#: Real production severity mapping (only "spaghetti" is catastrophic) so the sanity check
#: exercises actual p_failure wiring, not a stand-in.
_SEVERITY_MAP_NAMES: dict[str, str] = {
    "spaghetti": "catastrophic",
    "cracking": "cosmetic",
    "layer_shifting": "cosmetic",
    "stringing": "cosmetic",
    "warping": "cosmetic",
    # "normal" omitted -- postprocess_classify short-circuits it before severity is consulted.
}


# argparse.Namespace parse_args(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: argparse.Namespace - parsed --weights, --imgsz (default 512), --opset (default 12),
#          --out (default models/argus_cls.onnx), --no-simplify, --test-data (default
#          datasets/argus_cls/test), --samples-per-class (default 1, deterministic: first N
#          images sorted by filename)
# Description: Defines and parses the CLI for exporting and verifying the classifier ONNX model.
# Side Effects: None (argparse may print usage/help and call sys.exit on bad input, but no
#               filesystem or network activity)
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained best.pt")
    parser.add_argument("--imgsz", type=int, default=512, help="Must match the imgsz used for training (default: 512)")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help=f"Final ONNX destination (default: {DEFAULT_OUT_PATH})")
    parser.add_argument("--no-simplify", action="store_true", help="Disable onnxslim simplification (default: simplify=True)")
    parser.add_argument(
        "--test-data",
        type=Path,
        default=DEFAULT_TEST_DATA_DIR,
        help=f"Classification test-split directory used for the ClassifierDetector sanity check "
        f"(default: {DEFAULT_TEST_DATA_DIR})",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=1,
        help="How many real test images per class to run through ClassifierDetector for the sanity "
        "check (default: 1; picked deterministically -- the first N images sorted by filename)",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


# tuple[Path, Optional[dict[int, str]]] export_to_onnx(Path weights, int imgsz, int opset, bool simplify)
# Inputs: Path weights - path to the trained classification best.pt checkpoint
#         int imgsz - export image size, must match the imgsz used for training
#         int opset - ONNX opset version to export with
#         bool simplify - whether to run onnxslim simplification on the exported graph
# Outputs: tuple[Path, Optional[dict[int, str]]] - path Ultralytics wrote the ONNX file to, and
#          the checkpoint's {class_id: class_name} mapping (model.names) in the model's real
#          output-index order, or None if unavailable
# Description: Runs the Ultralytics ONNX export for the classification checkpoint and returns
#              the output path plus the model's own class-name mapping, so the caller can build
#              class_names for the ClassifierDetector sanity check from the actual model rather
#              than assuming any particular order (see module docstring's class-order pitfall).
# Side Effects: Imports ultralytics.YOLO lazily; loads the checkpoint into memory; writes an
#               .onnx file to disk (path chosen by Ultralytics, alongside the weights file).
def export_to_onnx(weights: Path, imgsz: int, opset: int, simplify: bool) -> tuple[Path, Optional[dict[int, str]]]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = getattr(model, "names", None)
    exported_path = model.export(format="onnx", opset=opset, simplify=simplify, dynamic=False, imgsz=imgsz)
    return Path(exported_path), names


# tuple[str, ...] ordered_class_names(Mapping[int, str] names_by_idx)
# Inputs: Mapping[int, str] names_by_idx - {class_id: class_name} mapping from the checkpoint
# Outputs: tuple[str, ...] - class names ordered by index, i.e. exactly what cfg.class_names
#          must be for ClassifierDetector to interpret the ONNX output correctly
# Description: Converts an {idx: name} mapping into an index-ordered tuple -- the model's real
#              output-index order (Ultralytics classification checkpoints index alphabetically
#              from training folder names, not any human-readable config order; see module
#              docstring's class-order pitfall).
# Side Effects: None
def ordered_class_names(names_by_idx: Mapping[int, str]) -> tuple[str, ...]:
    return tuple(names_by_idx[i] for i in sorted(names_by_idx))


# --------------------------------------------------------------------------
# Verification 1: raw onnxruntime output shape
# --------------------------------------------------------------------------


# tuple[int, ...] verify_onnx_output_shape(Path onnx_path, int imgsz, int num_classes)
# Inputs: Path onnx_path - path to the exported classifier ONNX model
#         int imgsz - input image size the model expects (square input)
#         int num_classes - expected number of output classes
# Outputs: tuple[int, ...] - the observed output shape (asserted to equal (1, num_classes))
# Description: Loads onnx_path with onnxruntime, runs one deterministic random input through
#              it, and asserts the output shape is exactly (1, num_classes) -- one probability
#              per class and nothing else baked in.
# Side Effects: Loads the ONNX file into an onnxruntime InferenceSession (CPU); runs one
#               forward pass. RNG is a locally-created np.random.default_rng(seed=1337), so it
#               does not touch global RNG state. Raises AssertionError if the output shape
#               doesn't match (1, num_classes).
def verify_onnx_output_shape(onnx_path: Path, imgsz: int, num_classes: int) -> tuple[int, ...]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]

    rng = np.random.default_rng(seed=1337)
    dummy = rng.random((1, 3, imgsz, imgsz), dtype=np.float32)

    outputs = session.run(None, {input_meta.name: dummy})
    raw = outputs[0]
    shape = tuple(raw.shape)

    expected = (1, num_classes)
    if shape != expected:
        raise AssertionError(
            f"Expected ONNX classifier output shape {expected} (one probability per class), got {shape}. "
            f"The runtime detector (argus.detectors.classifier.ClassifierDetector) expects a raw output "
            f"row of exactly num_classes={num_classes} values."
        )
    return shape


# --------------------------------------------------------------------------
# Verification 2: the project's own ClassifierDetector, on real images
# --------------------------------------------------------------------------


# list[tuple[Path, str]] pick_sample_images(Path test_data_dir, Sequence[str] class_names, int samples_per_class)
# Inputs: Path test_data_dir - classification test-split directory (test_data_dir/<class>/*.jpg)
#         Sequence[str] class_names - class names to sample, in the model's real output order
#         int samples_per_class - how many images to pick per class
# Outputs: list[tuple[Path, str]] - (image_path, class_name) pairs for the sanity check
# Description: Deterministically picks up to samples_per_class images from
#              test_data_dir/<class>/ for each of class_names (sorted by filename, first N).
#              Classes with no test-split subdirectory (or no images) are skipped -- not every
#              class necessarily has test images (see split_report.json's "evaluable" field).
# Side Effects: None (read-only filesystem listing)
def pick_sample_images(test_data_dir: Path, class_names: Sequence[str], samples_per_class: int) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []
    for cname in class_names:
        class_dir = test_data_dir / cname
        if not class_dir.is_dir():
            continue
        images = sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        samples.extend((p, cname) for p in images[:samples_per_class])
    return samples


# DetectorConfig build_sanity_check_config(Path onnx_path, tuple[str, ...] class_names, int input_size)
# Inputs: Path onnx_path - path to the exported classifier ONNX model
#         tuple[str, ...] class_names - class names in the model's real output-index order
#         int input_size - input image size the model expects (square input)
# Outputs: DetectorConfig - config with default_threshold=0.0 (so every non-"normal"
#          prediction always surfaces a Detection) and the real production severity map
#          (_SEVERITY_MAP_NAMES, where only "spaghetti" is catastrophic)
# Description: Builds a DetectorConfig for the ClassifierDetector sanity check: a zero
#              threshold so every non-"normal" prediction is visible regardless of confidence,
#              but the real production severity mapping so p_failure is computed through the
#              actual catastrophic-path wiring rather than a stand-in.
# Side Effects: None (imports argus.config/argus.types; constructs a config object, no I/O)
def build_sanity_check_config(onnx_path: Path, class_names: tuple[str, ...], input_size: int):
    """Threshold 0.0 so every non-"normal" prediction surfaces regardless of confidence,
    but the real production severity map so p_failure runs through actual catastrophic wiring."""
    from argus.config import DetectorConfig
    from argus.types import Severity

    severity = {name: Severity(value) for name, value in _SEVERITY_MAP_NAMES.items()}
    return DetectorConfig(
        kind="classification",
        model_path=str(onnx_path),
        input_size=input_size,
        providers=("CPUExecutionProvider",),
        default_threshold=0.0,
        class_thresholds={},
        severity=severity,
        class_names=class_names,
    )


# list[dict[str, object]] run_classifier_detector_sanity_check(Path onnx_path, tuple[str, ...] class_names, Sequence[tuple[Path, str]] samples, int input_size)
# Inputs: Path onnx_path - path to the exported classifier ONNX model
#         tuple[str, ...] class_names - class names in the model's real output-index order
#         Sequence[tuple[Path, str]] samples - (image_path, true_class) pairs to classify
#         int input_size - input image size the model expects (square input)
# Outputs: list[dict[str, object]] - one result dict per sample: path, true_class,
#          predicted_class, confidence (or None if "normal"), severity, p_failure, correct
#          (bool), and inference_ms
# Description: Loads onnx_path through the project's own ClassifierDetector (the actual runtime
#              code, not a reimplementation) and classifies each sample end to end
#              (preprocess -> onnxruntime session -> postprocess), proving the trained artifact
#              and the runtime detector agree with each other.
# Side Effects: Reads each sample image from disk via cv2.imread; constructs and closes an
#               onnxruntime-backed ClassifierDetector session (detector.close() in a finally
#               block); raises IOError if an image fails to load; raises AssertionError if any
#               confidence or p_failure value falls outside [0, 1].
def run_classifier_detector_sanity_check(
    onnx_path: Path, class_names: tuple[str, ...], samples: Sequence[tuple[Path, str]], input_size: int
) -> list[dict[str, object]]:
    import cv2

    from argus.detectors.classifier import ClassifierDetector
    from argus.types import Frame

    cfg = build_sanity_check_config(onnx_path, class_names, input_size)
    detector = ClassifierDetector(cfg)

    results: list[dict[str, object]] = []
    try:
        for seq, (path, true_class) in enumerate(samples):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise IOError(f"Failed to read test image: {path}")
            frame = Frame(image=image, timestamp=0.0, seq=seq)
            result = detector.infer(frame)

            if not (0.0 <= result.p_failure <= 1.0):
                raise AssertionError(f"p_failure out of [0, 1] range for {path}: {result.p_failure!r}")

            if result.detections:
                det = result.detections[0]
                if not (0.0 <= det.confidence <= 1.0):
                    raise AssertionError(f"Detection confidence out of [0, 1] range for {path}: {det.confidence!r}")
                predicted_name: Optional[str] = det.class_name
                confidence: Optional[float] = det.confidence
                severity = det.severity.value
            else:
                # Only possible when the argmax prediction is "normal" at
                # threshold 0.0 (nothing else could fail to clear a 0.0
                # threshold) -- see postprocess_classify.
                predicted_name = "normal"
                confidence = None
                severity = None

            results.append(
                {
                    "path": str(path),
                    "true_class": true_class,
                    "predicted_class": predicted_name,
                    "confidence": confidence,
                    "severity": severity,
                    "p_failure": result.p_failure,
                    "correct": predicted_name == true_class,
                    "inference_ms": result.inference_ms,
                }
            )
    finally:
        detector.close()
    return results


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


# None main(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: None
# Description: CLI entry point. Exports the classification checkpoint to ONNX, derives
#              class_names from the checkpoint's own output-index order, copies the exported
#              file to --out, verifies the raw output shape, runs the ClassifierDetector
#              sanity check against real test images (skipped with a warning if none are
#              found), and prints a combined verification report.
# Side Effects: Raises FileNotFoundError if --weights doesn't exist; raises RuntimeError if the
#               exported model has no `names` attribute; writes an ONNX file to disk via
#               Ultralytics export; creates --out's parent directory and copies the exported
#               file there (shutil.copy2); runs onnxruntime verification and (if test images
#               exist) the real ClassifierDetector sanity check, which reads test images from
#               disk; prints export progress and a verification report to stdout.
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    print(
        f"[export_classifier] Exporting '{args.weights}' -> ONNX (opset={args.opset}, "
        f"simplify={not args.no_simplify}, dynamic=False, imgsz={args.imgsz}) ..."
    )
    exported_path, model_names = export_to_onnx(args.weights, args.imgsz, args.opset, simplify=not args.no_simplify)
    print(f"[export_classifier] Ultralytics wrote: {exported_path}")

    if not model_names:
        raise RuntimeError(
            "Exported model has no `names` attribute -- cannot determine class count/order. "
            "This shouldn't happen for a classification checkpoint trained via training/train.py."
        )
    class_names = ordered_class_names(model_names)
    num_classes = len(class_names)
    print(f"[export_classifier] Model class order (real output-index order): {list(class_names)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported_path, args.out)
    print(f"[export_classifier] Copied to: {args.out}")

    print("[export_classifier] Verifying output shape with a raw onnxruntime pass ...")
    shape = verify_onnx_output_shape(args.out, args.imgsz, num_classes)
    print(f"[export_classifier] Output shape: {shape}  (expected (1, {num_classes})) -- OK")

    print(
        f"[export_classifier] Running the project's own ClassifierDetector against real test images "
        f"from '{args.test_data}' ..."
    )
    samples = pick_sample_images(args.test_data, class_names, args.samples_per_class)
    if not samples:
        print(
            f"[export_classifier] WARNING: no test images found under '{args.test_data}' for any of "
            f"{list(class_names)} -- skipping the ClassifierDetector sanity check."
        )
        sanity_results: list[dict[str, object]] = []
    else:
        sanity_results = run_classifier_detector_sanity_check(args.out, class_names, samples, args.imgsz)

    print()
    print("=" * 88)
    print("ONNX CLASSIFIER EXPORT VERIFICATION")
    print("=" * 88)
    print(f"Model path:   {args.out}")
    print(f"Input size:   {args.imgsz}x{args.imgsz}")
    print(f"Num classes:  {num_classes}")
    print(f"Output shape: {shape}  (raw onnxruntime pass, random input) -- matches (1, num_classes): OK")
    print()
    if sanity_results:
        print(f"ClassifierDetector sanity check -- {len(sanity_results)} real test image(s) via the project's own runtime code:")
        header = f"{'true_class':<18}{'predicted':<18}{'confidence':>12}{'severity':>14}{'p_failure':>12}{'match':>8}{'ms':>8}"
        print(header)
        print("-" * len(header))
        num_correct = 0
        for r in sanity_results:
            conf_str = f"{r['confidence']:.4f}" if r["confidence"] is not None else "--"
            sev_str = r["severity"] if r["severity"] is not None else "--"
            match = "yes" if r["correct"] else "no"
            if r["correct"]:
                num_correct += 1
            print(
                f"{r['true_class']:<18}{str(r['predicted_class']):<18}{conf_str:>12}{str(sev_str):>14}"
                f"{r['p_failure']:>12.4f}{match:>8}{r['inference_ms']:>8.1f}"
            )
        print()
        print(
            f"{num_correct}/{len(sanity_results)} correct on this tiny sample (NOT a real accuracy estimate -- "
            f"see training/evaluate_classifier.py for that; this is purely a smoke check that the exported "
            f".onnx file and ClassifierDetector agree with each other and produce probabilities in [0, 1])."
        )
        print()
        print("All confidence and p_failure values were within [0, 1] -- SANE. (An out-of-range value would")
        print("have raised AssertionError above rather than reaching this summary.)")
    print("=" * 88)


if __name__ == "__main__":
    main()
