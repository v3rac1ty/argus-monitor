"""Export the trained classification checkpoint (best.pt) to ONNX and verify
it two independent ways:

  1. A raw onnxruntime pass on a random input confirms the exported model's
     output shape is exactly ``(1, num_classes)`` -- one probability per
     class, nothing else baked in.
  2. The project's OWN runtime code, `argus.detectors.classifier.
     ClassifierDetector`, loads the exported .onnx file and classifies a
     handful of REAL test images end to end (preprocess -> onnxruntime
     session -> postprocess). This is the check that actually matters: it
     proves the trained artifact and the runtime detector agree with each
     other, not just that the .onnx file loads in isolation.

Class-order pitfall (see training/evaluate_classifier.py's module
docstring for the full story, and its measured
`check_class_order_matches_model` finding): Ultralytics classification
checkpoints index their output ALPHABETICALLY from the training folder
names (``['cracking', 'layer_shifting', 'normal', 'spaghetti', 'stringing',
'warping']``), NOT the "normal, spaghetti, cracking, layer_shifting,
stringing, warping" order config.example.yaml's commented-out
classification block lists for human readability.
`ClassifierDetector` has no way to know the model's true index order on its
own -- it *trusts* `cfg.class_names` to already match it. This script
therefore builds `class_names` from the loaded checkpoint's own `model.
names` (index-sorted), NEVER from config.example.yaml's human-readable
list, so step 2's sanity check can't itself be mislabeled the same way a
naive config wiring would be.

Usage:
    python training/export_classifier_onnx.py --weights runs/train/cls_v1/weights/best.pt --imgsz 512
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

#: The real production severity mapping from config.example.yaml's
#: commented-out classification block: only "spaghetti" is catastrophic
#: (can drive an automated pause/cancel); everything else is cosmetic.
#: Used for the ClassifierDetector sanity check so it exercises the actual
#: p_failure wiring, not a stripped-down stand-in.
_SEVERITY_MAP_NAMES: dict[str, str] = {
    "spaghetti": "catastrophic",
    "cracking": "cosmetic",
    "layer_shifting": "cosmetic",
    "stringing": "cosmetic",
    "warping": "cosmetic",
    # "normal" deliberately omitted -- postprocess_classify short-circuits
    # any "normal" prediction to zero detections before severity is ever
    # consulted (see argus.detectors.classifier module docstring).
}


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


def export_to_onnx(weights: Path, imgsz: int, opset: int, simplify: bool) -> tuple[Path, Optional[dict[int, str]]]:
    """Run the Ultralytics ONNX export and return `(exported_path, names)`.

    `names` is the `{class_id: class_name}` mapping baked into the
    checkpoint (`model.names`) -- the model's REAL output-index order (see
    module docstring). Returning it lets the caller build `class_names` for
    the ClassifierDetector sanity check from the actual model being
    exported instead of assuming any particular order.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = getattr(model, "names", None)
    exported_path = model.export(format="onnx", opset=opset, simplify=simplify, dynamic=False, imgsz=imgsz)
    return Path(exported_path), names


def ordered_class_names(names_by_idx: Mapping[int, str]) -> tuple[str, ...]:
    """`{idx: name}` -> a tuple ordered by index -- the model's real
    output-index order, i.e. exactly what `cfg.class_names` must be for
    `ClassifierDetector` to interpret the ONNX output correctly."""
    return tuple(names_by_idx[i] for i in sorted(names_by_idx))


# --------------------------------------------------------------------------
# Verification 1: raw onnxruntime output shape
# --------------------------------------------------------------------------


def verify_onnx_output_shape(onnx_path: Path, imgsz: int, num_classes: int) -> tuple[int, ...]:
    """Load `onnx_path` with onnxruntime, run one random input through it,
    and assert the output shape is exactly `(1, num_classes)` -- one
    probability per class and nothing else. Raises `AssertionError`
    otherwise. Returns the observed shape."""
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


def pick_sample_images(test_data_dir: Path, class_names: Sequence[str], samples_per_class: int) -> list[tuple[Path, str]]:
    """Deterministically pick up to `samples_per_class` images from
    `test_data_dir/<class>/` for each of `class_names` (sorted by filename,
    first N). Classes with no test-split subdirectory (or no images) are
    skipped -- not every class necessarily has test images (see
    datasets/argus_cls/split_report.json's `evaluable` field)."""
    samples: list[tuple[Path, str]] = []
    for cname in class_names:
        class_dir = test_data_dir / cname
        if not class_dir.is_dir():
            continue
        images = sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        samples.extend((p, cname) for p in images[:samples_per_class])
    return samples


def build_sanity_check_config(onnx_path: Path, class_names: tuple[str, ...], input_size: int):
    """Build a `DetectorConfig` for the sanity check: threshold 0.0 for
    every class (so every non-"normal" prediction always surfaces a
    `Detection`, regardless of confidence -- we want to SEE the raw
    argmax/confidence for each sample, not have some silently suppressed),
    but the REAL production severity mapping (`_SEVERITY_MAP_NAMES`) so
    `p_failure` is computed through the actual catastrophic-path wiring,
    not a stand-in."""
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


def run_classifier_detector_sanity_check(
    onnx_path: Path, class_names: tuple[str, ...], samples: Sequence[tuple[Path, str]], input_size: int
) -> list[dict[str, object]]:
    """Load `onnx_path` through the project's own `ClassifierDetector` and
    classify each of `samples`. Returns one result dict per sample with the
    true class (from its test-split subdirectory), the predicted class and
    confidence (or `None` if the prediction was "normal", which
    `postprocess_classify` always excludes), and `p_failure`.

    Raises `AssertionError` if any confidence or `p_failure` value falls
    outside `[0, 1]` -- the actual "sane probabilities" check.
    """
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
