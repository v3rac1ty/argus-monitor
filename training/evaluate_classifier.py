"""Evaluates a trained classifier on ``--split`` (default test): overall accuracy and per-class P/R/F1, the confusion matrix, a binary normal-vs-defect collapse with false-positive rate, the real runtime ``p_failure`` threshold sweep for the catastrophic class (``--catastrophic-class``, default ``spaghetti``; the only class allowed to auto-pause a print), and a source-confound diagnostic splitting catastrophic-class recall by dataset origin (this dataset's ``normal`` is 100% Hugging Face while other defects are ~100% FDM; ``spaghetti`` is the one class mixed across both).

Three inference backends are supported, selected automatically from ``--weights``' suffix (``select_backend``): an Ultralytics ``.pt`` checkpoint (the original 6-class model), a ``.onnx`` export (e.g. the binary ``failure``/``normal`` model trained outside Ultralytics via tinygrad), or a Hailo ``.hef`` (the same binary model, quantized and compiled for the AI HAT+ -- see docs/hailo-deployment.md). The ONNX backend reuses the real production pre/post-processing from ``argus.detectors.classifier`` (``preprocess_classify``, ``probabilities_from_output``, ``class_names_from_onnx_metadata``) so train/eval/deploy can never silently diverge, and it fails loudly if the ONNX file carries no class-name metadata -- unlike ``ClassifierDetector`` this script has no ``DetectorConfig.class_names`` to fall back on. The Hailo backend reuses ``argus.detectors.hailo.HailoDetector.predict_proba`` the same way, for the same reason -- but a compiled HEF carries no class-name metadata AT ALL (not even the fallback-worthy kind ONNX has), so ``--class-names`` is REQUIRED for it. Reporting class order and the catastrophic-class index are both resolved from the model's own class names (``resolve_report_class_order``, ``resolve_catastrophic_class_index``), so the same code path handles the curated 6-class order and the binary model's ``("failure", "normal")`` order without assuming either.

``--min-recall`` rejects the vacuous "precision=1.0 at ~zero recall" convention: a threshold so high the model almost never fires trivially has no false positives and no real value. Ultralytics assigns class indices alphabetically, not the human-readable order used for reporting here -- a mismatch is flagged loudly since ``ClassifierDetector`` trusts ``cfg.class_names`` to match the model's real order.

Usage: python training/evaluate_classifier.py --weights <best.pt> [--split test] [--target-precision 0.95]
       python training/evaluate_classifier.py --weights <model.onnx> --imgsz 320 --catastrophic-class failure
       python training/evaluate_classifier.py --weights models/argus_bin.hef --imgsz 320 \\
           --catastrophic-class failure --class-names failure,normal
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "train" / "cls_v1" / "weights" / "best.pt"
DEFAULT_DATA_DIR = REPO_ROOT / "datasets" / "argus_cls"
DEFAULT_OUT_PATH = REPO_ROOT / "runs" / "classifier_evaluation.json"

#: Human-readable class order for reporting only -- NOT assumed to be the model's actual
#: output-index order (Ultralytics orders alphabetically; see check_class_order_matches_model).
REPORT_CLASS_ORDER: tuple[str, ...] = ("normal", "spaghetti", "cracking", "layer_shifting", "stringing", "warping")

NORMAL_CLASS_NAME = "normal"
#: Only "spaghetti" is CATASTROPHIC per config.example.yaml's severity
#: mapping; every other defect class is cosmetic and "normal" never emits a
#: detection (see argus.detectors.classifier.postprocess_classify).
CATASTROPHIC_CLASS_NAME = "spaghetti"

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

#: Output filename prefixes build_classification_dataset.py writes, used to recover each
#: test image's source for the source-confound diagnostic.
_SOURCE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("fdm_", "fdm"),
    ("argus_v2_", "argus_v2"),
    ("hf_normal_", "hf"),
)


# str infer_source(str filename)
# Inputs: str filename - a test-split image filename
# Outputs: str - "fdm", "argus_v2", "hf", or "unknown" if no known prefix matches
# Description: Recovers a test image's source dataset from its output filename prefix
#              (_SOURCE_PREFIXES, matching what training/build_classification_dataset.py
#              actually writes). Used by the SOURCE-CONFOUND DIAGNOSTIC (analysis 5) to split
#              spaghetti recall by dataset origin. Returns "unknown" rather than guessing when
#              no prefix matches, so callers must report that explicitly.
# Side Effects: None
def infer_source(filename: str) -> str:
    for prefix, source in _SOURCE_PREFIXES:
        if filename.startswith(prefix):
            return source
    return "unknown"


# --------------------------------------------------------------------------
# Pure functions -- confusion matrix, per-class metrics, binary collapse.
# --------------------------------------------------------------------------


# np.ndarray build_confusion_matrix(Sequence[str] y_true, Sequence[str] y_pred, Sequence[str] class_order)
# Inputs: Sequence[str] y_true - true class name per sample
#         Sequence[str] y_pred - predicted class name per sample, same length as y_true
#         Sequence[str] class_order - class names defining row/column order of the matrix
# Outputs: np.ndarray - len(class_order) x len(class_order) int64 matrix, rows = true class,
#          cols = predicted class, both indexed by position in class_order
# Description: Builds a confusion matrix from parallel true/predicted class-name sequences.
#              Operates on plain class names (never raw model indices), so it's testable with
#              synthetic data and no model/GPU.
# Side Effects: Raises ValueError if y_true and y_pred differ in length; raises KeyError (via
#               the underlying dict lookup) if a label isn't present in class_order.
def build_confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], class_order: Sequence[str]) -> np.ndarray:
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must be the same length, got {len(y_true)} vs {len(y_pred)}")
    index = {name: i for i, name in enumerate(class_order)}
    n = len(class_order)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[index[t], index[p]] += 1
    return cm


# float top1_accuracy(np.ndarray cm)
# Inputs: np.ndarray cm - confusion matrix from build_confusion_matrix
# Outputs: float - overall top-1 accuracy (trace / total), or 0.0 if the matrix is empty
# Description: Computes overall top-1 accuracy from a confusion matrix: trace / total.
# Side Effects: None
def top1_accuracy(cm: np.ndarray) -> float:
    total = int(cm.sum())
    if total == 0:
        return 0.0
    return float(np.trace(cm)) / total


# dict[str, dict[str, float | int]] per_class_prf1(np.ndarray cm, Sequence[str] class_order)
# Inputs: np.ndarray cm - confusion matrix from build_confusion_matrix
#         Sequence[str] class_order - class names matching cm's row/column order
# Outputs: dict[str, dict[str, float | int]] - per class name: precision, recall, f1, support
#          (true instance count), predicted_count
# Description: Computes per-class precision/recall/F1/support from a confusion matrix built
#              over the same class_order. Precision for a class with zero predictions, and
#              recall for a class with zero true instances, are both reported as 0.0 (rather
#              than raising or reporting NaN) -- a class the model never predicts at all is a
#              real, reportable failure mode, not an error.
# Side Effects: None
def per_class_prf1(cm: np.ndarray, class_order: Sequence[str]) -> dict[str, dict[str, float | int]]:
    """Zero-prediction precision and zero-support recall are reported as 0.0, not NaN --
    a class the model never predicts is a real failure mode, not an error."""
    n = cm.shape[0]
    out: dict[str, dict[str, float | int]] = {}
    for i, name in enumerate(class_order):
        tp = int(cm[i, i])
        support = int(cm[i, :].sum())
        predicted_count = int(cm[:, i].sum())
        precision = tp / predicted_count if predicted_count > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted_count": predicted_count,
        }
    return out


# list[str] binary_labels(Sequence[str] names, str normal_name)
# Inputs: Sequence[str] names - 6-way class names to collapse
#         str normal_name - the class name treated as "normal", default NORMAL_CLASS_NAME
#         ("normal"); matched case-insensitively after stripping
# Outputs: list[str] - "normal" or "defect" for each input name
# Description: Collapses a sequence of 6-way class names into a binary "normal"/"defect" label
#              sequence, for the analysis-3 binary collapse.
# Side Effects: None
def binary_labels(names: Sequence[str], normal_name: str = NORMAL_CLASS_NAME) -> list[str]:
    normal_name = normal_name.strip().lower()
    return ["normal" if n.strip().lower() == normal_name else "defect" for n in names]


# dict[str, object] binary_metrics(Sequence[str] y_true_bin, Sequence[str] y_pred_bin)
# Inputs: Sequence[str] y_true_bin - true "normal"/"defect" labels (from binary_labels)
#         Sequence[str] y_pred_bin - predicted "normal"/"defect" labels, same length
# Outputs: dict[str, object] - positive_class ("defect"), precision, recall, f1, a
#          confusion_matrix dict (tp, fp, fn, tn), num_normal, num_defect, and
#          false_positive_rate (of truly-normal images, fraction called some defect)
# Description: Computes precision/recall/F1 for the positive class "defect", a 2x2 confusion
#              matrix, and the plain false-positive rate -- the metric that actually determines
#              how often a healthy print gets flagged.
# Side Effects: None
def binary_metrics(y_true_bin: Sequence[str], y_pred_bin: Sequence[str]) -> dict[str, object]:
    tp = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == "defect" and p == "defect")
    fp = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == "normal" and p == "defect")
    fn = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == "defect" and p == "normal")
    tn = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == "normal" and p == "normal")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "positive_class": "defect",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "num_normal": fp + tn,
        "num_defect": tp + fn,
        "false_positive_rate": false_positive_rate,
    }


# --------------------------------------------------------------------------
# Pure functions -- the p_failure / catastrophic-path threshold sweep.
# Mirrors training/evaluate.py's sweep functions, including the min_recall
# floor against the vacuous precision=1.0-at-zero-predictions convention.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CatastrophicRecord:
    """`is_argmax` matters because the runtime rule only ever fires for the argmax class --
    a high spaghetti probability on an image predicted as something else can't drive p_failure."""

    true_name: str
    is_argmax: bool
    catastrophic_prob: float


# list[float] sweep_thresholds(float start, float end, float step)
# Inputs: float start - first confidence threshold in the sweep
#         float end - last confidence threshold in the sweep (inclusive)
#         float step - increment between thresholds
# Outputs: list[float] - confidence thresholds from start to end (inclusive) in steps of step,
#          rounded to 10 decimal places to avoid float accumulation artifacts
# Description: Builds the confidence-threshold sweep list; identical to
#              training/evaluate.py's sweep_thresholds.
# Side Effects: None
def sweep_thresholds(start: float, end: float, step: float) -> list[float]:
    """Identical to training/evaluate.py's `sweep_thresholds`."""
    n_steps = int(round((end - start) / step)) + 1
    return [round(start + i * step, 10) for i in range(n_steps) if start + i * step <= end + 1e-9]


# dict[str, float | int] evaluate_catastrophic_threshold(Sequence[CatastrophicRecord] records, float threshold, str catastrophic_name)
# Inputs: Sequence[CatastrophicRecord] records - one CatastrophicRecord per test image
#         float threshold - confidence threshold to evaluate
#         str catastrophic_name - the catastrophic class name, default CATASTROPHIC_CLASS_NAME
#         ("spaghetti")
# Outputs: dict[str, float | int] - threshold, precision, recall, tp, fp, fn, tn at this
#          threshold
# Description: Computes precision/recall of the REAL runtime catastrophic-path rule at one
#              confidence threshold: a record counts as a positive prediction iff its argmax
#              class is catastrophic_name AND catastrophic_prob >= threshold (mirroring
#              argus.detectors.classifier.postprocess_classify's actual gating). Precision when
#              zero predictions survive at this threshold is reported as 1.0 -- the same
#              vacuous-but-conventional fill training/evaluate.py uses (matching Ultralytics'
#              ap_per_class), which is why find_lowest_threshold_for_precision's min_recall
#              floor exists.
# Side Effects: None
def evaluate_catastrophic_threshold(
    records: Sequence[CatastrophicRecord], threshold: float, catastrophic_name: str = CATASTROPHIC_CLASS_NAME
) -> dict[str, float | int]:
    tp = fp = fn = tn = 0
    for r in records:
        is_true = r.true_name == catastrophic_name
        predicted_positive = r.is_argmax and r.catastrophic_prob >= threshold
        if predicted_positive and is_true:
            tp += 1
        elif predicted_positive and not is_true:
            fp += 1
        elif not predicted_positive and is_true:
            fn += 1
        else:
            tn += 1

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return {"threshold": threshold, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# list[dict[str, float | int]] sweep_catastrophic(Sequence[CatastrophicRecord] records, Sequence[float] thresholds, str catastrophic_name)
# Inputs: Sequence[CatastrophicRecord] records - one CatastrophicRecord per test image
#         Sequence[float] thresholds - confidence thresholds to evaluate (from sweep_thresholds)
#         str catastrophic_name - the catastrophic class name, default CATASTROPHIC_CLASS_NAME
#         ("spaghetti")
# Outputs: list[dict[str, float | int]] - evaluate_catastrophic_threshold's result for each
#          threshold, in the same order
# Description: Evaluates the catastrophic-path rule at every threshold in thresholds.
# Side Effects: None
def sweep_catastrophic(
    records: Sequence[CatastrophicRecord], thresholds: Sequence[float], catastrophic_name: str = CATASTROPHIC_CLASS_NAME
) -> list[dict[str, float | int]]:
    return [evaluate_catastrophic_threshold(records, t, catastrophic_name) for t in thresholds]


# tuple[Optional[dict[str, float | int]], Optional[dict[str, float | int]]] find_lowest_threshold_for_precision(Sequence[dict[str, float | int]] sweep, float target_precision, float min_recall)
# Inputs: Sequence[dict[str, float | int]] sweep - ascending-by-threshold sweep points (from
#         sweep_catastrophic)
#         float target_precision - minimum precision a threshold must reach
#         float min_recall - minimum recall a threshold must also reach, to reject the vacuous
#         precision=1.0-at-zero-predictions convention
# Outputs: tuple[Optional[dict[str, float | int]], Optional[dict[str, float | int]]] - (result,
#          vacuous_example): result is the first point meeting both target_precision and
#          min_recall, or None; vacuous_example is set (only when result is None) to the first
#          point that met target_precision but not min_recall
# Description: Scans sweep ascending by threshold and returns the first point whose precision
#              meets target_precision AND whose recall is at least min_recall. Mirrors
#              training/evaluate.py's find_lowest_threshold_for_precision exactly (same
#              two-return-value shape, same rationale): without the min_recall floor, a
#              threshold so high the model almost never fires can "achieve" perfect precision
#              purely by making almost no predictions -- a statistically meaningless,
#              undeployable operating point.
# Side Effects: None
def find_lowest_threshold_for_precision(
    sweep: Sequence[dict[str, float | int]], target_precision: float, min_recall: float
) -> tuple[Optional[dict[str, float | int]], Optional[dict[str, float | int]]]:
    """Mirrors training/evaluate.py's function of the same name: without the min_recall
    floor, a threshold so high the model never fires can "achieve" perfect precision vacuously."""
    vacuous_example: Optional[dict[str, float | int]] = None
    for pt in sweep:
        if pt["precision"] >= target_precision:
            if pt["recall"] >= min_recall:
                return pt, None
            if vacuous_example is None:
                vacuous_example = pt
    return None, vacuous_example


# dict[str, float | int] best_supported_point(Sequence[dict[str, float | int]] sweep, float min_recall)
# Inputs: Sequence[dict[str, float | int]] sweep - sweep points (from sweep_catastrophic)
#         float min_recall - the recall floor a sweep point must clear to count as "supported"
# Outputs: dict[str, float | int] - the chosen sweep point
# Description: Picks the highest-precision sweep point that still clears min_recall (i.e.
#              backed by real detections, not the vacuous 0-predictions artifact). Falls back
#              to the single highest-recall point if the class never clears min_recall anywhere
#              in the sweep. Mirrors training/evaluate.py's best_supported_point.
# Side Effects: None
def best_supported_point(sweep: Sequence[dict[str, float | int]], min_recall: float) -> dict[str, float | int]:
    supported = [pt for pt in sweep if pt["recall"] >= min_recall]
    if supported:
        return max(supported, key=lambda pt: pt["precision"])
    return max(sweep, key=lambda pt: pt["recall"])


# --------------------------------------------------------------------------
# Pure function -- source-confound recall split.
# --------------------------------------------------------------------------


# dict[str, dict[str, object]] group_recall_by_source(Sequence[str] sources, Sequence[bool] correct)
# Inputs: Sequence[str] sources - per-sample provenance label (e.g. "fdm", "argus_v2",
#         "unknown"), same length as correct
#         Sequence[bool] correct - whether the model's argmax matched the true class, per sample
# Outputs: dict[str, dict[str, object]] - {source: {"support", "correct", "recall"}}; recall is
#          None (not 0.0) for a source with zero support, so callers can't mistake "no data"
#          for "zero recall"
# Description: Groups correctness by source, used by the SOURCE-CONFOUND DIAGNOSTIC (analysis
#              5) to compute spaghetti recall separately per dataset origin.
# Side Effects: Raises ValueError if sources and correct differ in length. No I/O.
def group_recall_by_source(sources: Sequence[str], correct: Sequence[bool]) -> dict[str, dict[str, object]]:
    if len(sources) != len(correct):
        raise ValueError(f"sources and correct must be the same length, got {len(sources)} vs {len(correct)}")
    totals: dict[str, int] = defaultdict(int)
    hits: dict[str, int] = defaultdict(int)
    for s, c in zip(sources, correct):
        totals[s] += 1
        if c:
            hits[s] += 1
    return {
        s: {
            "support": totals[s],
            "correct": hits[s],
            "recall": (hits[s] / totals[s]) if totals[s] > 0 else None,
        }
        for s in sorted(totals)
    }


# --------------------------------------------------------------------------
# I/O -- dataset listing + model inference (not unit tested; exercised via a real run).
# --------------------------------------------------------------------------


# list[tuple[Path, str]] list_split_images(Path data_dir, str split, Sequence[str] class_names)
# Inputs: Path data_dir - classification dataset root (e.g. datasets/argus_cls)
#         str split - split name, "train", "val", or "test"
#         Sequence[str] class_names - class names to list images for, e.g. REPORT_CLASS_ORDER
# Outputs: list[tuple[Path, str]] - (image_path, true_class_name) pairs, sorted by class then
#          filename for determinism
# Description: Lists every image directly inside data_dir/split/<class>/ for each of
#              class_names. A missing class subdirectory is simply skipped (some classes may be
#              evaluable: false in split_report.json and so have no test images at all) --
#              callers should report which classes came back with zero images rather than
#              silently proceeding as if that's expected.
# Side Effects: Raises FileNotFoundError if data_dir/split doesn't exist. Read-only filesystem
#               traversal otherwise.
def list_split_images(data_dir: Path, split: str, class_names: Sequence[str]) -> list[tuple[Path, str]]:
    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    records: list[tuple[Path, str]] = []
    for cname in class_names:
        class_dir = split_dir / cname
        if not class_dir.is_dir():
            continue
        for p in sorted(class_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                records.append((p, cname))
    return records


# str select_backend(Path weights)
# Inputs: Path weights - path to the trained checkpoint (Ultralytics ``best.pt``, an ONNX
#         export, or a compiled Hailo ``.hef``)
# Outputs: str - "onnx" for a ".onnx" suffix, "hailo" for a ".hef" suffix (both
#          case-insensitive), else "ultralytics"
# Description: Picks which inference backend main() should use, purely from the weights file's
#              suffix -- the tinygrad-trained binary model is only ever available as ONNX or a
#              HEF compiled from it (no Ultralytics checkpoint exists for it), while the
#              original 6-class model is a ``.pt`` Ultralytics checkpoint.
# Side Effects: None (pure function of its input; does not touch the filesystem)
def select_backend(weights: Path) -> str:
    suffix = Path(weights).suffix.lower()
    if suffix == ".onnx":
        return "onnx"
    if suffix == ".hef":
        return "hailo"
    return "ultralytics"


# tuple[np.ndarray, dict[int, str]] run_inference(Path weights, Sequence[Path] image_paths, int imgsz, int batch, str device)
# Inputs: Path weights - path to the trained classification best.pt checkpoint
#         Sequence[Path] image_paths - images to classify
#         int imgsz - input image size the model expects
#         int batch - inference batch size
#         str device - CUDA device index, list, or "cpu"
# Outputs: tuple[np.ndarray, dict[int, str]] - (probs, names): probs is a
#          (len(image_paths), num_classes) float64 array (rows in the same order as
#          image_paths); names is the model's own {index: class_name} mapping, the model's real
#          output-index order that probs' columns are indexed by
# Description: Runs the Ultralytics classifier over image_paths and returns per-image
#              probability rows alongside the model's real class-name/index mapping, so callers
#              never have to assume an index order themselves.
# Side Effects: Imports ultralytics.YOLO lazily; loads the checkpoint into memory; runs a full
#               GPU/CPU inference pass over every image (reads each image file from disk).
#               Raises RuntimeError if model.predict returns a different number of results than
#               inputs.
def run_inference(
    weights: Path, image_paths: Sequence[Path], imgsz: int, batch: int, device: str
) -> tuple[np.ndarray, dict[int, str]]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names: dict[int, str] = dict(model.names)

    str_paths = [str(p) for p in image_paths]
    results = model.predict(str_paths, imgsz=imgsz, batch=batch, device=device, verbose=False)
    if len(results) != len(str_paths):
        raise RuntimeError(f"model.predict returned {len(results)} results for {len(str_paths)} input images")

    num_classes = len(names)
    probs = np.zeros((len(results), num_classes), dtype=np.float64)
    for i, r in enumerate(results):
        row = r.probs.data
        row = row.cpu().numpy() if hasattr(row, "cpu") else np.asarray(row)
        probs[i, :] = row
    return probs, names


# Optional[int] _static_input_size(Optional[Sequence[object]] shape)
# Inputs: Optional[Sequence[object]] shape - the ONNX-declared input shape (e.g. from
#         session.get_inputs()[0].shape), possibly containing symbolic/dynamic dims
# Outputs: Optional[int] - the static square spatial size (e.g. 320), or None if the shape is
#          missing, not 4D, dynamic, or non-square
# Description: Extracts a usable static input size from a declared ONNX input shape, so
#              run_inference_onnx can prefer the model's own baked-in size over --imgsz, exactly
#              like argus.detectors.classifier.ClassifierDetector does for the real runtime path.
#              Duplicated from argus.detectors.classifier._static_input_size rather than imported
#              -- that helper is private to that module (itself duplicated from
#              onnx_yolo._static_input_size for the same reason), and this evaluation script
#              stays free of any dependency on another module's private helpers.
# Side Effects: None (pure function of its input).
def _static_input_size(shape: Optional[Sequence[object]]) -> Optional[int]:
    if shape is None or len(shape) != 4:
        return None
    h, w = shape[2], shape[3]
    if isinstance(h, int) and isinstance(w, int) and h > 0 and h == w:
        return h
    return None


# tuple[np.ndarray, dict[int, str]] run_inference_onnx(Path onnx_path, Sequence[Path] image_paths, int imgsz)
# Inputs: Path onnx_path - path to the exported classifier ONNX model
#         Sequence[Path] image_paths - images to classify
#         int imgsz - FALLBACK input image size, used only when the ONNX graph itself declares a
#         dynamic (non-static) input shape; when the graph declares a static square shape (as
#         both the 6-class and binary exports do), that shape wins -- see _static_input_size and
#         argus.detectors.classifier.ClassifierDetector's identical precedence
# Outputs: tuple[np.ndarray, dict[int, str]] - (probs, names): probs is a
#          (len(image_paths), num_classes) float64 array (rows in the same order as
#          image_paths); names is the model's own {index: class_name} mapping, read from the
#          ONNX file's own metadata (the model's real output-index order that probs' columns are
#          indexed by) -- the SAME shape of result run_inference returns, so build_report and
#          every downstream metric function are backend-agnostic
# Description: Runs the exported ONNX classifier over image_paths, batch size 1 in a loop (test
#              splits here are a few hundred images, not worth batching). Reuses the real
#              production pre/post-processing from argus.detectors.classifier --
#              preprocess_classify for the exact resize-short-side + center-crop the printer runs
#              at inference time, and probabilities_from_output for the same raw-logits-vs
#              -already-softmaxed auto-detection production uses -- so train/eval/deploy can
#              never silently diverge by reimplementing either step here. Class names come from
#              class_names_from_onnx_metadata; unlike ClassifierDetector this script has no
#              DetectorConfig.class_names to fall back on, so missing/unparseable metadata is a
#              hard error rather than a guess.
# Side Effects: Imports onnxruntime and cv2 lazily; constructs an onnxruntime.InferenceSession
#               (CPUExecutionProvider) and reads each image file from disk via cv2.imread; runs
#               one inference pass per image. Raises ValueError if the ONNX file carries no
#               parseable class-name metadata. Raises IOError if an image fails to load.
def run_inference_onnx(onnx_path: Path, image_paths: Sequence[Path], imgsz: int) -> tuple[np.ndarray, dict[int, str]]:
    import cv2
    import onnxruntime as ort

    from argus.detectors.classifier import (
        class_names_from_onnx_metadata,
        preprocess_classify,
        probabilities_from_output,
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    class_names = class_names_from_onnx_metadata(session)
    if class_names is None:
        raise ValueError(
            f"ONNX model at '{onnx_path}' has no parseable class-name metadata (missing or "
            "unparseable 'names' entry in custom_metadata_map). Unlike "
            "argus.detectors.classifier.ClassifierDetector, this evaluation script has no "
            "DetectorConfig.class_names to fall back on -- guessing an index order here would "
            "silently produce a confidently wrong report. Re-export the model (see "
            "training/export_classifier_onnx.py) so it embeds its own class-name metadata."
        )
    names_by_idx: dict[int, str] = dict(enumerate(class_names))

    input_meta = session.get_inputs()[0]
    static_size = _static_input_size(input_meta.shape)
    effective_imgsz = static_size if static_size is not None else imgsz
    size_source = "the model's own static input shape" if static_size is not None else "--imgsz (model shape is dynamic)"
    print(f"[evaluate_classifier] ONNX backend: using input size {effective_imgsz} ({size_source})")

    num_classes = len(class_names)
    probs = np.zeros((len(image_paths), num_classes), dtype=np.float64)
    for i, path in enumerate(image_paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Failed to read image: {path}")
        blob = preprocess_classify(image, effective_imgsz)
        raw = session.run(None, {input_meta.name: blob})[0]
        probs[i, :] = np.asarray(probabilities_from_output(raw)).reshape(-1)
    return probs, names_by_idx


# tuple[np.ndarray, dict[int, str]] run_inference_hailo(Path hef_path, Sequence[Path] image_paths, int imgsz, Sequence[str] class_names)
# Inputs: Path hef_path - path to the compiled Hailo HEF (see docs/hailo-deployment.md)
#         Sequence[Path] image_paths - images to classify
#         int imgsz - input image size the HEF was compiled for; unlike run_inference_onnx there
#         is no model-declared static shape to prefer here, so this value is always authoritative
#         -- get it wrong and every image is cropped to the wrong size with no error raised
#         Sequence[str] class_names - class names in the model's real training-time output-index
#         order (from --class-names on the CLI -- a compiled HEF has no embedded metadata
#         equivalent to ONNX's custom_metadata_map for this script to fall back on)
# Outputs: tuple[np.ndarray, dict[int, str]] - (probs, names): probs is a
#          (len(image_paths), num_classes) float64 array (rows in the same order as
#          image_paths); names is dict(enumerate(class_names)) -- the SAME shape of result
#          run_inference/run_inference_onnx return, so build_report and every downstream metric
#          function are backend-agnostic
# Description: Runs the compiled HEF over image_paths through
#              argus.detectors.hailo.HailoDetector's own HailoRT plumbing (its predict_proba
#              method), batch size 1 in a loop -- same as run_inference_onnx. Reuses the real
#              production preprocessing (preprocess_classify_hailo, via predict_proba) and
#              probability normalization (probabilities_from_output, also via predict_proba) so
#              this evaluation can never silently diverge from what actually runs on the Pi. A
#              HailoDetector is constructed with default_threshold=0.0 and empty
#              class_thresholds/severity purely as plumbing to reach predict_proba -- this
#              function never calls postprocess_classify, so none of that threshold/severity
#              configuration is actually exercised.
# Side Effects: Imports cv2, argus.config, argus.detectors.hailo, and argus.types lazily.
#               Constructs a HailoDetector (allocating real HailoRT resources -- VDevice, network
#               group, vstreams; requires hailo_platform installed and a Hailo device attached,
#               i.e. this only runs on the Pi with the AI HAT+, never on this project's macOS dev
#               machine) and closes it in a finally block. Reads each image file from disk via
#               cv2.imread; runs one Hailo-8 inference pass per image. Raises IOError if an image
#               fails to load.
def run_inference_hailo(
    hef_path: Path, image_paths: Sequence[Path], imgsz: int, class_names: Sequence[str]
) -> tuple[np.ndarray, dict[int, str]]:
    import cv2

    from argus.config import DetectorConfig
    from argus.detectors.hailo import HailoDetector

    print(f"[evaluate_classifier] Hailo backend: using input size {imgsz} (--imgsz; no model-declared shape to prefer)")

    cfg = DetectorConfig(
        kind="hailo",
        model_path=str(hef_path),
        input_size=imgsz,
        class_names=tuple(class_names),
        default_threshold=0.0,
        class_thresholds={},
        severity={},
    )
    detector = HailoDetector(cfg)
    try:
        num_classes = len(class_names)
        probs = np.zeros((len(image_paths), num_classes), dtype=np.float64)
        for i, path in enumerate(image_paths):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise IOError(f"Failed to read image: {path}")
            probs[i, :] = detector.predict_proba(image)
    finally:
        detector.close()

    names_by_idx: dict[int, str] = dict(enumerate(class_names))
    return probs, names_by_idx


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


# Optional[str] check_class_order_matches_model(Mapping[int, str] names_by_idx, Sequence[str] report_order)
# Inputs: Mapping[int, str] names_by_idx - the model's own {index: class_name} mapping
#         (model.names), its real output-index order
#         Sequence[str] report_order - the human-readable order this report and
#         config.example.yaml's commented-out class_names list use, e.g. REPORT_CLASS_ORDER
# Outputs: Optional[str] - None if the orders match; otherwise a warning string describing the
#          mismatch
# Description: Compares the model's real output-index order (Ultralytics assigns indices
#              alphabetically from training folder names) against report_order. Because
#              ClassifierDetector (src/argus/detectors/classifier.py) trusts cfg.class_names to
#              already be in the model's index order, a mismatch here means wiring up
#              config.example.yaml's class_names list as literally written would silently
#              mislabel every prediction -- this function is the check that catches that before
#              deployment.
# Side Effects: None
def check_class_order_matches_model(names_by_idx: Mapping[int, str], report_order: Sequence[str]) -> Optional[str]:
    model_order = [names_by_idx[i] for i in sorted(names_by_idx)]
    if list(report_order) == model_order:
        return None
    return (
        "CLASS ORDER MISMATCH: the trained model's real output-index order is "
        f"{model_order} (Ultralytics assigns indices alphabetically from the training "
        f"folder names), but config.example.yaml's commented-out classification "
        f"class_names list (and this report's human-readable ordering) is {list(report_order)}. "
        "argus.detectors.classifier.ClassifierDetector trusts cfg.class_names to already be in "
        "the model's index order -- wiring up that config block with the list AS WRITTEN would "
        "silently mislabel every prediction. This script itself is unaffected (it looks up class "
        "names by string, never assumes an index order), but config.example.yaml's class_names "
        "list must be corrected to the model's real order before the classification path is ever "
        "deployed."
    )


# tuple[str, ...] resolve_report_class_order(Mapping[int, str] names_by_idx)
# Inputs: Mapping[int, str] names_by_idx - the model's own {index: class_name} mapping
# Outputs: tuple[str, ...] - the class order to use for this report's confusion matrix and
#          per-class table
# Description: Uses the curated human-readable REPORT_CLASS_ORDER when the model's classes are
#              exactly the known 6-class set (regardless of the model's real index order) --
#              preserving existing behaviour and check_class_order_matches_model's
#              alphabetical-vs-human-order warning for that model. For any other class set (e.g.
#              the binary failure/normal model trained outside Ultralytics), there is no curated
#              human order to assume, so this falls back to the model's own real index order.
# Side Effects: None
def resolve_report_class_order(names_by_idx: Mapping[int, str]) -> tuple[str, ...]:
    model_order = tuple(names_by_idx[i] for i in sorted(names_by_idx))
    if set(model_order) == set(REPORT_CLASS_ORDER):
        return REPORT_CLASS_ORDER
    return model_order


# int resolve_catastrophic_class_index(Mapping[int, str] names_by_idx, str catastrophic_class_name)
# Inputs: Mapping[int, str] names_by_idx - the model's own {index: class_name} mapping
#         str catastrophic_class_name - the class name to treat as catastrophic, from
#         --catastrophic-class (default CATASTROPHIC_CLASS_NAME, "spaghetti")
# Outputs: int - the model's output index for catastrophic_class_name
# Description: Resolves --catastrophic-class against the model's actual class names, so a typo
#              or a class the model doesn't have at all (e.g. running --catastrophic-class
#              spaghetti against the binary failure/normal model) fails loudly with the model's
#              real classes listed, rather than raising an opaque KeyError deep inside
#              build_report.
# Side Effects: Raises ValueError if catastrophic_class_name isn't one of the model's classes.
def resolve_catastrophic_class_index(names_by_idx: Mapping[int, str], catastrophic_class_name: str) -> int:
    idx_by_name = {name: i for i, name in names_by_idx.items()}
    if catastrophic_class_name not in idx_by_name:
        raise ValueError(
            f"--catastrophic-class '{catastrophic_class_name}' is not one of this model's classes: "
            f"{sorted(idx_by_name)}. Pass --catastrophic-class with one of those names (e.g. "
            "'failure' for the binary failure/normal model, or 'spaghetti' for the 6-class model)."
        )
    return idx_by_name[catastrophic_class_name]


# dict[str, object] build_report(Sequence[str] y_true, np.ndarray probs, Mapping[int, str] names_by_idx, Sequence[str] sources, argparse.Namespace args)
# Inputs: Sequence[str] y_true - true class name per test image
#         np.ndarray probs - (num_images, num_classes) probability array from run_inference,
#         columns indexed by names_by_idx
#         Mapping[int, str] names_by_idx - the model's own {index: class_name} mapping
#         Sequence[str] sources - per-image provenance label from infer_source, same length as
#         y_true
#         argparse.Namespace args - parsed CLI args (target_precision, min_recall, sweep
#         bounds, weights, data, split)
# Outputs: dict[str, object] - the full evaluation report: class-order warning, overall
#          accuracy, per-class P/R/F1, the confusion matrix, the binary normal-vs-defect
#          collapse, the p_failure catastrophic-path sweep, the source-confound diagnostic, and
#          a human-readable interpretation
# Description: Orchestrates all five analyses described in the module docstring (overall
#              accuracy/per-class metrics, confusion matrix, binary normal-vs-defect collapse,
#              the real runtime catastrophic-path threshold sweep for args.catastrophic_class,
#              and the source-confound diagnostic splitting that class's recall by dataset
#              origin) into one JSON-serializable report, then appends a human-readable
#              interpretation via build_interpretation. The reporting class order is resolved
#              from the model's own class names (resolve_report_class_order) rather than assumed
#              to be the 6-class REPORT_CLASS_ORDER, so this works unchanged for the binary
#              failure/normal model.
# Side Effects: None (pure computation over already-computed inference results; no I/O)
def build_report(
    y_true: Sequence[str],
    probs: np.ndarray,
    names_by_idx: Mapping[int, str],
    sources: Sequence[str],
    args: argparse.Namespace,
) -> dict[str, object]:
    argmax_idx = probs.argmax(axis=1)
    y_pred = [names_by_idx[int(i)] for i in argmax_idx]

    report_class_order = resolve_report_class_order(names_by_idx)
    class_order_warning = check_class_order_matches_model(names_by_idx, report_class_order)

    # -- 1 & 2: overall accuracy, per-class P/R/F1, full confusion matrix --
    cm = build_confusion_matrix(y_true, y_pred, report_class_order)
    overall_accuracy = top1_accuracy(cm)
    per_class = per_class_prf1(cm, report_class_order)

    # -- 3: binary normal-vs-defect collapse --
    y_true_bin = binary_labels(y_true)
    y_pred_bin = binary_labels(y_pred)
    binary = binary_metrics(y_true_bin, y_pred_bin)

    # -- 4: catastrophic-path threshold sweep (args.catastrophic_class, default "spaghetti") --
    catastrophic_class_name = args.catastrophic_class
    cat_idx = resolve_catastrophic_class_index(names_by_idx, catastrophic_class_name)
    records = [
        CatastrophicRecord(
            true_name=t,
            is_argmax=(int(a) == cat_idx),
            catastrophic_prob=float(probs[i, cat_idx]),
        )
        for i, (t, a) in enumerate(zip(y_true, argmax_idx))
    ]
    thresholds = sweep_thresholds(args.sweep_start, args.sweep_end, args.sweep_step)
    sweep = sweep_catastrophic(records, thresholds, catastrophic_class_name)
    target_result, vacuous_example = find_lowest_threshold_for_precision(sweep, args.target_precision, args.min_recall)
    best_point = best_supported_point(sweep, args.min_recall)

    p_failure_report = {
        "catastrophic_class": catastrophic_class_name,
        "target_precision": args.target_precision,
        "min_recall": args.min_recall,
        "confidence_sweep": sweep,
        "lowest_threshold_for_target_precision": target_result,
        "target_reachable": target_result is not None,
        "vacuous_precision_only": vacuous_example,
        "best_real_operating_point": best_point,
    }

    # -- 5: SOURCE-CONFOUND DIAGNOSTIC --
    catastrophic_mask = [t == catastrophic_class_name for t in y_true]
    catastrophic_sources = [s for s, m in zip(sources, catastrophic_mask) if m]
    catastrophic_correct = [p == catastrophic_class_name for p, m in zip(y_pred, catastrophic_mask) if m]
    by_source = group_recall_by_source(catastrophic_sources, catastrophic_correct)
    unknown_count = by_source.get("unknown", {}).get("support", 0)

    fdm_entry = by_source.get("fdm")
    argus_entry = by_source.get("argus_v2")
    recall_gap: Optional[float] = None
    if fdm_entry is not None and argus_entry is not None and fdm_entry["recall"] is not None and argus_entry["recall"] is not None:
        recall_gap = abs(float(fdm_entry["recall"]) - float(argus_entry["recall"]))

    source_confound = {
        "provenance_recoverable": unknown_count == 0,
        "method": (
            "recovered from the output filename prefix training/build_classification_dataset.py "
            "actually writes ('fdm_' vs 'argus_v2_' vs 'hf_normal_')"
        ),
        "by_source": by_source,
        "unknown_provenance_count": int(unknown_count),
        "recall_gap": recall_gap,
        "conclusion": _source_confound_conclusion(by_source, fdm_entry, argus_entry, recall_gap, unknown_count),
    }

    report: dict[str, object] = {
        "weights": str(args.weights),
        "data_dir": str(args.data),
        "split": args.split,
        "num_images": len(y_true),
        "model_class_order": [names_by_idx[i] for i in sorted(names_by_idx)],
        "report_class_order": list(report_class_order),
        "class_order_warning": class_order_warning,
        "overall": {
            "top1_accuracy": overall_accuracy,
            "num_correct": int(np.trace(cm)),
            "num_total": int(cm.sum()),
        },
        "per_class": per_class,
        "confusion_matrix": {"labels": list(report_class_order), "matrix": cm.tolist()},
        "binary_normal_vs_defect": binary,
        "p_failure_catastrophic": p_failure_report,
        "source_confound": source_confound,
    }
    report["interpretation"] = build_interpretation(report)
    return report


# str _source_confound_conclusion(Mapping[str, dict[str, object]] by_source, Optional[dict[str, object]] fdm_entry, Optional[dict[str, object]] argus_entry, Optional[float] recall_gap, int unknown_count)
# Inputs: Mapping[str, dict[str, object]] by_source - group_recall_by_source's full result, used
#         only to detect the degenerate single-source (or zero-source) case
#         Optional[dict[str, object]] fdm_entry - group_recall_by_source's "fdm" entry, or None
#         if no FDM-sourced catastrophic-class test images were found
#         Optional[dict[str, object]] argus_entry - group_recall_by_source's "argus_v2" entry,
#         or None if no argus_v2-sourced catastrophic-class test images were found
#         Optional[float] recall_gap - |FDM recall - argus_v2 recall|, or None if undefined
#         int unknown_count - number of catastrophic-class test images with unrecoverable
#         provenance
# Outputs: str - a human-readable verdict: "not applicable" if every catastrophic-class test
#          image resolves to a single source (e.g. the binary model's single-Hugging-Face-source
#          dataset), otherwise the FDM-vs-argus_v2 gap classification (LARGE/MODERATE/SMALL, or
#          inconclusive if either of those two specific sources has no data)
# Description: Interprets the SOURCE-CONFOUND DIAGNOSTIC (analysis 5) result. When by_source has
#              at most one entry (all catastrophic-class test images share one provenance, or
#              there are none at all), a cross-source comparison is degenerate by construction --
#              this returns a "not applicable" verdict rather than falling through to the
#              FDM-vs-argus_v2 language, which would misleadingly imply those two specific
#              sources were expected. Otherwise classifies the FDM-vs-argus_v2 gap size and
#              explains what it does and doesn't prove about the model's reliance on
#              lighting/framing/compression cues versus the actual defect.
# Side Effects: None (pure string formatting)
def _source_confound_conclusion(
    by_source: Mapping[str, dict[str, object]],
    fdm_entry: Optional[dict[str, object]],
    argus_entry: Optional[dict[str, object]],
    recall_gap: Optional[float],
    unknown_count: int,
) -> str:
    if len(by_source) <= 1:
        if by_source:
            (only_source,) = by_source.keys()
            source_desc = f"a single source ('{only_source}')"
        else:
            source_desc = "no source at all (zero catastrophic-class test images)"
        return (
            "Cross-source comparison is NOT APPLICABLE: every catastrophic-class test image "
            f"resolves to {source_desc}, so there is no second source to compare recall against. "
            "This is expected for a single-source dataset (e.g. the binary failure/normal model's "
            "dataset, which is entirely from one Hugging Face source) and is not itself evidence "
            "for or against a source confound -- it simply means this particular diagnostic has "
            "nothing to compare here."
        )
    if fdm_entry is None or argus_entry is None:
        return (
            "Could not compute a per-source recall split -- catastrophic-class test images from one "
            "or both sources (fdm, argus_v2) were not found. Provenance-based diagnosis is "
            "inconclusive."
        )
    if recall_gap is None:
        return "One source had zero catastrophic-class test images; recall gap is undefined."

    fdm_recall = fdm_entry["recall"]
    argus_recall = argus_entry["recall"]
    fdm_n = fdm_entry["support"]
    argus_n = argus_entry["support"]
    unknown_note = (
        f" ({unknown_count} catastrophic-class test image(s) had unrecoverable provenance.)" if unknown_count else ""
    )

    if recall_gap >= 0.20:
        verdict = (
            f"LARGE gap ({recall_gap:.3f}) between FDM recall ({fdm_recall:.3f}, n={fdm_n}) and argus_v2 "
            f"recall ({argus_recall:.3f}, n={argus_n}): strong evidence the model is partly keying on "
            "dataset origin (lighting/framing/compression) rather than the catastrophic-class defect "
            "itself. Recall on whichever source is real deployment conditions should be treated as the "
            "model's true catastrophic-class recall, not the pooled/average figure."
        )
    elif recall_gap >= 0.10:
        verdict = (
            f"MODERATE gap ({recall_gap:.3f}) between FDM recall ({fdm_recall:.3f}, n={fdm_n}) and "
            f"argus_v2 recall ({argus_recall:.3f}, n={argus_n}): some evidence of source-keying, though "
            "with n as small as this the gap could also be sampling noise. Not conclusive either way."
        )
    else:
        verdict = (
            f"SMALL gap ({recall_gap:.3f}) between FDM recall ({fdm_recall:.3f}, n={fdm_n}) and argus_v2 "
            f"recall ({argus_recall:.3f}, n={argus_n}): no strong evidence the model is keying on dataset "
            "origin for the catastrophic class specifically. This does NOT clear the model of the "
            "confound generally -- it only means the catastrophic class's two sources score similarly; "
            "other defect classes may have no second source to run this same check against at all."
        )
    return verdict + unknown_note


# str build_interpretation(dict[str, object] report)
# Inputs: dict[str, object] report - the in-progress report dict from build_report (must
#         already have "p_failure_catastrophic" and "source_confound" populated)
# Outputs: str - a multi-line human-readable verdict: whether the model is fit to drive
#          automated print-pausing on the catastrophic-class path (report["p_failure_catastrophic"]
#          ["catastrophic_class"]), a precision summary, a source-confound summary, and (if
#          present) the class-order warning
# Description: Synthesizes the catastrophic-path precision result and the source-confound
#              recall gap into one bottom-line recommendation on whether this model should be
#              allowed to drive automated pause/cancel actions, or should stay notify_only.
# Side Effects: None (pure string formatting)
def build_interpretation(report: dict[str, object]) -> str:
    cat = report["p_failure_catastrophic"]  # type: ignore[assignment]
    confound = report["source_confound"]  # type: ignore[assignment]
    catastrophic_class_name = cat["catastrophic_class"]
    target = cat["target_precision"]
    min_recall = cat["min_recall"]
    result = cat["lowest_threshold_for_target_precision"]
    vacuous = cat["vacuous_precision_only"]
    best = cat["best_real_operating_point"]

    if result is not None:
        precision_ok = True
        precision_summary = (
            f"the {catastrophic_class_name} (catastrophic) path reaches the target precision ({target:.2f}) at "
            f"confidence threshold {result['threshold']:.2f}: precision={result['precision']:.3f}, "
            f"recall={result['recall']:.3f}"
        )
    elif vacuous is not None:
        precision_ok = False
        precision_summary = (
            f"the {catastrophic_class_name} (catastrophic) path only 'reaches' the target precision "
            f"({target:.2f}) at a vacuous, near-zero-recall operating point (precision={vacuous['precision']:.3f}, "
            f"recall={vacuous['recall']:.3f} @ conf={vacuous['threshold']:.2f} -- the model almost never "
            f"fires there); its best REAL operating point (recall >= {min_recall:.2f}) is "
            f"precision={best['precision']:.3f}, recall={best['recall']:.3f} @ conf={best['threshold']:.2f}"
        )
    else:
        precision_ok = False
        precision_summary = (
            f"the {catastrophic_class_name} (catastrophic) path NEVER reaches the target precision "
            f"({target:.2f}) at any threshold tried; its best real operating point (recall >= {min_recall:.2f}) "
            f"is precision={best['precision']:.3f}, recall={best['recall']:.3f} @ conf={best['threshold']:.2f}"
        )

    gap = confound["recall_gap"]
    confound_ok: Optional[bool]
    if gap is None:
        confound_ok = None
        confound_summary = "the source-confound recall split could not be fully computed"
    else:
        confound_ok = gap < 0.20
        by_source = confound["by_source"]
        fdm = by_source.get("fdm", {})
        argus = by_source.get("argus_v2", {})
        confound_summary = (
            f"{catastrophic_class_name} recall is {fdm.get('recall', float('nan')):.3f} on FDM-sourced test "
            f"images (n={fdm.get('support', 0)}) vs {argus.get('recall', float('nan')):.3f} on argus_v2-sourced "
            f"test images (n={argus.get('support', 0)}), a gap of {gap:.3f}"
        )

    fit_for_pausing = bool(precision_ok and (confound_ok is not False))

    verdict_line = (
        f"This model IS reasonably fit to drive automated print-pausing on the {catastrophic_class_name} path, "
        "subject to the caveats above."
        if fit_for_pausing
        else "This model is NOT fit to drive automated print-pausing (action_mode should stay notify_only)."
    )

    lines = [
        f"INTERPRETATION: {verdict_line}",
        f"  - Catastrophic-path precision: {precision_summary}.",
        f"  - Source confound: {confound_summary}. {confound['conclusion']}",
        f"  - Only '{catastrophic_class_name}' can ever drive an automated pause/cancel per the deployed "
        "severity mapping (config.example.yaml for the 6-class model); 'normal' never emits a detection "
        "at all (argus.detectors.classifier.postprocess_classify), so this catastrophic-path number is "
        "the entire automated-action false-positive story for any other class configured as "
        "cosmetic-only.",
    ]
    if report.get("class_order_warning"):
        lines.append(f"  - {report['class_order_warning']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------


# None print_report(dict[str, object] report)
# Inputs: dict[str, object] report - the full evaluation report from build_report
# Outputs: None
# Description: Prints the human-readable evaluation report to stdout: overall accuracy and
#              per-class table, the confusion matrix, the binary normal-vs-defect breakdown and
#              false-positive rate, the p_failure catastrophic-path threshold sweep, the
#              source-confound diagnostic, and the final interpretation.
# Side Effects: Prints the full evaluation report to stdout. No filesystem writes.
def print_report(report: dict[str, object]) -> None:
    print()
    print("=" * 88)
    print(f"CLASSIFIER EVALUATION -- {str(report['split']).upper()} SPLIT (held out from training)")
    print("=" * 88)
    print(f"Weights: {report['weights']}")
    print(f"Images:  {report['num_images']}")
    if report.get("class_order_warning"):
        print()
        print("!" * 88)
        print(report["class_order_warning"])
        print("!" * 88)

    overall = report["overall"]  # type: ignore[assignment]
    print()
    print(f"[1] OVERALL TOP-1 ACCURACY: {overall['top1_accuracy']:.4f}  ({overall['num_correct']}/{overall['num_total']})")
    print()
    header = f"{'class':<18}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}{'pred_count':>12}"
    print(header)
    print("-" * len(header))
    report_catastrophic_class = report["p_failure_catastrophic"]["catastrophic_class"]  # type: ignore[index]
    for cname in report["report_class_order"]:  # type: ignore[union-attr]
        e = report["per_class"][cname]  # type: ignore[index]
        tag = "  [CATASTROPHIC]" if cname == report_catastrophic_class else ""
        print(
            f"{cname:<18}{e['precision']:>10.3f}{e['recall']:>10.3f}{e['f1']:>10.3f}"
            f"{e['support']:>10}{e['predicted_count']:>12}{tag}"
        )

    print()
    print("[2] CONFUSION MATRIX (rows=true, cols=predicted)")
    cm_info = report["confusion_matrix"]  # type: ignore[assignment]
    labels = cm_info["labels"]
    matrix = cm_info["matrix"]
    col_w = max(10, max(len(l) for l in labels) + 2)
    print(" " * 18 + "".join(f"{l:>{col_w}}" for l in labels))
    for i, row_label in enumerate(labels):
        print(f"{row_label:<18}" + "".join(f"{v:>{col_w}}" for v in matrix[i]))

    print()
    b = report["binary_normal_vs_defect"]  # type: ignore[assignment]
    print("[3] BINARY: normal vs defect (positive class = 'defect')")
    print(f"  precision={b['precision']:.4f}  recall={b['recall']:.4f}  f1={b['f1']:.4f}")
    bcm = b["confusion_matrix"]
    print(f"  2x2 (rows=true, cols=predicted):        pred_normal   pred_defect")
    print(f"    true_normal                          {bcm['tn']:>12}  {bcm['fp']:>12}")
    print(f"    true_defect                          {bcm['fn']:>12}  {bcm['tp']:>12}")
    print(
        f"  FALSE-POSITIVE RATE (truly normal, called some defect): {b['false_positive_rate']:.4f}  "
        f"({bcm['fp']}/{bcm['fp'] + bcm['tn']} normal images)"
    )

    print()
    cat = report["p_failure_catastrophic"]  # type: ignore[assignment]
    target = cat["target_precision"]
    min_recall = cat["min_recall"]
    print(f"[4] p_failure ANALYSIS -- catastrophic path ('{cat['catastrophic_class']}' only)")
    print(
        f"  Real runtime rule: p_failure = '{cat['catastrophic_class']}' probability, ONLY when "
        f"'{cat['catastrophic_class']}' is the model's own top-1 prediction AND that probability clears "
        f"the threshold."
    )
    print(f"  Threshold sweep (target precision >= {target:.2f}, min_recall floor {min_recall:.2f}):")
    print(f"  {'conf':>6}{'precision':>12}{'recall':>10}{'tp':>6}{'fp':>6}{'fn':>6}{'tn':>6}")
    for pt in cat["confidence_sweep"]:
        print(
            f"  {pt['threshold']:>6.2f}{pt['precision']:>12.3f}{pt['recall']:>10.3f}"
            f"{pt['tp']:>6}{pt['fp']:>6}{pt['fn']:>6}{pt['tn']:>6}"
        )
    result = cat["lowest_threshold_for_target_precision"]
    if result is not None:
        print(
            f"  -> LOWEST threshold reaching target precision (with recall >= {min_recall:.2f}): "
            f"conf={result['threshold']:.2f}  precision={result['precision']:.3f}  recall={result['recall']:.3f}"
        )
    elif cat["vacuous_precision_only"] is not None:
        v = cat["vacuous_precision_only"]
        best = cat["best_real_operating_point"]
        print(
            f"  -> VACUOUS: target precision {target:.2f} is only 'met' at near-zero recall "
            f"(precision={v['precision']:.3f} recall={v['recall']:.3f} @ conf={v['threshold']:.2f} -- the "
            f"model effectively never fires there). Best REAL operating point: "
            f"precision={best['precision']:.3f} recall={best['recall']:.3f} @ conf={best['threshold']:.2f}."
        )
    else:
        best = cat["best_real_operating_point"]
        print(
            f"  -> target precision {target:.2f} NOT REACHABLE at any threshold tried. Best real "
            f"operating point: precision={best['precision']:.3f} recall={best['recall']:.3f} "
            f"@ conf={best['threshold']:.2f}."
        )

    print()
    print(
        f"[5] SOURCE-CONFOUND DIAGNOSTIC -- '{cat['catastrophic_class']}' recall by source "
        "(the most important check)"
    )
    sc = report["source_confound"]  # type: ignore[assignment]
    print(f"  Provenance method: {sc['method']}")
    for source, entry in sc["by_source"].items():
        recall_str = f"{entry['recall']:.3f}" if entry["recall"] is not None else "n/a"
        print(f"  {source:<12} support={entry['support']:>4}  correct={entry['correct']:>4}  recall={recall_str}")
    if sc["unknown_provenance_count"]:
        print(
            f"  WARNING: {sc['unknown_provenance_count']} '{cat['catastrophic_class']}' test image(s) had "
            "unrecoverable provenance."
        )
    if sc["recall_gap"] is not None:
        print(f"  Recall gap (|FDM - argus_v2|): {sc['recall_gap']:.3f}")
    print(f"  Conclusion: {sc['conclusion']}")

    print()
    print("-" * 88)
    print(report["interpretation"])
    print("-" * 88)
    print("=" * 88)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


# argparse.Namespace parse_args(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: argparse.Namespace - parsed evaluation options (weights, data, split, imgsz, batch,
#          device, target_precision, min_recall, sweep_start/end/step, catastrophic_class, out).
#          Notable defaults: --split test, --target-precision 0.95, --min-recall 0.05 (the
#          vacuous-precision floor, same rationale as training/evaluate.py), --catastrophic-class
#          spaghetti (the 6-class model's convention; pass "failure" for the binary model).
# Description: Defines and parses the CLI for evaluating the classification checkpoint, for
#              either backend (select_backend picks between them from --weights' suffix).
# Side Effects: None (argparse may print usage/help and call sys.exit on bad input, but no
#               filesystem or network activity)
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help=f"Path to trained best.pt, exported .onnx, or a compiled Hailo .hef (default: {DEFAULT_WEIGHTS}); backend is auto-selected from the suffix (select_backend)")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR, help=f"Classification dataset root (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Which split to evaluate (default: test)")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=512,
        help=(
            "Input image size (default: 512, matching the 6-class Ultralytics model; the binary "
            "ONNX/HEF model is trained at 320 -- pass --imgsz 320 for it). For the ONNX backend "
            "this is only a FALLBACK: run_inference_onnx prefers the model's own static input "
            "shape baked into the ONNX graph (exactly like "
            "argus.detectors.classifier.ClassifierDetector does at runtime) whenever the graph "
            "declares one, and only falls back to this value if the graph's input shape is "
            "dynamic. The Ultralytics (.pt) and Hailo (.hef) backends always use this value "
            "directly -- neither has a model-declared shape this script knows how to read, so "
            "getting it wrong silently crops every image to the wrong size."
        ),
    )
    parser.add_argument("--batch", type=int, default=32, help="Batch size for the Ultralytics (.pt) backend (default: 32). The ONNX and Hailo backends always run batch size 1 in a loop.")
    parser.add_argument("--device", type=str, default="0", help="CUDA device for the Ultralytics (.pt) backend (default: 0). Unused by the ONNX/Hailo backends.")
    parser.add_argument(
        "--class-names",
        type=str,
        default=None,
        help=(
            "Comma-separated class names in the model's real training-time output-index order "
            "(e.g. 'failure,normal'). REQUIRED for the Hailo (.hef) backend -- a compiled HEF "
            "carries no embedded class-name metadata for this script to fall back on, unlike the "
            "ONNX backend's custom_metadata_map (class_names_from_onnx_metadata). Ignored by the "
            "ONNX and Ultralytics backends, which always resolve class names from the model "
            "itself."
        ),
    )
    parser.add_argument(
        "--catastrophic-class",
        type=str,
        default=CATASTROPHIC_CLASS_NAME,
        help=(
            f"Which class drives the p_failure catastrophic-path threshold sweep and the "
            f"source-confound diagnostic (default: '{CATASTROPHIC_CLASS_NAME}', the 6-class model's "
            "only CATASTROPHIC-severity class). Pass 'failure' for the binary failure/normal model. "
            "Must be one of the model's actual class names (resolve_catastrophic_class_index raises "
            "a clear error listing the model's real classes otherwise)."
        ),
    )
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.05,
        help=(
            "A candidate threshold must also reach at least this much recall to count as 'reaching' "
            "--target-precision (default: 0.05). At a high enough confidence threshold the model makes "
            "zero predictions for a class, which this script (matching training/evaluate.py's convention) "
            "reports as precision=1.0 by definition -- a statistically meaningless 'perfect' score, not a "
            "usable operating point. This floor rejects that vacuous case."
        ),
    )
    parser.add_argument("--sweep-start", type=float, default=0.05)
    parser.add_argument("--sweep-end", type=float, default=0.95)
    parser.add_argument("--sweep-step", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help=f"Where to write the full JSON report (default: {DEFAULT_OUT_PATH})")
    return parser.parse_args(argv)


# None main(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: None
# Description: CLI entry point. Lists the requested split's images (discovering the actual class
#              subdirectories present on disk rather than assuming the 6-class REPORT_CLASS_ORDER,
#              so this works for the binary failure/normal dataset too), warns about missing
#              6-class-model classes when applicable, recovers each image's source provenance,
#              selects the inference backend from --weights' suffix (select_backend) and runs it,
#              builds the full evaluation report, prints it, and writes it to disk as JSON.
# Side Effects: Raises FileNotFoundError if --weights or --data don't exist; raises
#               RuntimeError if the split has no images at all; runs a full GPU/CPU (Ultralytics)
#               or CPU (ONNX) inference pass reading every listed image from disk (see
#               run_inference / run_inference_onnx); prints progress, warnings, and the full
#               evaluation report to stdout; creates --out's parent directory and writes the full
#               JSON report to --out (default runs/classifier_evaluation.json).
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.data.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {args.data}")

    print(f"[evaluate_classifier] Listing '{args.split}' split images under '{args.data}' ...")
    # Discover the class subdirectories actually present on disk rather than assuming
    # REPORT_CLASS_ORDER -- the binary model's dataset has "failure"/"normal" folders, not the
    # 6-class set, and list_split_images only lists whatever class_names it's given.
    split_dir = args.data / args.split
    discovered_classes = tuple(sorted(p.name for p in split_dir.iterdir() if p.is_dir())) if split_dir.is_dir() else ()
    class_names_for_listing = discovered_classes if discovered_classes else REPORT_CLASS_ORDER

    records = list_split_images(args.data, args.split, class_names_for_listing)
    if not records:
        raise RuntimeError(f"No images found for split '{args.split}' under '{args.data}'")

    present_classes = {c for _, c in records}
    # The "missing classes" warning only makes sense when this looks like the known 6-class
    # dataset (some classes may legitimately have zero test images, e.g. evaluable: false in
    # split_report.json); for any other class set (e.g. the binary dataset) there is no such
    # expectation to check.
    if set(class_names_for_listing) <= set(REPORT_CLASS_ORDER):
        missing_classes = [c for c in REPORT_CLASS_ORDER if c not in present_classes]
        if missing_classes:
            print(f"[evaluate_classifier] WARNING: these classes have ZERO images in split '{args.split}': {missing_classes}")

    paths = [p for p, _ in records]
    y_true = [c for _, c in records]
    sources = [infer_source(p.name) for p in paths]

    backend = select_backend(args.weights)
    print(f"[evaluate_classifier] Running inference ({backend} backend) on {len(paths)} images with '{args.weights}' ...")
    if backend == "onnx":
        probs, names_by_idx = run_inference_onnx(args.weights, paths, args.imgsz)
    elif backend == "hailo":
        if not args.class_names:
            raise ValueError(
                "--class-names is required when evaluating a Hailo .hef file -- a compiled HEF "
                "carries no embedded class-name metadata for this script to fall back on (unlike "
                "the ONNX backend's custom_metadata_map). Pass e.g. --class-names failure,normal "
                "in the model's real training-time output order (the same order used when the "
                "model was exported to ONNX, before HEF compilation)."
            )
        class_names = tuple(name.strip() for name in args.class_names.split(","))
        probs, names_by_idx = run_inference_hailo(args.weights, paths, args.imgsz, class_names)
    else:
        probs, names_by_idx = run_inference(args.weights, paths, args.imgsz, args.batch, args.device)

    report = build_report(y_true, probs, names_by_idx, sources, args)
    print_report(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[evaluate_classifier] Full report written to: {args.out}")


if __name__ == "__main__":
    main()
