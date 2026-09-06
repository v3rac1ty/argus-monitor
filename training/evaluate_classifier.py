"""Evaluates a trained 6-class classifier on ``--split`` (default test): overall accuracy and per-class P/R/F1, the confusion matrix, a binary normal-vs-defect collapse with false-positive rate, the real runtime ``p_failure`` threshold sweep for ``spaghetti`` (the only class allowed to auto-pause a print), and a source-confound diagnostic splitting spaghetti recall by dataset origin (this dataset's ``normal`` is 100% Hugging Face while other defects are ~100% FDM; ``spaghetti`` is the one class mixed across both).

``--min-recall`` rejects the vacuous "precision=1.0 at ~zero recall" convention: a threshold so high the model almost never fires trivially has no false positives and no real value. Ultralytics assigns class indices alphabetically, not the human-readable order used for reporting here -- a mismatch is flagged loudly since ``ClassifierDetector`` trusts ``cfg.class_names`` to match the model's real order.

Usage: python training/evaluate_classifier.py --weights <best.pt> [--split test] [--target-precision 0.95]
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
#              the real runtime catastrophic-path threshold sweep for "spaghetti", and the
#              source-confound diagnostic splitting spaghetti recall by dataset origin) into one
#              JSON-serializable report, then appends a human-readable interpretation via
#              build_interpretation.
# Side Effects: None (pure computation over already-computed inference results; no I/O)
def build_report(
    y_true: Sequence[str],
    probs: np.ndarray,
    names_by_idx: Mapping[int, str],
    sources: Sequence[str],
    args: argparse.Namespace,
) -> dict[str, object]:
    idx_by_name = {name: i for i, name in names_by_idx.items()}
    argmax_idx = probs.argmax(axis=1)
    y_pred = [names_by_idx[int(i)] for i in argmax_idx]

    class_order_warning = check_class_order_matches_model(names_by_idx, REPORT_CLASS_ORDER)

    # -- 1 & 2: overall accuracy, per-class P/R/F1, full confusion matrix --
    cm = build_confusion_matrix(y_true, y_pred, REPORT_CLASS_ORDER)
    overall_accuracy = top1_accuracy(cm)
    per_class = per_class_prf1(cm, REPORT_CLASS_ORDER)

    # -- 3: binary normal-vs-defect collapse --
    y_true_bin = binary_labels(y_true)
    y_pred_bin = binary_labels(y_pred)
    binary = binary_metrics(y_true_bin, y_pred_bin)

    # -- 4: catastrophic-path (spaghetti) threshold sweep --
    cat_idx = idx_by_name[CATASTROPHIC_CLASS_NAME]
    records = [
        CatastrophicRecord(
            true_name=t,
            is_argmax=(int(a) == cat_idx),
            catastrophic_prob=float(probs[i, cat_idx]),
        )
        for i, (t, a) in enumerate(zip(y_true, argmax_idx))
    ]
    thresholds = sweep_thresholds(args.sweep_start, args.sweep_end, args.sweep_step)
    sweep = sweep_catastrophic(records, thresholds)
    target_result, vacuous_example = find_lowest_threshold_for_precision(sweep, args.target_precision, args.min_recall)
    best_point = best_supported_point(sweep, args.min_recall)

    p_failure_report = {
        "catastrophic_class": CATASTROPHIC_CLASS_NAME,
        "target_precision": args.target_precision,
        "min_recall": args.min_recall,
        "confidence_sweep": sweep,
        "lowest_threshold_for_target_precision": target_result,
        "target_reachable": target_result is not None,
        "vacuous_precision_only": vacuous_example,
        "best_real_operating_point": best_point,
    }

    # -- 5: SOURCE-CONFOUND DIAGNOSTIC --
    spaghetti_mask = [t == CATASTROPHIC_CLASS_NAME for t in y_true]
    spaghetti_sources = [s for s, m in zip(sources, spaghetti_mask) if m]
    spaghetti_correct = [p == CATASTROPHIC_CLASS_NAME for p, m in zip(y_pred, spaghetti_mask) if m]
    by_source = group_recall_by_source(spaghetti_sources, spaghetti_correct)
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
        "conclusion": _source_confound_conclusion(fdm_entry, argus_entry, recall_gap, unknown_count),
    }

    report: dict[str, object] = {
        "weights": str(args.weights),
        "data_dir": str(args.data),
        "split": args.split,
        "num_images": len(y_true),
        "model_class_order": [names_by_idx[i] for i in sorted(names_by_idx)],
        "report_class_order": list(REPORT_CLASS_ORDER),
        "class_order_warning": class_order_warning,
        "overall": {
            "top1_accuracy": overall_accuracy,
            "num_correct": int(np.trace(cm)),
            "num_total": int(cm.sum()),
        },
        "per_class": per_class,
        "confusion_matrix": {"labels": list(REPORT_CLASS_ORDER), "matrix": cm.tolist()},
        "binary_normal_vs_defect": binary,
        "p_failure_catastrophic": p_failure_report,
        "source_confound": source_confound,
    }
    report["interpretation"] = build_interpretation(report)
    return report


# str _source_confound_conclusion(Optional[dict[str, object]] fdm_entry, Optional[dict[str, object]] argus_entry, Optional[float] recall_gap, int unknown_count)
# Inputs: Optional[dict[str, object]] fdm_entry - group_recall_by_source's "fdm" entry, or None
#         if no FDM-sourced spaghetti test images were found
#         Optional[dict[str, object]] argus_entry - group_recall_by_source's "argus_v2" entry,
#         or None if no argus_v2-sourced spaghetti test images were found
#         Optional[float] recall_gap - |FDM recall - argus_v2 recall|, or None if undefined
#         int unknown_count - number of spaghetti test images with unrecoverable provenance
# Outputs: str - a human-readable verdict on whether the recall gap indicates the model is
#          keying on dataset origin rather than the spaghetti defect itself (LARGE/MODERATE/
#          SMALL gap, or inconclusive if either source has no data)
# Description: Interprets the FDM-vs-argus_v2 spaghetti recall gap for the SOURCE-CONFOUND
#              DIAGNOSTIC (analysis 5), classifying the gap size and explaining what it does and
#              doesn't prove about the model's reliance on lighting/framing/compression cues
#              versus the actual defect.
# Side Effects: None (pure string formatting)
def _source_confound_conclusion(
    fdm_entry: Optional[dict[str, object]],
    argus_entry: Optional[dict[str, object]],
    recall_gap: Optional[float],
    unknown_count: int,
) -> str:
    if fdm_entry is None or argus_entry is None:
        return (
            "Could not compute a per-source recall split -- spaghetti test images from one or both "
            "sources (fdm, argus_v2) were not found. Provenance-based diagnosis is inconclusive."
        )
    if recall_gap is None:
        return "One source had zero spaghetti test images; recall gap is undefined."

    fdm_recall = fdm_entry["recall"]
    argus_recall = argus_entry["recall"]
    fdm_n = fdm_entry["support"]
    argus_n = argus_entry["support"]
    unknown_note = f" ({unknown_count} spaghetti test image(s) had unrecoverable provenance.)" if unknown_count else ""

    if recall_gap >= 0.20:
        verdict = (
            f"LARGE gap ({recall_gap:.3f}) between FDM recall ({fdm_recall:.3f}, n={fdm_n}) and argus_v2 "
            f"recall ({argus_recall:.3f}, n={argus_n}): strong evidence the model is partly keying on "
            "dataset origin (lighting/framing/compression) rather than the spaghetti defect itself. "
            "Recall on whichever source is real deployment conditions should be treated as the model's "
            "true spaghetti recall, not the pooled/average figure."
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
            "origin for spaghetti specifically. This does NOT clear the model of the confound generally "
            "-- it only means spaghetti's two sources score similarly; the other four defect classes "
            "have no second source to run this same check against at all."
        )
    return verdict + unknown_note


# str build_interpretation(dict[str, object] report)
# Inputs: dict[str, object] report - the in-progress report dict from build_report (must
#         already have "p_failure_catastrophic" and "source_confound" populated)
# Outputs: str - a multi-line human-readable verdict: whether the model is fit to drive
#          automated print-pausing on the spaghetti path, a precision summary, a source-confound
#          summary, and (if present) the class-order warning
# Description: Synthesizes the catastrophic-path precision result and the source-confound
#              recall gap into one bottom-line recommendation on whether this model should be
#              allowed to drive automated pause/cancel actions, or should stay notify_only.
# Side Effects: None (pure string formatting)
def build_interpretation(report: dict[str, object]) -> str:
    cat = report["p_failure_catastrophic"]  # type: ignore[assignment]
    confound = report["source_confound"]  # type: ignore[assignment]
    target = cat["target_precision"]
    min_recall = cat["min_recall"]
    result = cat["lowest_threshold_for_target_precision"]
    vacuous = cat["vacuous_precision_only"]
    best = cat["best_real_operating_point"]

    if result is not None:
        precision_ok = True
        precision_summary = (
            f"the spaghetti (catastrophic) path reaches the target precision ({target:.2f}) at confidence "
            f"threshold {result['threshold']:.2f}: precision={result['precision']:.3f}, recall={result['recall']:.3f}"
        )
    elif vacuous is not None:
        precision_ok = False
        precision_summary = (
            f"the spaghetti (catastrophic) path only 'reaches' the target precision ({target:.2f}) at a "
            f"vacuous, near-zero-recall operating point (precision={vacuous['precision']:.3f}, "
            f"recall={vacuous['recall']:.3f} @ conf={vacuous['threshold']:.2f} -- the model almost never "
            f"fires there); its best REAL operating point (recall >= {min_recall:.2f}) is "
            f"precision={best['precision']:.3f}, recall={best['recall']:.3f} @ conf={best['threshold']:.2f}"
        )
    else:
        precision_ok = False
        precision_summary = (
            f"the spaghetti (catastrophic) path NEVER reaches the target precision ({target:.2f}) at any "
            f"threshold tried; its best real operating point (recall >= {min_recall:.2f}) is "
            f"precision={best['precision']:.3f}, recall={best['recall']:.3f} @ conf={best['threshold']:.2f}"
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
            f"spaghetti recall is {fdm.get('recall', float('nan')):.3f} on FDM-sourced test images "
            f"(n={fdm.get('support', 0)}) vs {argus.get('recall', float('nan')):.3f} on argus_v2-sourced "
            f"test images (n={argus.get('support', 0)}), a gap of {gap:.3f}"
        )

    fit_for_pausing = bool(precision_ok and (confound_ok is not False))

    verdict_line = (
        "This model IS reasonably fit to drive automated print-pausing on the spaghetti path, subject to "
        "the caveats above."
        if fit_for_pausing
        else "This model is NOT fit to drive automated print-pausing (action_mode should stay notify_only)."
    )

    lines = [
        f"INTERPRETATION: {verdict_line}",
        f"  - Catastrophic-path precision: {precision_summary}.",
        f"  - Source confound: {confound_summary}. {confound['conclusion']}",
        "  - Only 'spaghetti' can ever drive an automated pause/cancel (config.example.yaml severity "
        "mapping); every other defect class is cosmetic-only, and 'normal' never emits a detection at "
        "all, so this catastrophic-path number is the entire automated-action false-positive story.",
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
    for cname in REPORT_CLASS_ORDER:
        e = report["per_class"][cname]  # type: ignore[index]
        tag = "  [CATASTROPHIC]" if cname == CATASTROPHIC_CLASS_NAME else ""
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
        f"  Real runtime rule: p_failure = spaghetti probability, ONLY when spaghetti is the model's "
        f"own top-1 prediction AND that probability clears the threshold."
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
    print("[5] SOURCE-CONFOUND DIAGNOSTIC -- spaghetti recall by source (the most important check)")
    sc = report["source_confound"]  # type: ignore[assignment]
    print(f"  Provenance method: {sc['method']}")
    for source, entry in sc["by_source"].items():
        recall_str = f"{entry['recall']:.3f}" if entry["recall"] is not None else "n/a"
        print(f"  {source:<12} support={entry['support']:>4}  correct={entry['correct']:>4}  recall={recall_str}")
    if sc["unknown_provenance_count"]:
        print(f"  WARNING: {sc['unknown_provenance_count']} spaghetti test image(s) had unrecoverable provenance.")
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
#          device, target_precision, min_recall, sweep_start/end/step, out). Notable defaults:
#          --split test, --target-precision 0.95, --min-recall 0.05 (the vacuous-precision
#          floor, same rationale as training/evaluate.py).
# Description: Defines and parses the CLI for evaluating the classification checkpoint.
# Side Effects: None (argparse may print usage/help and call sys.exit on bad input, but no
#               filesystem or network activity)
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help=f"Path to trained best.pt (default: {DEFAULT_WEIGHTS})")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR, help=f"Classification dataset root (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Which split to evaluate (default: test)")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="0")
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
# Description: CLI entry point. Lists the requested split's images, warns about classes with
#              zero images, recovers each image's source provenance, runs inference, builds the
#              full evaluation report, prints it, and writes it to disk as JSON.
# Side Effects: Raises FileNotFoundError if --weights or --data don't exist; raises
#               RuntimeError if the split has no images at all; runs a full GPU/CPU inference
#               pass reading every listed image from disk (see run_inference); prints progress,
#               warnings, and the full evaluation report to stdout; creates --out's parent
#               directory and writes the full JSON report to --out (default
#               runs/classifier_evaluation.json).
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.data.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {args.data}")

    print(f"[evaluate_classifier] Listing '{args.split}' split images under '{args.data}' ...")
    records = list_split_images(args.data, args.split, REPORT_CLASS_ORDER)
    if not records:
        raise RuntimeError(f"No images found for split '{args.split}' under '{args.data}'")

    present_classes = {c for _, c in records}
    missing_classes = [c for c in REPORT_CLASS_ORDER if c not in present_classes]
    if missing_classes:
        print(f"[evaluate_classifier] WARNING: these classes have ZERO images in split '{args.split}': {missing_classes}")

    paths = [p for p, _ in records]
    y_true = [c for _, c in records]
    sources = [infer_source(p.name) for p in paths]

    print(f"[evaluate_classifier] Running inference on {len(paths)} images with '{args.weights}' ...")
    probs, names_by_idx = run_inference(args.weights, paths, args.imgsz, args.batch, args.device)

    report = build_report(y_true, probs, names_by_idx, sources, args)
    print_report(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[evaluate_classifier] Full report written to: {args.out}")


if __name__ == "__main__":
    main()
