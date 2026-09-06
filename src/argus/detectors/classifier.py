"""ONNX whole-frame classifier detector: an alternative to
`onnx_yolo.OnnxYoloDetector` when a straight classifier outperforms an object
detector on a small dataset. Emits at most one whole-frame `Detection` per
inference (never localizes).

A `normal` prediction, or one below its per-class threshold, yields an empty
detection tuple -- so `DetectionResult.p_failure` is 0.0, same as no
detection at all. An unrecognized class name defaults to `Severity.COSMETIC`
so it can never drive a stop decision.
"""

from __future__ import annotations

import ast
import logging
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np
import onnxruntime as ort

from argus.config import DetectorConfig
from argus.detectors.base import Detector
from argus.types import Detection, DetectionResult, Frame, Severity

logger = logging.getLogger(__name__)

#: Absolute tolerances used by `probabilities_from_output` to decide whether
#: a raw model output already looks like a probability distribution.
_PROB_VALUE_EPS = 1e-6
_PROB_SUM_TOLERANCE = 1e-3


# --------------------------------------------------------------------------
# Pure pre/post-processing functions
# --------------------------------------------------------------------------


# np.ndarray preprocess_classify(np.ndarray image, int input_size)
# Inputs: np.ndarray image - source BGR frame (HxWx3, uint8) to preprocess
#         int input_size - target square spatial size the model expects
# Outputs: np.ndarray - a (1, 3, input_size, input_size) float32 blob in [0, 1], RGB,
#          channel-first
# Description: Resizes `image` so its short side equals `input_size`, center-crops to
#              input_size x input_size, converts BGR->RGB and HWC->CHW, scales to [0, 1], and
#              adds a batch dimension. Must exactly mirror the classification dataset builder's
#              training-time preprocessing (resize-short-side + center-crop) -- NOT the
#              letterboxing `onnx_yolo.preprocess` uses for the detection path.
# Side Effects: None (pure function of its inputs).
def preprocess_classify(image: np.ndarray, input_size: int) -> np.ndarray:
    """Must exactly match the classification dataset builder's training-time
    resize-short-side + center-crop (not the detection path's letterboxing) --
    diverging here silently skews every confidence the model produces."""
    orig_h, orig_w = image.shape[:2]
    short_side = min(orig_h, orig_w)
    scale = input_size / short_side

    # max(...) is a defensive clamp: mathematically the scaled short side is
    # always exactly input_size and the scaled long side is always >=
    # input_size, but round() could in principle land a hair under that due
    # to floating-point error, which would make the crop below go negative.
    resized_w = max(input_size, int(round(orig_w * scale)))
    resized_h = max(input_size, int(round(orig_h * scale)))

    if (resized_w, resized_h) != (orig_w, orig_h):
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image

    top = (resized_h - input_size) // 2
    left = (resized_w - input_size) // 2
    cropped = resized[top : top + input_size, left : left + input_size]

    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1)
    blob = np.ascontiguousarray(chw, dtype=np.float32) / 255.0
    return blob[np.newaxis, ...]


# np.ndarray softmax(np.ndarray logits)
# Inputs: np.ndarray logits - raw scores, any shape, softmax applied along the last axis
# Outputs: np.ndarray - probability distribution(s) of the same shape as `logits`, summing to
#          1.0 along the last axis
# Description: Numerically stable softmax: subtracts the per-row max before exponentiating so
#              large logits cannot overflow `exp`.
# Side Effects: None (pure function of its input).
def softmax(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    shifted = arr - np.max(arr, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


# np.ndarray probabilities_from_output(np.ndarray raw)
# Inputs: np.ndarray raw - raw classifier output row(s), shape (N,) or (B, N)
# Outputs: np.ndarray - a valid probability distribution of the same shape, summing to 1.0
#          along the last axis
# Description: Detects whether `raw` already looks like a probability distribution (every
#              value in [0, 1] within a small tolerance, each row summing to 1.0 within 1e-3)
#              and returns it unchanged if so; otherwise treats it as raw logits and applies
#              `softmax`. This avoids double-softmax-ing Ultralytics classification heads (and
#              similar exports) that already apply softmax internally -- softmax-ing an
#              already-softmaxed output would flatten sharp confidences toward uniform and
#              silently wreck every configured threshold.
# Side Effects: None (pure function of its input).
def probabilities_from_output(raw: np.ndarray) -> np.ndarray:
    """Ultralytics classification heads already output softmaxed
    probabilities; softmax-ing them again would flatten sharp confidences
    toward uniform and silently invalidate every configured threshold. Only
    applies `softmax` when `raw` doesn't already look like a distribution."""
    arr = np.asarray(raw, dtype=np.float64)
    in_unit_interval = bool(
        np.all(arr >= -_PROB_VALUE_EPS) and np.all(arr <= 1.0 + _PROB_VALUE_EPS)
    )
    sums = np.sum(arr, axis=-1)
    sums_near_one = bool(np.all(np.abs(sums - 1.0) <= _PROB_SUM_TOLERANCE))

    if in_unit_interval and sums_near_one:
        return arr
    return softmax(arr)


# tuple[Detection, ...] postprocess_classify(np.ndarray raw, Sequence[str] class_names, Mapping[str, float] class_thresholds, float default_threshold, Mapping[str, Severity] severity_map, tuple[int, int] orig_shape)
# Inputs: np.ndarray raw - raw classifier ONNX output, shape (N,) or (1, N) where
#                 N == len(class_names)
#         Sequence[str] class_names - ordered class names matching the model's output index
#                 order
#         Mapping[str, float] class_thresholds - per-class-name confidence threshold overrides
#         float default_threshold - threshold used for any class name absent from
#                 `class_thresholds`
#         Mapping[str, Severity] severity_map - per-class-name Severity; a name absent from it
#                 defaults to Severity.COSMETIC so an unrecognized class can never drive a stop
#                 decision
#         tuple[int, int] orig_shape - (height, width) of the original frame, used to build the
#                 whole-frame placeholder bbox
# Outputs: tuple[Detection, ...] - empty, or exactly one Detection for the predicted class
# Description: Converts `raw` to probabilities via `probabilities_from_output`, takes the
#              argmax as the predicted class, and returns `()` if the prediction is "normal"
#              (case-insensitive -- a classifier's `normal` class has no "not a defect"
#              Severity to map to) or below its configured threshold (an uncertain prediction
#              must never be able to stop a print). Otherwise returns a single Detection whose
#              bbox spans the whole frame (0, 0, W, H) (a classifier localizes nothing; the box
#              is a placeholder for downstream notification-image code written against the
#              detection path's bbox contract), confidence is the predicted class's
#              probability, and severity comes from `severity_map`. Raises `ValueError` if
#              `raw`'s shape doesn't match `class_names`.
# Side Effects: None (pure function of its inputs).
def postprocess_classify(
    raw: np.ndarray,
    class_names: Sequence[str],
    class_thresholds: Mapping[str, float],
    default_threshold: float,
    severity_map: Mapping[str, Severity],
    orig_shape: tuple[int, int],
) -> tuple[Detection, ...]:
    arr = np.asarray(raw)
    if arr.ndim == 2:
        if arr.shape[0] != 1:
            raise ValueError(f"expected a batch size of 1, got raw output shape {raw.shape}")
        arr = arr[0]
    elif arr.ndim != 1:
        raise ValueError(
            f"unexpected classifier raw output shape {raw.shape}, expected (N,) or (1, N)"
        )

    num_classes = len(class_names)
    if arr.shape[0] != num_classes:
        raise ValueError(
            f"classifier raw output has {arr.shape[0]} classes but class_names has "
            f"{num_classes} entries {tuple(class_names)}; these must match exactly"
        )

    probs = probabilities_from_output(arr)
    pred_idx = int(np.argmax(probs))
    pred_name = class_names[pred_idx]
    pred_confidence = float(probs[pred_idx])

    if pred_name.strip().lower() == "normal":
        return ()

    threshold = class_thresholds.get(pred_name, default_threshold)
    if pred_confidence < threshold:
        return ()

    severity = severity_map.get(pred_name, Severity.COSMETIC)
    orig_h, orig_w = orig_shape[0], orig_shape[1]
    bbox = (0.0, 0.0, float(orig_w), float(orig_h))

    detection = Detection(
        class_id=pred_idx,
        class_name=pred_name,
        confidence=pred_confidence,
        bbox=bbox,
        severity=severity,
    )
    return (detection,)


# Optional[tuple[str, ...]] class_names_from_onnx_metadata(Any session)
# Inputs: Any session - the onnxruntime InferenceSession (or a duck-typed fake in tests)
#                 whose embedded class-name metadata should be read
# Outputs: Optional[tuple[str, ...]] - class names ordered by index, or None if the metadata is
#          missing, unparseable, empty, or has non-dense integer keys
# Description: Best-effort extraction of the model's own training-time class-name order from
#              its ONNX `metadata_props` (`session.get_modelmeta().custom_metadata_map["names"]`,
#              a Python dict-repr string Ultralytics writes at export time), so a wrong
#              `class_names` order in config is never the only line of defense. Deliberately
#              defensive: any exception while reading or parsing is caught and logged rather
#              than allowed to crash startup.
# Side Effects: Logs a warning (via the module logger) if the metadata is present but fails to
#               parse; otherwise none.
def class_names_from_onnx_metadata(session: Any) -> Optional[tuple[str, ...]]:
    """Reads Ultralytics' `names` dict-repr string from
    `custom_metadata_map`, so a wrong `class_names` order in config is never
    the only line of defense. Returns None (rather than a partial/garbled
    result) if the entry is missing, unparseable, empty, or its keys aren't
    a dense `0..N-1` range."""
    try:
        modelmeta = session.get_modelmeta()
        custom_metadata = getattr(modelmeta, "custom_metadata_map", None) or {}
        names_raw = custom_metadata.get("names")
        if not names_raw:
            return None
        parsed = ast.literal_eval(names_raw)
        if not isinstance(parsed, dict) or not parsed:
            return None
        indices = sorted(parsed.keys())
        if indices != list(range(len(parsed))):
            return None
        return tuple(str(parsed[i]) for i in indices)
    except Exception:
        logger.warning(
            "ClassifierDetector: failed to parse class-name metadata from the ONNX model's "
            "custom_metadata_map['names']; treating it as absent",
            exc_info=True,
        )
        return None


# tuple[str, ...] resolve_class_names(Any session, Sequence[str] configured_class_names)
# Inputs: Any session - the onnxruntime InferenceSession (or duck-typed fake) to read
#                 embedded class-name metadata from
#         Sequence[str] configured_class_names - `DetectorConfig.class_names` as configured
#                 (may be empty)
# Outputs: tuple[str, ...] - the class-name order to use for this detector instance
# Description: Reconciles `configured_class_names` against
#              `class_names_from_onnx_metadata(session)` per the rules in this function's own
#              docstring: metadata's order wins when config is empty; matching values are
#              returned as independently-confirmed; a genuine disagreement raises `ValueError`
#              naming both orderings rather than silently picking one; metadata absent falls
#              back to config with a logged warning; both absent raises `ValueError`.
# Side Effects: Logs an info or warning message (via the module logger) describing which source
#               of truth was used.
def resolve_class_names(session: Any, configured_class_names: Sequence[str]) -> tuple[str, ...]:
    configured = tuple(configured_class_names)
    metadata = class_names_from_onnx_metadata(session)

    if metadata is None:
        if not configured:
            raise ValueError(
                "ClassifierDetector requires cfg.class_names to be a non-empty tuple when the "
                "ONNX model carries no usable class-name metadata (set detector.class_names in "
                "config)"
            )
        logger.warning(
            "ClassifierDetector: could not read class-name metadata from the ONNX model "
            "(missing or unparseable 'names' entry in custom_metadata_map) -- falling back to "
            "configured detector.class_names=%s UNVERIFIED. If this order doesn't match the "
            "model's actual training-time class order, predictions will be silently mislabeled.",
            configured,
        )
        return configured

    if not configured:
        logger.info(
            "ClassifierDetector: detector.class_names not set in config -- using the ONNX "
            "model's own embedded class-name order %s",
            metadata,
        )
        return metadata

    if configured != metadata:
        raise ValueError(
            "detector.class_names in config disagrees with the class-name order embedded in "
            f"the ONNX model's own metadata. Configured detector.class_names: {list(configured)}. "
            f"Model's actual order (from ONNX metadata; Ultralytics assigns indices "
            f"alphabetically from training folder names): {list(metadata)}. Fix "
            "detector.class_names in config to match the model's order -- getting this wrong "
            "silently mislabels every prediction (see argus.detectors.classifier module "
            "docstring)."
        )

    return configured


# Optional[int] _static_input_size(Optional[Sequence[object]] shape)
# Inputs: Optional[Sequence[object]] shape - the ONNX-declared input shape (e.g. from
#                 `session.get_inputs()[0].shape`), possibly containing symbolic/dynamic dims
# Outputs: Optional[int] - the static square spatial size (e.g. 512), or None if the shape is
#          missing, not 4D, dynamic, or non-square
# Description: Extracts a usable static input size from a declared ONNX input shape so the
#              caller can fall back to `cfg.input_size` when the model's shape doesn't pin one
#              down. Duplicated from `onnx_yolo._static_input_size` rather than imported --
#              that helper is private to that module and this one is deliberately kept free of
#              any dependency on the detection-path module.
# Side Effects: None (pure function of its input).
def _static_input_size(shape: Optional[Sequence[object]]) -> Optional[int]:
    # Duplicated from onnx_yolo._static_input_size rather than imported: that
    # helper is private, and this module stays free of detection-path deps.
    if shape is None or len(shape) != 4:
        return None
    h, w = shape[2], shape[3]
    if isinstance(h, int) and isinstance(w, int) and h > 0 and h == w:
        return h
    return None


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------


class ClassifierDetector(Detector):
    """`Detector` implementation backed by an ONNX whole-frame classifier.
    `cfg.class_names` is reconciled against the model's own ONNX metadata via
    `resolve_class_names`, so a mismatched order fails loudly at startup
    instead of silently mislabeling every prediction.
    """

    # None __init__(DetectorConfig cfg)
    # Inputs: DetectorConfig cfg - detector configuration: model path, providers, input size,
    #                 class names, thresholds, and severity map
    # Outputs: None
    # Description: Loads the ONNX classifier model into an onnxruntime InferenceSession,
    #              resolves the class-name order via `resolve_class_names` (reconciled against
    #              the model's own embedded metadata), and determines the model's input size
    #              (its static declared shape if available, else `cfg.input_size`).
    # Side Effects: Reads `cfg.model_path` from disk; raises `FileNotFoundError` if it doesn't
    #               exist. Constructs an `onnxruntime.InferenceSession` (allocates model
    #               resources). Logs an info message. Mutates the new instance's state
    #               (`_cfg`, `_session`, `_input_name`, `_class_names`, `_input_size`).
    def __init__(self, cfg: DetectorConfig) -> None:
        self._cfg = cfg

        model_path = Path(cfg.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"ONNX classifier model not found at '{model_path}' -- has it been trained/exported yet?"
            )

        self._session: Optional[ort.InferenceSession] = ort.InferenceSession(
            str(model_path), providers=list(cfg.providers)
        )
        self._input_name = self._session.get_inputs()[0].name

        self._class_names = resolve_class_names(self._session, cfg.class_names)

        static_size = _static_input_size(self._session.get_inputs()[0].shape)
        self._input_size = static_size if static_size is not None else cfg.input_size
        logger.info(
            "ClassifierDetector: using input size %d (%s), classes=%s",
            self._input_size,
            "from model's static input shape" if static_size is not None else "from config.input_size",
            self._class_names,
        )

    # DetectionResult infer(Frame frame)
    # Inputs: Frame frame - the captured camera frame to classify
    # Outputs: DetectionResult - zero or one Detection (see `postprocess_classify`), plus the
    #          measured inference time in milliseconds
    # Description: Runs the full classify pipeline for one frame: `preprocess_classify` builds
    #              the model input blob, the ONNX session runs inference, and
    #              `postprocess_classify` turns the raw output into a `DetectionResult`. Only a
    #              CATASTROPHIC-severity detection here would ever raise `p_failure`; a
    #              `normal` prediction produces no detection at all (see module docstring).
    # Side Effects: Runs ONNX model inference (CPU/GPU work via `self._session.run`); reads
    #               `time.perf_counter()` to measure elapsed time. Raises `RuntimeError` if the
    #               detector has already been `close()`d.
    def infer(self, frame: Frame) -> DetectionResult:
        if self._session is None:
            raise RuntimeError("ClassifierDetector is closed")

        start = time.perf_counter()
        blob = preprocess_classify(frame.image, self._input_size)
        raw = self._session.run(None, {self._input_name: blob})[0]

        detections = postprocess_classify(
            raw,
            self._class_names,
            self._cfg.class_thresholds,
            self._cfg.default_threshold,
            self._cfg.severity,
            frame.image.shape[:2],
        )
        inference_ms = (time.perf_counter() - start) * 1000.0
        return DetectionResult(detections=detections, inference_ms=inference_ms)

    # None close()
    # Inputs: None
    # Outputs: None
    # Description: Releases the ONNX InferenceSession so `infer` can no longer be called on
    #              this instance.
    # Side Effects: Mutates instance state, dropping the reference to `self._session` (allowing
    #               the underlying onnxruntime resources to be garbage-collected). Subsequent
    #               `infer` calls will raise `RuntimeError`.
    def close(self) -> None:
        self._session = None
