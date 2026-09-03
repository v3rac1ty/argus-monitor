"""ONNX whole-frame classifier detector.

This is an alternative to `onnx_yolo.OnnxYoloDetector` for the case where a
small dataset makes a straight image classifier more accurate than an object
detector. The `DecisionEngine` only ever consumes `DetectionResult.p_failure`
(the max confidence among *catastrophic*-severity detections) -- it never
looks at bounding boxes -- so a classifier that never localizes anything can
still drive the whole system: it just emits at most one whole-frame
`Detection` per inference, or none at all.

As with `onnx_yolo.py`, all tensor pre/post-processing lives in module-level
*pure functions* (`preprocess_classify`, `softmax`, `probabilities_from_output`,
`postprocess_classify`) that operate on plain numpy arrays and are fully
unit-testable with synthetic arrays and no model file on disk.
`ClassifierDetector` is a thin wrapper that owns the
`onnxruntime.InferenceSession` and glues those pure functions to the
`Detector` interface.

How `p_failure` is derived on this path
----------------------------------------
`Severity` (in `argus.types`, a frozen contract this module does not modify)
has only `CATASTROPHIC` and `COSMETIC` -- there is no "not a defect" value.
Rather than force the trained `normal` class into one of those two buckets,
a `normal` prediction is instead excluded entirely: `postprocess_classify`
returns an empty detection tuple for it. An empty tuple means
`DetectionResult.catastrophic` is empty too, so `DetectionResult.p_failure`
(defined in `argus.types`) is `0.0` -- exactly as if nothing had been
detected. The same empty-tuple treatment applies to any non-`normal`
prediction whose confidence falls below its configured per-class threshold
(the model could be right, but "uncertain" must never be able to stop a
print). For every other prediction, exactly one `Detection` is emitted whose
`confidence` is the predicted class's probability and whose `severity` comes
from `DetectorConfig.severity` (defaulting to `Severity.COSMETIC` for a name
that config doesn't recognize, so an unknown class can never drive a stop
decision); `DetectionResult.p_failure` then picks that confidence up only if
`severity` is `CATASTROPHIC`, via the ordinary logic already in
`argus.types.DetectionResult`.
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


def preprocess_classify(image: np.ndarray, input_size: int) -> np.ndarray:
    """Resize-short-side-then-center-crop `image` into a model-ready blob.

    **This must match the training-time preprocessing exactly.** The
    classification dataset builder resizes each image so its short side is
    512px, then center-crops to 512x512 -- i.e. a standard
    resize-short-side + center-crop pipeline, not the letterboxing
    `onnx_yolo.preprocess` uses for the detection path. Diverging from that
    here (different interpolation, resizing the long side instead, a
    non-centered crop, etc.) would silently skew every confidence the model
    produces, so if the dataset builder's resize/crop logic ever changes,
    this function must change with it.

    Steps: resize so `min(height, width) == input_size` (preserving aspect
    ratio), center-crop to `input_size x input_size`, BGR -> RGB,
    HWC -> CHW, cast to float32 and scale to [0, 1], then add a batch
    dimension.

    Returns a `(1, 3, input_size, input_size)` float32 array in [0, 1].
    """
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


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis.

    Subtracts the per-row max before exponentiating so large logits (e.g.
    from an unbounded classification head) cannot overflow `exp`.
    """
    arr = np.asarray(logits, dtype=np.float64)
    shifted = arr - np.max(arr, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def probabilities_from_output(raw: np.ndarray) -> np.ndarray:
    """Turn a raw classifier output row into a probability distribution.

    Ultralytics classification heads (and many other classifier exports)
    already apply softmax internally, so their ONNX output is already a
    valid probability distribution. Blindly softmax-ing such an output again
    would flatten already-sharp confidences toward uniform and silently
    wreck every configured threshold. This function instead detects whether
    `raw` already looks like a probability distribution -- every value in
    `[0, 1]` (within a small float tolerance) and each row summing to `1.0`
    within `1e-3` -- and only applies `softmax` when it does not (i.e. `raw`
    looks like raw, unnormalized logits).

    Works on both a single row (shape `(N,)`) and a batch (shape `(B, N)`);
    the checks are applied along the last axis.
    """
    arr = np.asarray(raw, dtype=np.float64)
    in_unit_interval = bool(
        np.all(arr >= -_PROB_VALUE_EPS) and np.all(arr <= 1.0 + _PROB_VALUE_EPS)
    )
    sums = np.sum(arr, axis=-1)
    sums_near_one = bool(np.all(np.abs(sums - 1.0) <= _PROB_SUM_TOLERANCE))

    if in_unit_interval and sums_near_one:
        return arr
    return softmax(arr)


def postprocess_classify(
    raw: np.ndarray,
    class_names: Sequence[str],
    class_thresholds: Mapping[str, float],
    default_threshold: float,
    severity_map: Mapping[str, Severity],
    orig_shape: tuple[int, int],
) -> tuple[Detection, ...]:
    """Turn a raw classifier ONNX output into zero or one `Detection`.

    Accepts `raw` shaped `(N,)` or `(1, N)` where `N == len(class_names)`;
    raises `ValueError` if `N` doesn't match. Converts to probabilities via
    `probabilities_from_output`, then takes the argmax as the predicted
    class:

    - if the predicted class is `"normal"` (case-insensitive) -> `()`. A
      classifier trained on a `normal` class has no "not a defect"
      `Severity` to map to (see module docstring), so `normal` predictions
      are excluded from the output entirely rather than tagged with either
      `Severity` value.
    - otherwise, if the predicted class's probability is below its
      configured threshold (`class_thresholds.get(name, default_threshold)`)
      -> `()`. An uncertain prediction must never be able to drive a stop
      decision.
    - otherwise -> a single `Detection` whose `bbox` spans the *whole
      frame* `(0, 0, W, H)` (a classifier localizes nothing; the box is a
      placeholder so downstream notification-image code, written against
      the detection path's bbox contract, still has something to draw),
      `confidence` is the predicted class's probability, and `severity`
      comes from `severity_map` (defaulting to `Severity.COSMETIC` for a
      name absent from it, so an unrecognized class can never drive the
      stop decision).
    """
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


def class_names_from_onnx_metadata(session: Any) -> Optional[tuple[str, ...]]:
    """Best-effort extraction of the model's own class-name order from ONNX
    metadata, so a wrong `class_names` order in config is never the only
    line of defense (see the class-order pitfall described in
    `training/evaluate_classifier.py` and `training/export_classifier_onnx.py`).

    Ultralytics writes the training-time class mapping into the exported
    model's `metadata_props` as a `names` entry -- a Python dict-repr
    string such as `"{0: 'cracking', 1: 'layer_shifting', ..., 5:
    'warping'}"`, indexed exactly the way the model's output columns are
    (Ultralytics assigns indices alphabetically from the training folder
    names). onnxruntime surfaces `metadata_props` as
    `session.get_modelmeta().custom_metadata_map`, a plain `dict[str, str]`.

    `session` is accepted as `Any` (duck-typed) rather than
    `ort.InferenceSession` so this also works against a lightweight fake
    session in tests, with no real model file required.

    Returns a tuple ordered by index, or `None` if:
      - `custom_metadata_map` has no `names` entry (or it's falsy), or
      - the entry doesn't parse as a Python literal, or
      - it doesn't parse to a non-empty dict, or
      - its keys aren't exactly the dense integer range `0..N-1` (a partial
        or garbled mapping can't be trusted to reconstruct index order).

    In every such case the caller must fall back to config rather than
    trust a partial/garbled read. This function is deliberately defensive:
    any exception while reading or parsing the metadata is caught and
    logged, never allowed to crash startup over some other exporter's
    model having a metadata quirk.
    """
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


def resolve_class_names(session: Any, configured_class_names: Sequence[str]) -> tuple[str, ...]:
    """Reconcile `configured_class_names` (from `DetectorConfig.class_names`)
    against `class_names_from_onnx_metadata(session)` so a wrong config
    order can never silently mislabel every prediction:

    - metadata present, config empty -> the metadata's order (the best
      default: it comes straight from the model that produced it, so there
      is no way for a human to get it wrong).
    - metadata present, config non-empty, and they DISAGREE -> raises
      `ValueError` naming both orderings explicitly, so the operator knows
      exactly which one to fix. The mismatch is never silently resolved in
      either direction.
    - metadata present, config non-empty, and they agree -> the configured
      order, now independently confirmed correct.
    - metadata absent or unparseable -> the configured order, with a
      logged warning that it could not be independently verified.
    - metadata absent AND config empty -> raises `ValueError` (nothing to
      go on at all).
    """
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


def _static_input_size(shape: Optional[Sequence[object]]) -> Optional[int]:
    """Return the model's static square spatial input size (e.g. 512 for an
    input shaped `[1, 3, 512, 512]`), or None if the shape is missing,
    non-4D, dynamic (a symbolic dim shows up as a non-int), or non-square --
    any of which means the caller should fall back to config.

    (Duplicated from `onnx_yolo._static_input_size` rather than imported --
    that helper is private to that module and this one is deliberately kept
    free of any dependency on the detection-path module.)
    """
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

    `cfg.class_names` (the ordered class list matching the model's output
    index order) is reconciled against the class-name order embedded in the
    ONNX model's own metadata via `resolve_class_names` -- see that
    function's docstring for the exact rules. In short: an empty config
    value defers to the model's metadata when available; a non-empty value
    that disagrees with the metadata raises `ValueError` rather than
    silently preferring either one. This makes the config-order mixup
    described in the module docstring (and `training/evaluate_classifier.
    py`'s class-order pitfall) something that fails loudly at startup
    instead of silently mislabeling every prediction.
    """

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

    def close(self) -> None:
        self._session = None
