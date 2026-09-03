"""ONNX detector supporting both YOLOv8 and YOLO26 output contracts.

All tensor pre/post-processing lives in module-level *pure functions* that
operate on plain numpy arrays (`letterbox`, `preprocess`, `postprocess`,
`postprocess_end2end`, `detect_layout`, and the private NMS helpers). This
keeps them fully unit-testable with synthetic arrays and no model file on
disk. `OnnxYoloDetector` is a thin wrapper that owns the
`onnxruntime.InferenceSession` and glues those pure functions to the
`Detector` interface; the session is only constructed in `__init__`, and
only when a real model path is given.

Two output contracts are supported, selected via `DetectorConfig.layout`:

- **YOLOv8** (`"yolov8"`): raw predictions shaped `(1, 4 + num_classes,
  num_anchors)` -- transposed relative to YOLOv5, and with no separate
  objectness channel: the per-class scores at columns 4.. are already the
  final confidences. `postprocess` handles both this layout and its
  transpose, then applies per-class thresholds and class-aware NMS.
- **YOLO26** (`"end2end"`): the model is NMS-free / end-to-end and its ONNX
  export already does the decoding -- output is `(1, max_det, 6)`, each row
  `[x1, y1, x2, y2, confidence, class_id]` in letterboxed input-pixel space,
  sorted by descending confidence and padded with zero/low-confidence rows.
  `postprocess_end2end` handles this: no transpose, no argmax, and
  critically **no NMS** (the model already did it -- redoing it here would
  silently drop valid adjacent detections).

`"auto"` (the default) picks between them via `detect_layout` -- see that
function's docstring for the shape heuristic and its one documented
ambiguity (exactly 2 classes).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import cv2
import numpy as np
import onnxruntime as ort

from argus.config import DetectorConfig
from argus.detectors.base import Detector
from argus.types import Detection, DetectionResult, Frame, Severity

logger = logging.getLogger(__name__)

#: Class order the model was trained with -- used when no explicit
#: `class_names` is passed to `OnnxYoloDetector`.
DEFAULT_CLASS_NAMES: tuple[str, ...] = (
    "error extrusion",
    "spaghetti",
    "stringing",
    "warping",
    "zits",
)

_LETTERBOX_COLOR: tuple[int, int, int] = (114, 114, 114)


# --------------------------------------------------------------------------
# Pure pre/post-processing functions
# --------------------------------------------------------------------------


def letterbox(
    image: np.ndarray,
    new_shape: Union[int, tuple[int, int]],
    color: tuple[int, int, int] = _LETTERBOX_COLOR,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize `image` preserving aspect ratio and pad to `new_shape` with grey.

    Returns `(padded_image, scale, (pad_w, pad_h))` where `scale` is the
    factor the original image was resized by and `(pad_w, pad_h)` is the
    (possibly fractional) left/top padding added -- exactly what's needed to
    invert the transform: `orig = (letterboxed - pad) / scale`.
    """
    if isinstance(new_shape, int):
        target_h, target_w = new_shape, new_shape
    else:
        target_h, target_w = new_shape

    orig_h, orig_w = image.shape[:2]
    scale = min(target_w / orig_w, target_h / orig_h)

    resized_w = int(round(orig_w * scale))
    resized_h = int(round(orig_h * scale))
    if (resized_w, resized_h) != (orig_w, orig_h):
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image

    dw = (target_w - resized_w) / 2.0
    dh = (target_h - resized_h) / 2.0

    # Split (possibly odd) total padding across both sides the same way
    # Ultralytics' own letterbox does, so a fractional half-pad still lands
    # the resized image on an integer pixel grid.
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return padded, scale, (dw, dh)


def preprocess(image: np.ndarray, input_size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Letterbox `image` to a square `input_size` and build a model-ready blob.

    Returns `(blob, scale, pad)` where `blob` is `(1, 3, H, W)` float32 in
    [0, 1], RGB, channel-first -- and `scale`/`pad` are passed through
    unchanged from `letterbox` for later use by `postprocess`.
    """
    padded, scale, pad = letterbox(image, input_size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1)
    blob = np.ascontiguousarray(chw, dtype=np.float32) / 255.0
    blob = blob[np.newaxis, ...]
    return blob, scale, pad


def _to_rows(raw: np.ndarray, expected_cols: int) -> np.ndarray:
    """Normalize a raw YOLOv8 output to a 2D `(num_anchors, 4 + num_classes)`
    array, handling both the native `(1, 4+nc, N)` layout and an
    already-transposed `(1, N, 4+nc)` one (or either without the batch dim)."""
    arr = np.asarray(raw)

    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"expected a batch size of 1, got raw output shape {raw.shape}")
        arr = arr[0]
    elif arr.ndim != 2:
        raise ValueError(f"unexpected raw output shape {raw.shape}")

    if arr.shape[1] == expected_cols:
        return arr
    if arr.shape[0] == expected_cols:
        return arr.T
    raise ValueError(
        f"raw output shape {raw.shape} doesn't match 4 + num_classes ({expected_cols}) on either axis"
    )


def postprocess(
    raw: np.ndarray,
    scale: float,
    pad: tuple[float, float],
    orig_shape: tuple[int, int],
    class_names: Sequence[str],
    class_thresholds: Mapping[str, float],
    default_threshold: float,
    severity_map: Mapping[str, Severity],
    nms_iou: float,
) -> tuple[Detection, ...]:
    """Turn a raw YOLOv8 ONNX output tensor into final `Detection`s.

    `scale`/`pad` (from `preprocess`/`letterbox`) invert the letterbox back
    to original-frame pixel coordinates. Per-class thresholds gate which
    boxes survive at all; class-aware NMS (different classes never suppress
    each other) then dedupes overlapping boxes of the same class. A class
    name absent from `severity_map` defaults to `Severity.COSMETIC` so an
    unrecognized class can never drive the stop decision.
    """
    num_classes = len(class_names)
    rows = _to_rows(raw, 4 + num_classes)

    if rows.shape[0] == 0:
        return ()

    boxes_cxcywh = rows[:, :4]
    class_scores = rows[:, 4 : 4 + num_classes]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]

    thresholds = np.array(
        [class_thresholds.get(class_names[c], default_threshold) for c in class_ids],
        dtype=np.float64,
    )
    keep_mask = confidences >= thresholds
    if not np.any(keep_mask):
        return ()

    boxes_cxcywh = boxes_cxcywh[keep_mask]
    class_ids = class_ids[keep_mask]
    confidences = confidences[keep_mask]

    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    pad_w, pad_h = pad
    x1 = (x1 - pad_w) / scale
    x2 = (x2 - pad_w) / scale
    y1 = (y1 - pad_h) / scale
    y2 = (y2 - pad_h) / scale

    orig_h, orig_w = orig_shape[0], orig_shape[1]
    x1 = np.clip(x1, 0, orig_w - 1)
    x2 = np.clip(x2, 0, orig_w - 1)
    y1 = np.clip(y1, 0, orig_h - 1)
    y2 = np.clip(y2, 0, orig_h - 1)

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep_idx = _class_aware_nms(boxes_xyxy, confidences, class_ids, nms_iou)

    detections = []
    for i in keep_idx:
        cid = int(class_ids[i])
        name = class_names[cid]
        severity = severity_map.get(name, Severity.COSMETIC)
        bbox = (
            float(boxes_xyxy[i, 0]),
            float(boxes_xyxy[i, 1]),
            float(boxes_xyxy[i, 2]),
            float(boxes_xyxy[i, 3]),
        )
        detections.append(
            Detection(
                class_id=cid,
                class_name=name,
                confidence=float(confidences[i]),
                bbox=bbox,
                severity=severity,
            )
        )
    return tuple(detections)


def detect_layout(raw: np.ndarray, num_classes: int) -> str:
    """Infer which ONNX output contract `raw` follows: `"yolov8"` (raw,
    NMS-pending predictions) or `"end2end"` (YOLO26's decoded, NMS-free
    output).

    Heuristic: squeeze the batch dim, then

    1. if any remaining axis equals `4 + num_classes` -> `"yolov8"`
    2. elif the last axis is exactly `6` -> `"end2end"`
    3. else raise `ValueError` naming the actual shape and `num_classes`.

    **Ambiguity:** when `num_classes == 2`, `4 + num_classes == 6`, so a
    `(*, 6)`-shaped output could legitimately be either layout and shape
    alone cannot disambiguate it. Check (1) runs first, so this case always
    resolves to `"yolov8"`. If you train a 2-class model and actually mean
    the end-to-end layout, do not rely on `"auto"` -- set `detector.layout`
    to `"end2end"` explicitly in config.
    """
    arr = np.asarray(raw)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"expected a batch size of 1, got raw output shape {raw.shape}")
        arr = arr[0]

    expected_cols = 4 + num_classes
    if expected_cols in arr.shape:
        return "yolov8"
    if arr.shape[-1] == 6:
        return "end2end"

    raise ValueError(
        f"cannot determine detector output layout from raw shape {raw.shape} with "
        f"num_classes={num_classes} (expected an axis of size 4+num_classes="
        f"{expected_cols} for the yolov8 layout, or a last axis of size 6 for the "
        "end2end layout) -- set detector.layout explicitly instead of 'auto'"
    )


def _probe_layout_shape(declared_shape: Optional[Sequence[object]]) -> tuple[int, ...]:
    """Turn an ONNX-declared output shape (whose batch/anchor dims may be
    dynamic and show up as strings, `None`, or negative sentinels) into a
    concrete shape usable to build a zero-filled probe array for
    `detect_layout`. Non-batch dynamic dims are replaced with `0`, which can
    never coincidentally equal `4 + num_classes` or `6` for a real model, so
    they never cause a false match. The batch dim (axis 0) is always forced
    to `1` regardless of what's declared, since every real inference call
    here uses a batch of exactly 1 -- leaving a dynamic batch dim as `0`
    would make `detect_layout` reject the probe outright."""
    if not declared_shape:
        return ()
    dims = [d if isinstance(d, int) and d > 0 else 0 for d in declared_shape]
    dims[0] = 1
    return tuple(dims)


_VALID_LAYOUTS = {"yolov8", "end2end"}


def _resolve_layout(configured: str, declared_output_shape: Optional[Sequence[object]], num_classes: int) -> str:
    """Resolve the concrete layout ("yolov8" or "end2end") to use for every
    inference: the configured value verbatim if it's explicit, otherwise
    `detect_layout`'s heuristic run once against the model's declared ONNX
    output shape (`declared_output_shape`, from
    `session.get_outputs()[0].shape`)."""
    if configured != "auto":
        if configured not in _VALID_LAYOUTS:
            raise ValueError(f"detector.layout must be one of {sorted(_VALID_LAYOUTS)}, got {configured!r}")
        return configured
    probe = np.zeros(_probe_layout_shape(declared_output_shape), dtype=np.float32)
    return detect_layout(probe, num_classes)


def postprocess_end2end(
    raw: np.ndarray,
    scale: float,
    pad: tuple[float, float],
    orig_shape: tuple[int, int],
    class_names: Sequence[str],
    class_thresholds: Mapping[str, float],
    default_threshold: float,
    severity_map: Mapping[str, Severity],
) -> tuple[Detection, ...]:
    """Turn a raw YOLO26 end-to-end ONNX output tensor into final
    `Detection`s.

    Expects `(1, max_det, 6)` (or `(max_det, 6)`), each row already decoded
    as `[x1, y1, x2, y2, confidence, class_id]` in letterboxed input-pixel
    space -- no transpose, no argmax needed. Padded rows (confidence <= 0,
    or a class id outside `range(len(class_names))` -- a padded row can
    carry a garbage class index) are dropped first. Per-class thresholds
    then gate which of the remaining boxes survive, exactly as in
    `postprocess`. `scale`/`pad` invert the letterbox back to
    original-frame pixel coordinates, and coordinates are clipped to the
    original frame bounds since decoded boxes can fall slightly outside
    `[0, input_size]`.

    Deliberately does **no NMS**: the model is NMS-free/end-to-end and
    already deduplicated its own output, so two heavily-overlapping
    same-class boxes here are both genuine and both kept -- unlike
    `postprocess`'s YOLOv8 path.  A class name absent from `severity_map`
    defaults to `Severity.COSMETIC` so an unrecognized class can never drive
    the stop decision.
    """
    arr = np.asarray(raw)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"expected a batch size of 1, got raw output shape {raw.shape}")
        arr = arr[0]
    elif arr.ndim != 2:
        raise ValueError(f"unexpected raw output shape {raw.shape}")

    if arr.shape[-1] != 6:
        raise ValueError(f"expected end2end output rows of width 6, got shape {raw.shape}")

    if arr.shape[0] == 0:
        return ()

    boxes_xyxy = arr[:, 0:4].astype(np.float64)
    confidences = arr[:, 4].astype(np.float64)
    class_ids = np.rint(arr[:, 5]).astype(np.int64)

    num_classes = len(class_names)
    valid_mask = (confidences > 0.0) & (class_ids >= 0) & (class_ids < num_classes)
    if not np.any(valid_mask):
        return ()

    boxes_xyxy = boxes_xyxy[valid_mask]
    confidences = confidences[valid_mask]
    class_ids = class_ids[valid_mask]

    thresholds = np.array(
        [class_thresholds.get(class_names[c], default_threshold) for c in class_ids],
        dtype=np.float64,
    )
    keep_mask = confidences >= thresholds
    if not np.any(keep_mask):
        return ()

    boxes_xyxy = boxes_xyxy[keep_mask]
    confidences = confidences[keep_mask]
    class_ids = class_ids[keep_mask]

    pad_w, pad_h = pad
    x1 = (boxes_xyxy[:, 0] - pad_w) / scale
    y1 = (boxes_xyxy[:, 1] - pad_h) / scale
    x2 = (boxes_xyxy[:, 2] - pad_w) / scale
    y2 = (boxes_xyxy[:, 3] - pad_h) / scale

    orig_h, orig_w = orig_shape[0], orig_shape[1]
    x1 = np.clip(x1, 0, orig_w - 1)
    x2 = np.clip(x2, 0, orig_w - 1)
    y1 = np.clip(y1, 0, orig_h - 1)
    y2 = np.clip(y2, 0, orig_h - 1)

    detections = []
    for i in range(boxes_xyxy.shape[0]):
        cid = int(class_ids[i])
        name = class_names[cid]
        severity = severity_map.get(name, Severity.COSMETIC)
        detections.append(
            Detection(
                class_id=cid,
                class_name=name,
                confidence=float(confidences[i]),
                bbox=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
                severity=severity,
            )
        )
    return tuple(detections)


def _class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    """NMS applied independently per class so boxes of different classes
    never suppress each other."""
    if boxes.shape[0] == 0:
        return []

    keep: list[int] = []
    for cid in np.unique(class_ids):
        idxs = np.where(class_ids == cid)[0]
        survivors = _nms(boxes[idxs], scores[idxs], iou_threshold)
        keep.extend(int(idxs[j]) for j in survivors)

    keep.sort(key=lambda i: -scores[i])
    return keep


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy single-class NMS in pure numpy (no cv2/torch NMS dependency).

    `boxes` is `(N, 4)` xyxy, `scores` is `(N,)`. Returns indices into
    `boxes`/`scores` to keep, highest score first.
    """
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)

        order = rest[iou <= iou_threshold]

    return np.array(keep, dtype=np.int64)


def _static_input_size(shape: Optional[Sequence[object]]) -> Optional[int]:
    """Return the model's static square spatial input size (e.g. 640 for an
    input shaped `[1, 3, 640, 640]`), or None if the shape is missing,
    non-4D, dynamic (a symbolic dim shows up as a non-int), or non-square --
    any of which means the caller should fall back to config."""
    if shape is None or len(shape) != 4:
        return None
    h, w = shape[2], shape[3]
    if isinstance(h, int) and isinstance(w, int) and h > 0 and h == w:
        return h
    return None


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------


class OnnxYoloDetector(Detector):
    """`Detector` implementation backed by an ONNX model -- either a
    legacy YOLOv8 export or an end-to-end YOLO26 export (see
    `DetectorConfig.layout`)."""

    def __init__(self, cfg: DetectorConfig, class_names: Optional[Sequence[str]] = None) -> None:
        self._cfg = cfg
        self._class_names: tuple[str, ...] = tuple(class_names) if class_names is not None else DEFAULT_CLASS_NAMES

        model_path = Path(cfg.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"ONNX model not found at '{model_path}' -- has it been trained/exported yet?"
            )

        self._session: Optional[ort.InferenceSession] = ort.InferenceSession(
            str(model_path), providers=list(cfg.providers)
        )
        self._input_name = self._session.get_inputs()[0].name

        static_size = _static_input_size(self._session.get_inputs()[0].shape)
        self._input_size = static_size if static_size is not None else cfg.input_size

        # Resolved once here and reused for every `infer()` call -- never
        # re-detected per-frame.
        declared_output_shape = self._session.get_outputs()[0].shape
        self._layout = _resolve_layout(cfg.layout, declared_output_shape, len(self._class_names))
        logger.info(
            "OnnxYoloDetector: using '%s' output layout (%s)",
            self._layout,
            f"configured explicitly as '{cfg.layout}'"
            if cfg.layout != "auto"
            else f"auto-detected from declared output shape {declared_output_shape}",
        )

    def infer(self, frame: Frame) -> DetectionResult:
        if self._session is None:
            raise RuntimeError("OnnxYoloDetector is closed")

        start = time.perf_counter()
        blob, scale, pad = preprocess(frame.image, self._input_size)
        raw = self._session.run(None, {self._input_name: blob})[0]

        if self._layout == "end2end":
            detections = postprocess_end2end(
                raw,
                scale,
                pad,
                frame.image.shape[:2],
                self._class_names,
                self._cfg.class_thresholds,
                self._cfg.default_threshold,
                self._cfg.severity,
            )
        else:
            detections = postprocess(
                raw,
                scale,
                pad,
                frame.image.shape[:2],
                self._class_names,
                self._cfg.class_thresholds,
                self._cfg.default_threshold,
                self._cfg.severity,
                self._cfg.nms_iou,
            )
        inference_ms = (time.perf_counter() - start) * 1000.0
        return DetectionResult(detections=detections, inference_ms=inference_ms)

    def close(self) -> None:
        self._session = None
