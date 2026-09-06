"""ONNX detector supporting both YOLOv8 and YOLO26 output contracts.

`DetectorConfig.layout` selects the contract ("auto" picks via
`detect_layout`): `postprocess` (YOLOv8) applies class-aware NMS to raw,
NMS-pending predictions; `postprocess_end2end` (YOLO26) applies **no NMS**
since that model is already NMS-free/end-to-end -- redoing it would silently
drop valid adjacent detections. All pre/post-processing is pure functions on
plain numpy arrays, unit-testable with no model file on disk.
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


# tuple[np.ndarray, float, tuple[float, float]] letterbox(np.ndarray image, Union[int, tuple[int, int]] new_shape, tuple[int, int, int] color)
# Inputs: np.ndarray image - source BGR frame (HxWx3, uint8) to letterbox
#         Union[int, tuple[int, int]] new_shape - target size; an int means a square
#                 new_shape x new_shape, else (target_h, target_w)
#         tuple[int, int, int] color - default `_LETTERBOX_COLOR` (114, 114, 114). BGR fill
#                 color for the padding
# Outputs: tuple[np.ndarray, float, tuple[float, float]] - (padded_image, scale, (pad_w,
#          pad_h)): `scale` is the resize factor applied to the original image, and
#          `(pad_w, pad_h)` is the (possibly fractional) left/top padding added -- together
#          exactly what's needed to invert the transform via
#          `orig = (letterboxed - pad) / scale`
# Description: Resizes `image` preserving aspect ratio to fit within `new_shape`, then pads
#              with `color` to reach `new_shape` exactly, splitting the (possibly odd) total
#              padding across both sides the same way Ultralytics' own letterbox does.
# Side Effects: None (pure function of its inputs).
def letterbox(
    image: np.ndarray,
    new_shape: Union[int, tuple[int, int]],
    color: tuple[int, int, int] = _LETTERBOX_COLOR,
) -> tuple[np.ndarray, float, tuple[float, float]]:
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


# tuple[np.ndarray, float, tuple[float, float]] preprocess(np.ndarray image, int input_size)
# Inputs: np.ndarray image - source BGR frame (HxWx3, uint8) to preprocess
#         int input_size - target square model input size
# Outputs: tuple[np.ndarray, float, tuple[float, float]] - (blob, scale, pad): `blob` is a
#          (1, 3, H, W) float32 array in [0, 1], RGB, channel-first; `scale`/`pad` are passed
#          through unchanged from `letterbox` for later use by `postprocess`/
#          `postprocess_end2end`
# Description: Letterboxes `image` to a square `input_size` via `letterbox`, then converts
#              BGR->RGB, HWC->CHW, and scales to [0, 1] to build the model-ready blob.
# Side Effects: None (pure function of its inputs).
def preprocess(image: np.ndarray, input_size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    padded, scale, pad = letterbox(image, input_size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1)
    blob = np.ascontiguousarray(chw, dtype=np.float32) / 255.0
    blob = blob[np.newaxis, ...]
    return blob, scale, pad


# np.ndarray _to_rows(np.ndarray raw, int expected_cols)
# Inputs: np.ndarray raw - raw YOLOv8 ONNX output, either `(1, 4+nc, N)`, `(1, N, 4+nc)`, or
#                 either of those without the batch dimension
#         int expected_cols - `4 + num_classes`, used to identify which axis is the
#                 per-anchor feature axis
# Outputs: np.ndarray - a 2D `(num_anchors, 4 + num_classes)` array
# Description: Normalizes a raw YOLOv8 output into the canonical 2D row-per-anchor layout,
#              handling both the native transposed-relative-to-YOLOv5 layout and an
#              already-transposed one. Raises `ValueError` if the shape doesn't match
#              `expected_cols` on either axis, or has an unexpected batch size/dimensionality.
# Side Effects: None (pure function of its inputs).
def _to_rows(raw: np.ndarray, expected_cols: int) -> np.ndarray:
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


# tuple[Detection, ...] postprocess(np.ndarray raw, float scale, tuple[float, float] pad, tuple[int, int] orig_shape, Sequence[str] class_names, Mapping[str, float] class_thresholds, float default_threshold, Mapping[str, Severity] severity_map, float nms_iou)
# Inputs: np.ndarray raw - raw YOLOv8 ONNX output tensor
#         float scale - letterbox scale factor from `preprocess`/`letterbox`
#         tuple[float, float] pad - (pad_w, pad_h) letterbox padding in pixels
#         tuple[int, int] orig_shape - (height, width) of the original frame, used to clip
#                 boxes back into bounds
#         Sequence[str] class_names - ordered class names matching the model's output columns
#         Mapping[str, float] class_thresholds - per-class-name confidence threshold overrides
#         float default_threshold - threshold used for any class name absent from
#                 `class_thresholds`
#         Mapping[str, Severity] severity_map - per-class-name Severity; a name absent from it
#                 defaults to Severity.COSMETIC so an unrecognized class can never drive the
#                 stop decision
#         float nms_iou - IoU threshold above which two same-class boxes are considered
#                 duplicates during NMS
# Outputs: tuple[Detection, ...] - final detections in ORIGINAL frame coordinates, after
#          per-class thresholding and class-aware NMS
# Description: Turns a raw YOLOv8 ONNX output into final `Detection`s: normalizes the tensor
#              via `_to_rows`, takes the argmax class per anchor, applies per-class confidence
#              thresholds, inverts the letterbox transform back to original-frame pixel
#              coordinates and clips to frame bounds, then runs class-aware NMS (`_class_aware_
#              nms`) so different classes never suppress each other. Unlike
#              `postprocess_end2end` (the YOLO26 path), this path DOES apply NMS -- the raw
#              YOLOv8 predictions are NMS-pending, whereas YOLO26's end-to-end output already
#              had NMS applied internally and must not be NMS'd again.
# Side Effects: None (pure function of its inputs).
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


# str detect_layout(np.ndarray raw, int num_classes)
# Inputs: np.ndarray raw - a raw (or zero-filled probe) model output tensor whose shape is
#                 used to infer the layout
#         int num_classes - number of trained classes, used to compute `4 + num_classes`
# Outputs: str - `"yolov8"` or `"end2end"`
# Description: Infers which ONNX output contract `raw` follows by shape heuristic: squeezes
#              the batch dim, then returns `"yolov8"` if any remaining axis equals
#              `4 + num_classes`, else `"end2end"` if the last axis is exactly 6, else raises
#              `ValueError`. When `num_classes == 2`, `4 + num_classes == 6` is ambiguous; the
#              `"yolov8"` check runs first so this case always resolves to `"yolov8"` (see the
#              function's own docstring for the documented workaround).
# Side Effects: None (pure function of its inputs).
def detect_layout(raw: np.ndarray, num_classes: int) -> str:
    """When num_classes == 2, 4+num_classes == 6 is ambiguous with the
    end2end row width; the yolov8 check runs first so this always resolves
    to "yolov8" -- a 2-class end2end model must set detector.layout
    explicitly rather than "auto"."""
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


# tuple[int, ...] _probe_layout_shape(Optional[Sequence[object]] declared_shape)
# Inputs: Optional[Sequence[object]] declared_shape - the ONNX-declared output shape (e.g.
#                 from `session.get_outputs()[0].shape`), whose batch/anchor dims may be
#                 dynamic and show up as strings, None, or negative sentinels
# Outputs: tuple[int, ...] - a concrete shape usable to build a zero-filled probe array for
#          `detect_layout`; `()` if `declared_shape` is falsy
# Description: Converts a possibly-symbolic ONNX-declared shape into a concrete one: replaces
#              any non-batch dynamic dim with 0 (which can never coincidentally match
#              `4 + num_classes` or 6 for a real model, so it never causes a false layout
#              match) and forces the batch dim (axis 0) to 1, since every real inference call
#              here uses a batch of exactly 1.
# Side Effects: None (pure function of its input).
def _probe_layout_shape(declared_shape: Optional[Sequence[object]]) -> tuple[int, ...]:
    if not declared_shape:
        return ()
    dims = [d if isinstance(d, int) and d > 0 else 0 for d in declared_shape]
    dims[0] = 1
    return tuple(dims)


_VALID_LAYOUTS = {"yolov8", "end2end"}


# str _resolve_layout(str configured, Optional[Sequence[object]] declared_output_shape, int num_classes)
# Inputs: str configured - `DetectorConfig.layout` value: `"auto"`, `"yolov8"`, or `"end2end"`
#         Optional[Sequence[object]] declared_output_shape - the model's declared ONNX output
#                 shape (from `session.get_outputs()[0].shape`), used only when `configured`
#                 is `"auto"`
#         int num_classes - number of trained classes, forwarded to `detect_layout`
# Outputs: str - the concrete layout to use for every inference call: `"yolov8"` or `"end2end"`
# Description: Resolves the layout once (meant to be called a single time at detector
#              construction and reused for every `infer()` call, never re-detected per-frame):
#              returns `configured` verbatim if it's explicit, otherwise runs `detect_layout`'s
#              heuristic against a zero-filled probe array built from `declared_output_shape`
#              via `_probe_layout_shape`. Raises `ValueError` if `configured` is neither
#              `"auto"` nor a valid layout name.
# Side Effects: None (pure function of its inputs).
def _resolve_layout(configured: str, declared_output_shape: Optional[Sequence[object]], num_classes: int) -> str:
    if configured != "auto":
        if configured not in _VALID_LAYOUTS:
            raise ValueError(f"detector.layout must be one of {sorted(_VALID_LAYOUTS)}, got {configured!r}")
        return configured
    probe = np.zeros(_probe_layout_shape(declared_output_shape), dtype=np.float32)
    return detect_layout(probe, num_classes)


# tuple[Detection, ...] postprocess_end2end(np.ndarray raw, float scale, tuple[float, float] pad, tuple[int, int] orig_shape, Sequence[str] class_names, Mapping[str, float] class_thresholds, float default_threshold, Mapping[str, Severity] severity_map)
# Inputs: np.ndarray raw - raw YOLO26 end-to-end ONNX output, `(1, max_det, 6)` or
#                 `(max_det, 6)`, each row already decoded as [x1, y1, x2, y2, confidence,
#                 class_id] in letterboxed input-pixel space
#         float scale - letterbox scale factor from `preprocess`/`letterbox`
#         tuple[float, float] pad - (pad_w, pad_h) letterbox padding in pixels
#         tuple[int, int] orig_shape - (height, width) of the original frame, used to clip
#                 boxes back into bounds
#         Sequence[str] class_names - ordered class names matching the model's class ids
#         Mapping[str, float] class_thresholds - per-class-name confidence threshold overrides
#         float default_threshold - threshold used for any class name absent from
#                 `class_thresholds`
#         Mapping[str, Severity] severity_map - per-class-name Severity; a name absent from it
#                 defaults to Severity.COSMETIC so an unrecognized class can never drive the
#                 stop decision
# Outputs: tuple[Detection, ...] - detections above their per-class thresholds, in ORIGINAL
#          frame coordinates
# Description: Decodes YOLO26's already-decoded end-to-end output: drops padded rows
#              (confidence <= 0, or a class id outside `range(len(class_names))`, since a
#              padded row can carry a garbage class index), applies per-class confidence
#              thresholds, then inverts the letterbox transform back to original-frame pixel
#              coordinates and clips to frame bounds (decoded boxes can fall slightly outside
#              [0, input_size]). Deliberately applies **no NMS** -- unlike `postprocess`'s
#              YOLOv8 path, the model here is NMS-free/end-to-end and already deduplicated its
#              own output, so two heavily-overlapping same-class boxes are both genuine and
#              both kept; redoing NMS here would silently drop valid adjacent detections.
# Side Effects: None (pure function of its inputs).
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
    """Deliberately applies no NMS: the model is NMS-free/end-to-end and
    already deduplicated its own output, so overlapping same-class boxes
    here are both genuine and both kept -- unlike `postprocess`'s YOLOv8
    path."""
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


# list[int] _class_aware_nms(np.ndarray boxes, np.ndarray scores, np.ndarray class_ids, float iou_threshold)
# Inputs: np.ndarray boxes - (N, 4) xyxy boxes
#         np.ndarray scores - (N,) confidence scores, one per box
#         np.ndarray class_ids - (N,) integer class id per box
#         float iou_threshold - IoU above which two same-class boxes are considered duplicates
# Outputs: list[int] - indices into `boxes`/`scores` to keep, sorted by descending score
# Description: Runs `_nms` independently within each distinct class id (so boxes of different
#              classes never suppress each other), then merges and re-sorts the per-class
#              survivors by descending score. This is the NMS step used by `postprocess` (the
#              YOLOv8 path) only -- `postprocess_end2end` (YOLO26) never calls this, since that
#              model's output is already NMS'd.
# Side Effects: None (pure function of its inputs).
def _class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    if boxes.shape[0] == 0:
        return []

    keep: list[int] = []
    for cid in np.unique(class_ids):
        idxs = np.where(class_ids == cid)[0]
        survivors = _nms(boxes[idxs], scores[idxs], iou_threshold)
        keep.extend(int(idxs[j]) for j in survivors)

    keep.sort(key=lambda i: -scores[i])
    return keep


# np.ndarray _nms(np.ndarray boxes, np.ndarray scores, float iou_threshold)
# Inputs: np.ndarray boxes - (N, 4) xyxy boxes, all of a single class
#         np.ndarray scores - (N,) confidence scores, one per box
#         float iou_threshold - IoU above which a lower-scoring box is suppressed by a
#                 higher-scoring one
# Outputs: np.ndarray - indices into `boxes`/`scores` to keep, highest score first
# Description: Greedy single-class non-maximum suppression implemented in pure numpy (no
#              cv2/torch NMS dependency): repeatedly takes the highest-remaining-score box,
#              keeps it, and removes any remaining box whose IoU with it exceeds
#              `iou_threshold`.
# Side Effects: None (pure function of its inputs).
def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
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


# Optional[int] _static_input_size(Optional[Sequence[object]] shape)
# Inputs: Optional[Sequence[object]] shape - the ONNX-declared input shape (e.g. from
#                 `session.get_inputs()[0].shape`), possibly containing symbolic/dynamic dims
# Outputs: Optional[int] - the static square spatial size (e.g. 640), or None if the shape is
#          missing, not 4D, dynamic, or non-square
# Description: Extracts a usable static input size from a declared ONNX input shape so the
#              caller can fall back to `cfg.input_size` when the model's shape doesn't pin one
#              down.
# Side Effects: None (pure function of its input).
def _static_input_size(shape: Optional[Sequence[object]]) -> Optional[int]:
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

    # None __init__(DetectorConfig cfg, Optional[Sequence[str]] class_names)
    # Inputs: DetectorConfig cfg - detector configuration: model path, providers, input size,
    #                 layout, thresholds, and severity map
    #         Optional[Sequence[str]] class_names - default None. Ordered class names matching
    #                 the model's output index order; falls back to `DEFAULT_CLASS_NAMES` when
    #                 not given
    # Outputs: None
    # Description: Loads the ONNX detection model into an onnxruntime InferenceSession,
    #              determines the model's input size (its static declared shape if available,
    #              else `cfg.input_size`), and resolves the output layout once via
    #              `_resolve_layout` (using `cfg.layout` and the model's declared output
    #              shape) so it never needs to be re-detected per-frame.
    # Side Effects: Reads `cfg.model_path` from disk; raises `FileNotFoundError` if it doesn't
    #               exist. Constructs an `onnxruntime.InferenceSession` (allocates model
    #               resources). Logs an info message describing the resolved layout. Mutates
    #               the new instance's state (`_cfg`, `_class_names`, `_session`,
    #               `_input_name`, `_input_size`, `_layout`).
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

    # DetectionResult infer(Frame frame)
    # Inputs: Frame frame - the captured camera frame to run detection on
    # Outputs: DetectionResult - detections from whichever postprocessing path matches the
    #          resolved layout (see `postprocess`/`postprocess_end2end`), plus the measured
    #          inference time in milliseconds
    # Description: Runs the full detect pipeline for one frame: `preprocess` builds the
    #              letterboxed model input blob, the ONNX session runs inference, and then
    #              either `postprocess_end2end` (YOLO26, no NMS) or `postprocess` (YOLOv8,
    #              with class-aware NMS) is dispatched based on `self._layout`, resolved once
    #              at construction time. Only a CATASTROPHIC-severity detection here would ever
    #              raise `p_failure`.
    # Side Effects: Runs ONNX model inference (CPU/GPU work via `self._session.run`); reads
    #               `time.perf_counter()` to measure elapsed time. Raises `RuntimeError` if the
    #               detector has already been `close()`d.
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
