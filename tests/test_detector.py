"""Tests for the detector layer: pure pre/post-processing functions,
OnnxYoloDetector's constructor-time failure mode, and MockDetector.

No model file and no network access is required -- postprocess/letterbox are
exercised with synthetic numpy arrays built by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from argus.config import DetectorConfig
from argus.detectors.base import Detector
from argus.detectors.mock import MockDetector
from argus.detectors.onnx_yolo import (
    DEFAULT_CLASS_NAMES,
    OnnxYoloDetector,
    _resolve_layout,
    _static_input_size,
    detect_layout,
    letterbox,
    postprocess,
    postprocess_end2end,
    preprocess,
)
from argus.types import Detection, DetectionResult, Frame, Severity

CLASS_NAMES = DEFAULT_CLASS_NAMES  # ("error extrusion", "spaghetti", "stringing", "warping", "zits")
NUM_CLASSES = len(CLASS_NAMES)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _row(cx: float, cy: float, w: float, h: float, class_id: int, confidence: float) -> np.ndarray:
    """Build one YOLOv8-style output row: [cx, cy, w, h, score_0, ..., score_{nc-1}]."""
    row = np.zeros(4 + NUM_CLASSES, dtype=np.float32)
    row[0:4] = (cx, cy, w, h)
    row[4 + class_id] = confidence
    return row


def _raw_from_rows(rows: list[np.ndarray], orientation: str) -> np.ndarray:
    """Stack rows into a raw model output tensor in the requested orientation:
    'N,C' -> (1, num_anchors, 4+nc); 'C,N' -> (1, 4+nc, num_anchors)."""
    arr = np.stack(rows, axis=0)  # (N, 4+nc)
    if orientation == "N,C":
        return arr[np.newaxis, ...]
    if orientation == "C,N":
        return arr.T[np.newaxis, ...]
    raise ValueError(orientation)


def _make_frame(shape: tuple[int, int, int] = (10, 10, 3)) -> Frame:
    return Frame(image=np.zeros(shape, dtype=np.uint8), timestamp=0.0, seq=0)


_DEFAULT_KW = dict(
    scale=1.0,
    pad=(0.0, 0.0),
    orig_shape=(640, 640),
    class_names=CLASS_NAMES,
    class_thresholds={},
    default_threshold=0.5,
    severity_map={},
    nms_iou=0.45,
)


def _postprocess(raw: np.ndarray, **overrides) -> tuple[Detection, ...]:
    kwargs = dict(_DEFAULT_KW)
    kwargs.update(overrides)
    return postprocess(raw, **kwargs)


def _e2e_row(x1: float, y1: float, x2: float, y2: float, class_id: float, confidence: float) -> np.ndarray:
    """Build one YOLO26-style end-to-end output row:
    [x1, y1, x2, y2, confidence, class_id] -- already-decoded box coordinates
    in letterboxed input-pixel space, class_id passed as a float exactly as
    the real ONNX output carries it."""
    return np.array([x1, y1, x2, y2, confidence, class_id], dtype=np.float32)


def _e2e_raw(rows: list[np.ndarray], max_det: int = 300) -> np.ndarray:
    """Stack end-to-end rows into a raw `(1, max_det, 6)` tensor, padding any
    unused slots with all-zero rows (confidence 0) -- matching what a real
    YOLO26 export emits for unused detection slots."""
    if len(rows) > max_det:
        raise ValueError("too many rows for max_det")
    stacked = np.zeros((max_det, 6), dtype=np.float32)
    for i, row in enumerate(rows):
        stacked[i] = row
    return stacked[np.newaxis, ...]


_DEFAULT_E2E_KW = dict(
    scale=1.0,
    pad=(0.0, 0.0),
    orig_shape=(640, 640),
    class_names=CLASS_NAMES,
    class_thresholds={},
    default_threshold=0.5,
    severity_map={},
)


def _postprocess_e2e(raw: np.ndarray, **overrides) -> tuple[Detection, ...]:
    kwargs = dict(_DEFAULT_E2E_KW)
    kwargs.update(overrides)
    return postprocess_end2end(raw, **kwargs)


# --------------------------------------------------------------------------
# letterbox geometry
# --------------------------------------------------------------------------


def test_letterbox_landscape_pads_vertically():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)  # h, w
    padded, scale, pad = letterbox(image, 640)

    assert padded.shape == (640, 640, 3)
    assert padded.dtype == np.uint8
    assert scale == pytest.approx(0.5)
    assert pad == pytest.approx((0.0, 140.0))

    # No horizontal padding; vertical padding rows should be the grey fill.
    assert tuple(int(c) for c in padded[0, 0]) == (114, 114, 114)
    assert tuple(int(c) for c in padded[-1, 0]) == (114, 114, 114)


def test_letterbox_portrait_pads_horizontally():
    image = np.zeros((1280, 720, 3), dtype=np.uint8)  # h, w
    padded, scale, pad = letterbox(image, 640)

    assert padded.shape == (640, 640, 3)
    assert scale == pytest.approx(0.5)
    assert pad == pytest.approx((140.0, 0.0))

    assert tuple(int(c) for c in padded[0, 0]) == (114, 114, 114)
    assert tuple(int(c) for c in padded[0, -1]) == (114, 114, 114)


def test_preprocess_returns_expected_blob_shape_and_range():
    image = np.random.default_rng(0).integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    blob, scale, pad = preprocess(image, 640)

    assert blob.shape == (1, 3, 640, 640)
    assert blob.dtype == np.float32
    assert blob.min() >= 0.0 and blob.max() <= 1.0
    assert scale == pytest.approx(0.5)
    assert pad == pytest.approx((0.0, 140.0))


# --------------------------------------------------------------------------
# Coordinate round-trip (the classic silent-failure case)
# --------------------------------------------------------------------------


def test_postprocess_roundtrip_recovers_original_box_within_1px():
    orig_shape = (720, 1280)  # h, w
    dummy = np.zeros((*orig_shape, 3), dtype=np.uint8)
    _, scale, pad = letterbox(dummy, 640)
    pad_w, pad_h = pad

    orig_box = (300.0, 150.0, 500.0, 400.0)  # x1, y1, x2, y2 in original coords

    # Forward-transform the known box into letterboxed input-pixel space,
    # exactly as the real image content would have been placed.
    lb_x1 = orig_box[0] * scale + pad_w
    lb_y1 = orig_box[1] * scale + pad_h
    lb_x2 = orig_box[2] * scale + pad_w
    lb_y2 = orig_box[3] * scale + pad_h
    cx, cy = (lb_x1 + lb_x2) / 2, (lb_y1 + lb_y2) / 2
    w, h = lb_x2 - lb_x1, lb_y2 - lb_y1

    class_id = CLASS_NAMES.index("spaghetti")
    raw = _raw_from_rows([_row(cx, cy, w, h, class_id, 0.9)], "N,C")

    dets = _postprocess(
        raw,
        scale=scale,
        pad=pad,
        orig_shape=orig_shape,
        severity_map={"spaghetti": Severity.CATASTROPHIC},
    )

    assert len(dets) == 1
    det = dets[0]
    assert det.class_name == "spaghetti"
    assert det.confidence == pytest.approx(0.9)
    for got, want in zip(det.bbox, orig_box):
        assert got == pytest.approx(want, abs=1.0)


# --------------------------------------------------------------------------
# Output-tensor orientation handling
# --------------------------------------------------------------------------


def test_postprocess_handles_both_raw_output_orientations():
    num_anchors = 8400
    rows = np.zeros((num_anchors, 4 + NUM_CLASSES), dtype=np.float32)
    real_idx = 4242
    rows[real_idx] = _row(320, 320, 100, 100, CLASS_NAMES.index("spaghetti"), 0.8)

    raw_nc = rows[np.newaxis, ...]  # (1, 8400, 9)
    raw_cn = rows.T[np.newaxis, ...]  # (1, 9, 8400)
    assert raw_nc.shape == (1, num_anchors, 4 + NUM_CLASSES)
    assert raw_cn.shape == (1, 4 + NUM_CLASSES, num_anchors)

    kw = dict(severity_map={"spaghetti": Severity.CATASTROPHIC})
    dets_nc = _postprocess(raw_nc, **kw)
    dets_cn = _postprocess(raw_cn, **kw)

    assert len(dets_nc) == 1 and len(dets_cn) == 1
    assert dets_nc[0].class_name == dets_cn[0].class_name == "spaghetti"
    assert dets_nc[0].confidence == pytest.approx(dets_cn[0].confidence, abs=1e-5)
    assert dets_nc[0].bbox == pytest.approx(dets_cn[0].bbox, abs=1e-4)


# --------------------------------------------------------------------------
# Per-class confidence thresholds
# --------------------------------------------------------------------------


def test_per_class_thresholds_applied_independently():
    rows = [
        _row(100, 100, 40, 40, CLASS_NAMES.index("spaghetti"), 0.55),
        _row(400, 400, 40, 40, CLASS_NAMES.index("warping"), 0.55),
    ]
    raw = _raw_from_rows(rows, "N,C")

    dets = _postprocess(
        raw,
        class_thresholds={"spaghetti": 0.50, "warping": 0.60},
        severity_map={"spaghetti": Severity.CATASTROPHIC, "warping": Severity.CATASTROPHIC},
    )

    names = {d.class_name for d in dets}
    assert names == {"spaghetti"}  # spaghetti (0.55 >= 0.50) passes, warping (0.55 < 0.60) rejected


# --------------------------------------------------------------------------
# Severity mapping / p_failure false-positive control
# --------------------------------------------------------------------------


def test_severity_mapping_catastrophic_vs_cosmetic():
    rows = [
        _row(100, 100, 40, 40, CLASS_NAMES.index("spaghetti"), 0.60),
        _row(400, 400, 40, 40, CLASS_NAMES.index("zits"), 0.99),
    ]
    raw = _raw_from_rows(rows, "N,C")
    severity_map = {"spaghetti": Severity.CATASTROPHIC, "zits": Severity.COSMETIC}

    dets = _postprocess(raw, severity_map=severity_map)

    assert len(dets) == 2
    by_name = {d.class_name: d for d in dets}
    assert by_name["spaghetti"].severity is Severity.CATASTROPHIC
    assert by_name["zits"].severity is Severity.COSMETIC


def test_p_failure_ignores_cosmetic_detections_even_if_higher_confidence():
    rows = [
        _row(100, 100, 40, 40, CLASS_NAMES.index("spaghetti"), 0.60),
        _row(400, 400, 40, 40, CLASS_NAMES.index("zits"), 0.99),
    ]
    raw = _raw_from_rows(rows, "N,C")
    severity_map = {"spaghetti": Severity.CATASTROPHIC, "zits": Severity.COSMETIC}

    dets = _postprocess(raw, severity_map=severity_map)
    result = DetectionResult(detections=dets, inference_ms=0.0)

    # p_failure must reflect only the catastrophic detection (0.60), never
    # the higher-confidence cosmetic one (0.99).
    assert result.p_failure == pytest.approx(0.60)


def test_p_failure_is_zero_for_cosmetic_only_frame():
    rows = [_row(100, 100, 40, 40, CLASS_NAMES.index("zits"), 0.95)]
    raw = _raw_from_rows(rows, "N,C")

    dets = _postprocess(raw, severity_map={"zits": Severity.COSMETIC})
    assert len(dets) == 1  # the detection itself is real...

    result = DetectionResult(detections=dets, inference_ms=0.0)
    assert result.p_failure == 0.0  # ...but it must not contribute to p_failure


def test_unknown_class_name_defaults_to_cosmetic():
    rows = [_row(100, 100, 40, 40, CLASS_NAMES.index("stringing"), 0.9)]
    raw = _raw_from_rows(rows, "N,C")

    # severity_map deliberately has no entry for "stringing".
    dets = _postprocess(raw, severity_map={})

    assert len(dets) == 1
    assert dets[0].severity is Severity.COSMETIC

    result = DetectionResult(detections=dets, inference_ms=0.0)
    assert result.p_failure == 0.0


# --------------------------------------------------------------------------
# Class-aware NMS
# --------------------------------------------------------------------------


def test_nms_keeps_overlapping_boxes_of_different_classes():
    rows = [
        _row(320, 320, 100, 100, CLASS_NAMES.index("spaghetti"), 0.9),
        _row(320, 320, 100, 100, CLASS_NAMES.index("warping"), 0.8),  # identical box, different class
    ]
    raw = _raw_from_rows(rows, "N,C")

    dets = _postprocess(
        raw,
        severity_map={"spaghetti": Severity.CATASTROPHIC, "warping": Severity.CATASTROPHIC},
    )

    assert len(dets) == 2
    assert {d.class_name for d in dets} == {"spaghetti", "warping"}


def test_nms_collapses_overlapping_boxes_of_same_class():
    class_id = CLASS_NAMES.index("spaghetti")
    rows = [
        _row(320, 320, 100, 100, class_id, 0.9),
        _row(325, 325, 100, 100, class_id, 0.7),  # heavily overlapping, same class
    ]
    raw = _raw_from_rows(rows, "N,C")

    dets = _postprocess(raw, severity_map={"spaghetti": Severity.CATASTROPHIC})

    assert len(dets) == 1
    assert dets[0].confidence == pytest.approx(0.9)  # higher-confidence survivor


# --------------------------------------------------------------------------
# Empty / all-below-threshold output
# --------------------------------------------------------------------------


def test_empty_raw_output_yields_no_detections():
    raw = np.zeros((1, 0, 4 + NUM_CLASSES), dtype=np.float32)

    dets = _postprocess(raw)
    assert dets == ()

    result = DetectionResult(detections=dets, inference_ms=1.0)
    assert result.p_failure == 0.0


def test_all_below_threshold_yields_no_detections():
    rows = [_row(320, 320, 100, 100, CLASS_NAMES.index("spaghetti"), 0.2)]  # below default 0.5
    raw = _raw_from_rows(rows, "N,C")

    dets = _postprocess(raw)
    assert dets == ()


# --------------------------------------------------------------------------
# postprocess_end2end (YOLO26 end-to-end layout)
# --------------------------------------------------------------------------


def test_postprocess_end2end_roundtrip_recovers_original_box_within_1px():
    orig_shape = (720, 1280)  # h, w
    dummy = np.zeros((*orig_shape, 3), dtype=np.uint8)
    _, scale, pad = letterbox(dummy, 640)
    pad_w, pad_h = pad

    orig_box = (300.0, 150.0, 500.0, 400.0)  # x1, y1, x2, y2 in original coords

    # Forward-transform the known box into letterboxed input-pixel space,
    # exactly as the model's own decoded output would express it.
    lb_x1 = orig_box[0] * scale + pad_w
    lb_y1 = orig_box[1] * scale + pad_h
    lb_x2 = orig_box[2] * scale + pad_w
    lb_y2 = orig_box[3] * scale + pad_h

    class_id = CLASS_NAMES.index("spaghetti")
    raw = _e2e_raw([_e2e_row(lb_x1, lb_y1, lb_x2, lb_y2, class_id, 0.9)])

    dets = _postprocess_e2e(
        raw,
        scale=scale,
        pad=pad,
        orig_shape=orig_shape,
        severity_map={"spaghetti": Severity.CATASTROPHIC},
    )

    assert len(dets) == 1
    det = dets[0]
    assert det.class_name == "spaghetti"
    assert det.confidence == pytest.approx(0.9)
    for got, want in zip(det.bbox, orig_box):
        assert got == pytest.approx(want, abs=1.0)


def test_postprocess_end2end_drops_padded_zero_confidence_rows():
    class_id = CLASS_NAMES.index("spaghetti")
    real_row = _e2e_row(100, 100, 200, 200, class_id, 0.8)
    raw = _e2e_raw([real_row])  # remaining 299 rows are all-zero (confidence 0) padding

    dets = _postprocess_e2e(raw, severity_map={"spaghetti": Severity.CATASTROPHIC})

    assert len(dets) == 1
    assert dets[0].class_name == "spaghetti"


def test_postprocess_end2end_drops_out_of_range_class_id_without_crashing():
    # A real sampled padded row: [-1.9611, 0.82404, 511.89, 509.89, 0.13599,
    # 16] -- nonzero confidence but class 16 with only 5 known classes
    # (0..4). Must be dropped, not crash on, and never mapped to a wrong
    # class name.
    garbage_row = _e2e_row(-1.9611, 0.82404, 511.89, 509.89, 16, 0.13599)
    real_row = _e2e_row(100, 100, 200, 200, CLASS_NAMES.index("spaghetti"), 0.8)
    raw = _e2e_raw([real_row, garbage_row])

    dets = _postprocess_e2e(raw, severity_map={"spaghetti": Severity.CATASTROPHIC})

    assert len(dets) == 1
    assert dets[0].class_name == "spaghetti"


def test_postprocess_end2end_per_class_thresholds_applied_independently():
    rows = [
        _e2e_row(100, 100, 140, 140, CLASS_NAMES.index("spaghetti"), 0.55),
        _e2e_row(400, 400, 440, 440, CLASS_NAMES.index("warping"), 0.55),
    ]
    raw = _e2e_raw(rows)

    dets = _postprocess_e2e(
        raw,
        class_thresholds={"spaghetti": 0.50, "warping": 0.60},
        severity_map={"spaghetti": Severity.CATASTROPHIC, "warping": Severity.CATASTROPHIC},
    )

    names = {d.class_name for d in dets}
    assert names == {"spaghetti"}  # spaghetti (0.55 >= 0.50) passes, warping (0.55 < 0.60) rejected


def test_postprocess_end2end_severity_mapping_and_p_failure_ignores_cosmetic():
    rows = [
        _e2e_row(100, 100, 140, 140, CLASS_NAMES.index("spaghetti"), 0.60),
        _e2e_row(400, 400, 440, 440, CLASS_NAMES.index("zits"), 0.99),
    ]
    raw = _e2e_raw(rows)
    severity_map = {"spaghetti": Severity.CATASTROPHIC, "zits": Severity.COSMETIC}

    dets = _postprocess_e2e(raw, severity_map=severity_map)

    assert len(dets) == 2
    by_name = {d.class_name: d for d in dets}
    assert by_name["spaghetti"].severity is Severity.CATASTROPHIC
    assert by_name["zits"].severity is Severity.COSMETIC

    result = DetectionResult(detections=dets, inference_ms=0.0)
    # p_failure must reflect only the catastrophic detection (0.60), never
    # the higher-confidence cosmetic one (0.99).
    assert result.p_failure == pytest.approx(0.60)


def test_postprocess_end2end_p_failure_is_zero_for_cosmetic_only_frame():
    row = _e2e_row(100, 100, 140, 140, CLASS_NAMES.index("zits"), 0.95)
    raw = _e2e_raw([row])

    dets = _postprocess_e2e(raw, severity_map={"zits": Severity.COSMETIC})
    assert len(dets) == 1  # the detection itself is real...

    result = DetectionResult(detections=dets, inference_ms=0.0)
    assert result.p_failure == 0.0  # ...but it must not contribute to p_failure


def test_postprocess_end2end_applies_no_nms_both_overlapping_boxes_survive():
    # The key behavioural difference from postprocess (YOLOv8): the model
    # has already run its own NMS, so postprocess_end2end must never
    # suppress overlapping boxes itself -- unlike the YOLOv8 path, where one
    # of two heavily-overlapping same-class boxes is suppressed (see
    # test_nms_collapses_overlapping_boxes_of_same_class above).
    class_id = CLASS_NAMES.index("spaghetti")
    rows = [
        _e2e_row(300, 300, 400, 400, class_id, 0.9),
        _e2e_row(305, 305, 405, 405, class_id, 0.7),  # heavily overlapping, same class
    ]
    raw = _e2e_raw(rows)

    dets = _postprocess_e2e(raw, severity_map={"spaghetti": Severity.CATASTROPHIC})

    assert len(dets) == 2
    assert sorted(d.confidence for d in dets) == pytest.approx([0.7, 0.9])


def test_postprocess_end2end_clips_out_of_bounds_coordinates():
    class_id = CLASS_NAMES.index("spaghetti")
    # A decoded box extending past the letterboxed frame on every side, as
    # the model can legitimately emit.
    raw = _e2e_raw([_e2e_row(-5.0, -5.0, 645.0, 645.0, class_id, 0.9)])

    dets = _postprocess_e2e(
        raw,
        orig_shape=(640, 640),
        severity_map={"spaghetti": Severity.CATASTROPHIC},
    )

    assert len(dets) == 1
    x1, y1, x2, y2 = dets[0].bbox
    assert x1 == pytest.approx(0.0)
    assert y1 == pytest.approx(0.0)
    assert x2 == pytest.approx(639.0)
    assert y2 == pytest.approx(639.0)


def test_postprocess_end2end_empty_raw_output_yields_no_detections():
    raw = _e2e_raw([])  # all rows are zero-confidence padding
    dets = _postprocess_e2e(raw)
    assert dets == ()


# --------------------------------------------------------------------------
# detect_layout (the "auto" layout heuristic) and explicit overrides
# --------------------------------------------------------------------------


def test_detect_layout_identifies_yolov8_shape():
    raw = np.zeros((1, 4 + NUM_CLASSES, 5376), dtype=np.float32)
    assert detect_layout(raw, num_classes=NUM_CLASSES) == "yolov8"


def test_detect_layout_identifies_end2end_shape():
    raw = np.zeros((1, 300, 6), dtype=np.float32)
    assert detect_layout(raw, num_classes=NUM_CLASSES) == "end2end"


def test_detect_layout_prefers_yolov8_when_num_classes_two_ambiguous():
    # 4 + num_classes == 6 when num_classes == 2, so a (1, 300, 6) output is
    # genuinely indistinguishable from an end2end output by shape alone.
    # The 4+nc check runs first, so "auto" must resolve to "yolov8" here --
    # documented ambiguity, not a bug.
    raw = np.zeros((1, 300, 6), dtype=np.float32)
    assert detect_layout(raw, num_classes=2) == "yolov8"


def test_detect_layout_raises_on_uninterpretable_shape():
    raw = np.zeros((1, 7, 100), dtype=np.float32)
    with pytest.raises(ValueError):
        detect_layout(raw, num_classes=NUM_CLASSES)


def test_explicit_layout_config_overrides_heuristic():
    # A declared output shape the heuristic alone would read as "yolov8"...
    yolov8_shape = (1, 4 + NUM_CLASSES, 5376)
    assert detect_layout(np.zeros(yolov8_shape), NUM_CLASSES) == "yolov8"

    # ...but an explicit config value must win outright.
    assert _resolve_layout("end2end", yolov8_shape, NUM_CLASSES) == "end2end"
    end2end_shape = (1, 300, 6)
    assert _resolve_layout("yolov8", end2end_shape, NUM_CLASSES) == "yolov8"

    # "auto" still defers to the heuristic in both directions.
    assert _resolve_layout("auto", yolov8_shape, NUM_CLASSES) == "yolov8"
    assert _resolve_layout("auto", end2end_shape, NUM_CLASSES) == "end2end"


# --------------------------------------------------------------------------
# _static_input_size helper
# --------------------------------------------------------------------------


def test_static_input_size_static_square():
    assert _static_input_size([1, 3, 640, 640]) == 640


def test_static_input_size_dynamic_axes_falls_back():
    assert _static_input_size([1, 3, "height", "width"]) is None
    # A dynamic batch dim doesn't matter -- only H/W (indices 2, 3) need to
    # be static ints for the model's own input size to be usable.
    assert _static_input_size(["batch", 3, 640, 640]) == 640


def test_static_input_size_non_square_falls_back():
    assert _static_input_size([1, 3, 640, 480]) is None


def test_static_input_size_wrong_rank_falls_back():
    assert _static_input_size([1, 3, 640]) is None
    assert _static_input_size(None) is None


# --------------------------------------------------------------------------
# OnnxYoloDetector construction without a model file
# --------------------------------------------------------------------------


def test_onnx_yolo_detector_missing_model_raises_filenotfound(tmp_path):
    cfg = DetectorConfig(model_path=str(tmp_path / "does_not_exist.onnx"))
    with pytest.raises(FileNotFoundError):
        OnnxYoloDetector(cfg)


# --------------------------------------------------------------------------
# MockDetector
# --------------------------------------------------------------------------


def test_mock_detector_rejects_empty_script():
    with pytest.raises(ValueError):
        MockDetector([])


def test_mock_detector_follows_script_then_clamps():
    detector = MockDetector([0.0, 0.6, 0.9])
    frame = _make_frame()

    r0 = detector.infer(frame)
    assert r0.detections == ()
    assert r0.p_failure == 0.0

    r1 = detector.infer(frame)
    assert len(r1.detections) == 1
    det = r1.detections[0]
    assert det.class_name == "spaghetti"
    assert det.severity is Severity.CATASTROPHIC
    assert det.confidence == pytest.approx(0.6)
    assert r1.p_failure == pytest.approx(0.6)

    r2 = detector.infer(frame)
    assert r2.p_failure == pytest.approx(0.9)

    # Past the end of the script: clamp on the last scripted value.
    r3 = detector.infer(frame)
    r4 = detector.infer(frame)
    assert r3.p_failure == pytest.approx(0.9)
    assert r4.p_failure == pytest.approx(0.9)

    assert detector.call_count == 5


def test_mock_detector_cycles_when_configured():
    detector = MockDetector([0.0, 0.8], cycle=True)
    frame = _make_frame()

    results = [detector.infer(frame).p_failure for _ in range(4)]
    assert results == pytest.approx([0.0, 0.8, 0.0, 0.8])
    assert detector.call_count == 4


def test_mock_detector_is_a_detector_and_supports_context_manager():
    with MockDetector([0.5]) as detector:
        assert isinstance(detector, Detector)
        result = detector.infer(_make_frame())
        assert result.p_failure == pytest.approx(0.5)
