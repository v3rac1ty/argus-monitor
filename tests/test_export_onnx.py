"""Tests for training/export_onnx.py's pure, model-file-free logic:
`resolve_num_classes`, `classify_output_shape`, and `parse_args` defaults.

`classify_output_shape` is a thin wrapper around
`argus.detectors.onnx_yolo.detect_layout` -- the same helper the runtime
detector uses to pick a layout -- so these tests double as a check that the
export script's verification step stays in sync with the runtime rather than
re-implementing its own copy of the shape rule. No ONNX file, onnxruntime
session, or trained weights are needed: everything here is synthetic numpy
arrays and plain function calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from export_onnx import (  # noqa: E402
    DEFAULT_NUM_CLASSES,
    classify_output_shape,
    parse_args,
    resolve_num_classes,
)

NUM_CLASSES = 5


# --------------------------------------------------------------------------
# parse_args
# --------------------------------------------------------------------------


def test_parse_args_defaults():
    args = parse_args(["--weights", "runs/train/x/weights/best.pt"])
    assert args.imgsz == 640
    assert args.opset == 12
    assert args.nc is None
    assert args.no_simplify is False


def test_parse_args_nc_override():
    args = parse_args(["--weights", "best.pt", "--nc", "3", "--imgsz", "512"])
    assert args.nc == 3
    assert args.imgsz == 512


# --------------------------------------------------------------------------
# resolve_num_classes
# --------------------------------------------------------------------------


def test_resolve_num_classes_cli_takes_priority():
    model_names = {0: "a", 1: "b", 2: "c"}
    nc, source = resolve_num_classes(7, model_names)
    assert nc == 7
    assert "--nc=7" in source


def test_resolve_num_classes_from_model_names():
    model_names = {0: "error extrusion", 1: "spaghetti", 2: "stringing", 3: "warping", 4: "zits"}
    nc, source = resolve_num_classes(None, model_names)
    assert nc == 5
    assert "spaghetti" in source


def test_resolve_num_classes_from_model_names_unsorted_keys():
    # dict insertion order isn't guaranteed to match class id order -- make
    # sure the class list in the reported source is still id-ordered.
    model_names = {2: "c", 0: "a", 1: "b"}
    nc, source = resolve_num_classes(None, model_names)
    assert nc == 3
    assert "['a', 'b', 'c']" in source


def test_resolve_num_classes_falls_back_to_default_when_no_names():
    nc, source = resolve_num_classes(None, None)
    assert nc == DEFAULT_NUM_CLASSES
    assert "default" in source


def test_resolve_num_classes_falls_back_to_default_when_empty_names():
    nc, source = resolve_num_classes(None, {})
    assert nc == DEFAULT_NUM_CLASSES


# --------------------------------------------------------------------------
# classify_output_shape
# --------------------------------------------------------------------------


def test_classify_output_shape_yolov8_native():
    raw = np.zeros((1, 4 + NUM_CLASSES, 100), dtype=np.float32)
    shape, layout = classify_output_shape(raw, NUM_CLASSES)
    assert shape == (1, 9, 100)
    assert layout == "yolov8"


def test_classify_output_shape_yolov8_transposed():
    raw = np.zeros((1, 100, 4 + NUM_CLASSES), dtype=np.float32)
    shape, layout = classify_output_shape(raw, NUM_CLASSES)
    assert shape == (1, 100, 9)
    assert layout == "yolov8"


def test_classify_output_shape_end2end():
    raw = np.zeros((1, 300, 6), dtype=np.float32)
    shape, layout = classify_output_shape(raw, NUM_CLASSES)
    assert shape == (1, 300, 6)
    assert layout == "end2end"


def test_classify_output_shape_unparseable_raises_assertion_error_naming_both_layouts():
    raw = np.zeros((1, 7, 100), dtype=np.float32)  # neither 9 nor 6 anywhere
    with pytest.raises(AssertionError) as excinfo:
        classify_output_shape(raw, NUM_CLASSES)
    message = str(excinfo.value)
    assert "(1, 7, 100)" in message
    assert "yolov8" in message.lower() or "YOLOv8" in message
    assert "end2end" in message.lower() or "end-to-end" in message.lower()


def test_classify_output_shape_matches_runtime_detect_layout():
    """classify_output_shape must never disagree with the runtime's own
    detect_layout -- exercise it against detect_layout directly to guard
    against the two re-diverging."""
    from argus.detectors.onnx_yolo import detect_layout

    raw = np.zeros((1, 300, 6), dtype=np.float32)
    _, layout = classify_output_shape(raw, NUM_CLASSES)
    assert layout == detect_layout(raw, NUM_CLASSES)
