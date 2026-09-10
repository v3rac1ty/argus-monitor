"""Tests for argus.detectors.hailo: pure pre/post-processing functions and
HailoDetector construction/inference/close, all against a lightweight fake
`hailo_platform` module -- no real HailoRT package and no Hailo-8 hardware
anywhere in this file, matching the style of test_classifier_detector.py's
fake onnxruntime session.

Coverage matches the deliverable's requirements:
  - preprocessing geometry matches `preprocess_classify`'s crop exactly, but
    yields uint8 NHWC (not float32 NCHW in [0, 1])
  - the output path (postprocess_classify reuse) produces IDENTICAL
    Detections to the ONNX path given the same raw probabilities
  - empty `class_names` raises a clear error
  - `close()` releases every HailoRT resource acquired at construction
  - `build_detector` dispatch for `kind: "hailo"` is covered separately in
    tests/test_service.py, alongside the existing classification/detection
    dispatch tests it mirrors
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from argus.config import DetectorConfig
from argus.detectors.classifier import postprocess_classify, preprocess_classify
from argus.detectors.hailo import HailoDetector, _flatten_hailo_output, preprocess_classify_hailo
from argus.types import Frame, Severity

CLASS_NAMES = ("failure", "normal")
_THRESHOLDS = {"failure": 0.75, "normal": 0.50}
_SEVERITY = {"failure": Severity.CATASTROPHIC, "normal": Severity.COSMETIC}

INPUT_NAME = "input_layer1"
OUTPUT_NAME = "output_layer1"


def _one_hot_logits(index: int, num_classes: int, prob: float) -> np.ndarray:
    """Build a raw probability-like row (sums to 1, already in [0,1]) with
    `prob` at `index` and the remainder spread evenly over the rest."""
    remainder = (1.0 - prob) / (num_classes - 1)
    row = np.full(num_classes, remainder, dtype=np.float64)
    row[index] = prob
    return row


# --------------------------------------------------------------------------
# preprocess_classify_hailo -- pure function, no hailo_platform dependency
# --------------------------------------------------------------------------


def test_preprocess_classify_hailo_shape_dtype_range():
    image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    blob = preprocess_classify_hailo(image, 128)
    assert blob.shape == (1, 128, 128, 3)
    assert blob.dtype == np.uint8
    assert blob.min() >= 0
    assert blob.max() <= 255


def test_preprocess_classify_hailo_portrait_and_landscape_shapes():
    for image in (
        np.random.randint(0, 256, (600, 300, 3), dtype=np.uint8),  # portrait
        np.random.randint(0, 256, (300, 600, 3), dtype=np.uint8),  # landscape
    ):
        blob = preprocess_classify_hailo(image, 128)
        assert blob.shape == (1, 128, 128, 3)
        assert blob.dtype == np.uint8


def test_preprocess_classify_hailo_matches_onnx_crop_geometry():
    # Same source image through both backends' preprocessors: the ONNX path
    # (float32 NCHW, RGB, /255) and the Hailo path (uint8 NHWC, RGB, no
    # scaling) must see the EXACT same crop -- both call the same shared
    # `resize_and_center_crop`. Undo Hailo's NHWC/uint8/unscaled encoding and
    # the two should be numerically identical (up to uint8 rounding).
    image = np.random.randint(0, 256, (400, 220, 3), dtype=np.uint8)
    size = 96

    onnx_blob = preprocess_classify(image, size)  # (1, 3, size, size) float32 in [0, 1]
    hailo_blob = preprocess_classify_hailo(image, size)  # (1, size, size, 3) uint8

    assert hailo_blob.shape == (1, size, size, 3)
    assert onnx_blob.shape == (1, 3, size, size)

    reconstructed = hailo_blob[0].astype(np.float32).transpose(2, 0, 1) / 255.0
    np.testing.assert_allclose(reconstructed, onnx_blob[0], atol=1.0 / 255.0 + 1e-6)


def test_preprocess_classify_hailo_no_host_side_scaling():
    # A bright, uniform image should stay near 255 in the Hailo blob -- if
    # /255 scaling were mistakenly applied here (duplicating what must
    # instead be baked into the compiled HEF), every value would collapse
    # toward 1 (as an int-truncated uint8, toward 0 or 1) instead.
    image = np.full((64, 64, 3), 250, dtype=np.uint8)
    blob = preprocess_classify_hailo(image, 32)
    assert blob.min() >= 249


def test_preprocess_classify_hailo_center_crop_picks_middle_region():
    # Same construction as test_classifier_detector.py's equivalent ONNX
    # test: a horizontal gradient image whose short side already equals
    # input_size, so the resize step is a no-op and only the center crop can
    # move which columns survive.
    width, height, input_size = 300, 100, 100
    columns = np.arange(width, dtype=np.uint8)
    image = np.tile(columns[np.newaxis, :, np.newaxis], (height, 1, 3))  # BGR, all channels equal

    blob = preprocess_classify_hailo(image, input_size)
    assert blob.shape == (1, input_size, input_size, 3)

    # NHWC layout: blob[0, row, :, channel]. Channel 0 is R (post BGR->RGB),
    # which equals the original column index for this synthetic image.
    recovered_columns = blob[0, 0, :, 0].astype(int)
    expected_left = (width - input_size) // 2
    expected = np.arange(expected_left, expected_left + input_size)
    np.testing.assert_array_equal(recovered_columns, expected)


# --------------------------------------------------------------------------
# _flatten_hailo_output -- pure function, no hailo_platform dependency
# --------------------------------------------------------------------------


def test_flatten_hailo_output_handles_various_shapes():
    flat = np.array([0.1, 0.2, 0.7], dtype=np.float32)
    for shaped in (flat, flat.reshape(1, 3), flat.reshape(1, 1, 1, 3)):
        result = _flatten_hailo_output(shaped)
        np.testing.assert_array_equal(result, flat)


# --------------------------------------------------------------------------
# Fake hailo_platform -- a minimal stand-in exposing only the surface
# HailoDetector actually touches, so its lifecycle (construction, infer,
# close) can be exercised with no real HailoRT package or Hailo-8 device.
# --------------------------------------------------------------------------


class _FakeVStreamInfo:
    def __init__(self, name: str):
        self.name = name


class _FakeContextManager:
    """Stand-in for HailoRT objects used as `with x:` context managers
    (network_group.activate(...) and InferVStreams(...), per HailoRT's
    published Python API examples)."""

    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    def __enter__(self) -> "_FakeContextManager":
        self.entered = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.exited = True
        return False


class _FakeInferPipeline(_FakeContextManager):
    def __init__(self, network_group: Any, input_params: Any, output_params: Any, output_value: np.ndarray) -> None:
        super().__init__()
        self.network_group = network_group
        self.input_params = input_params
        self.output_params = output_params
        self.output_value = output_value
        self.infer_calls: list[dict] = []

    def infer(self, input_data: dict) -> dict:
        self.infer_calls.append(input_data)
        return {OUTPUT_NAME: self.output_value}


class _FakeNetworkGroup:
    def __init__(self) -> None:
        self.activations: list[_FakeContextManager] = []

    def create_params(self) -> str:
        return "network_group_params"

    def activate(self, params: Any) -> _FakeContextManager:
        cm = _FakeContextManager()
        self.activations.append(cm)
        return cm


class _FakeHEF:
    def __init__(self, path: str) -> None:
        self.path = path

    def get_input_vstream_infos(self) -> list[_FakeVStreamInfo]:
        return [_FakeVStreamInfo(INPUT_NAME)]

    def get_output_vstream_infos(self) -> list[_FakeVStreamInfo]:
        return [_FakeVStreamInfo(OUTPUT_NAME)]


class _FakeVDevice:
    def __init__(self, network_group: _FakeNetworkGroup) -> None:
        self._network_group = network_group
        self.released = False
        self.configure_calls: list[tuple] = []

    def configure(self, hef: Any, configure_params: Any) -> list[_FakeNetworkGroup]:
        self.configure_calls.append((hef, configure_params))
        return [self._network_group]

    def release(self) -> None:
        self.released = True


class _FakeConfigureParams:
    @staticmethod
    def create_from_hef(hef: Any, interface: Any) -> tuple:
        return ("configure_params", hef, interface)


class _FakeVStreamParams:
    @staticmethod
    def make(network_group: Any, quantized: bool, format_type: Any) -> dict:
        return {"network_group": network_group, "quantized": quantized, "format_type": format_type}


class _Enum:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _install_fake_hailo_platform(monkeypatch: pytest.MonkeyPatch, output_value: np.ndarray):
    """Registers a fake `hailo_platform` module in sys.modules so
    HailoDetector's lazy `import hailo_platform` (inside __init__) picks it
    up -- the same sys.modules-injection trick as monkeypatching
    onnxruntime, just one level further removed since this import happens
    inside a method rather than at module scope. Returns
    (network_group, vdevice, infer_pipelines) for lifecycle assertions."""
    network_group = _FakeNetworkGroup()
    vdevice = _FakeVDevice(network_group)
    infer_pipelines: list[_FakeInferPipeline] = []

    def _infer_vstreams_factory(ng: Any, input_params: Any, output_params: Any) -> _FakeInferPipeline:
        pipeline = _FakeInferPipeline(ng, input_params, output_params, output_value)
        infer_pipelines.append(pipeline)
        return pipeline

    fake_module = types.ModuleType("hailo_platform")
    fake_module.HEF = _FakeHEF  # type: ignore[attr-defined]
    fake_module.VDevice = lambda: vdevice  # type: ignore[attr-defined]
    fake_module.ConfigureParams = _FakeConfigureParams  # type: ignore[attr-defined]
    fake_module.HailoStreamInterface = _Enum(PCIe="PCIe")  # type: ignore[attr-defined]
    fake_module.FormatType = _Enum(UINT8="UINT8", FLOAT32="FLOAT32")  # type: ignore[attr-defined]
    fake_module.InputVStreamParams = _FakeVStreamParams  # type: ignore[attr-defined]
    fake_module.OutputVStreamParams = _FakeVStreamParams  # type: ignore[attr-defined]
    fake_module.InferVStreams = _infer_vstreams_factory  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "hailo_platform", fake_module)
    return network_group, vdevice, infer_pipelines


def _hailo_config(
    model_path: Path,
    class_names: tuple[str, ...] = CLASS_NAMES,
    input_size: int = 64,
) -> DetectorConfig:
    return DetectorConfig(
        kind="hailo",
        model_path=str(model_path),
        input_size=input_size,
        class_names=class_names,
        default_threshold=0.50,
        class_thresholds=dict(_THRESHOLDS),
        severity=dict(_SEVERITY),
    )


def _fake_model_path(tmp_path: Path) -> Path:
    """A path that merely needs to exist -- HailoDetector only checks
    is_file() before handing it to the (fake) hailo_platform.HEF
    constructor; its bytes are never actually parsed as a HEF."""
    path = tmp_path / "model.hef"
    path.write_bytes(b"")
    return path


# --------------------------------------------------------------------------
# Construction-time validation -- these must not require hailo_platform at
# all, since each raises before (or without ever reaching) the lazy import.
# --------------------------------------------------------------------------


def test_hailo_detector_empty_class_names_raises_clear_error(tmp_path):
    cfg = _hailo_config(_fake_model_path(tmp_path), class_names=())
    with pytest.raises(ValueError, match="class_names"):
        HailoDetector(cfg)


def test_hailo_detector_missing_model_file_raises_file_not_found(tmp_path):
    cfg = _hailo_config(tmp_path / "does_not_exist.hef")
    with pytest.raises(FileNotFoundError):
        HailoDetector(cfg)


def test_hailo_detector_raises_clear_error_when_hailo_platform_missing(tmp_path, monkeypatch):
    # sys.modules[name] = None is the standard way to make `import name`
    # raise ImportError/ModuleNotFoundError without needing the real
    # package absent from the environment.
    monkeypatch.setitem(sys.modules, "hailo_platform", None)
    cfg = _hailo_config(_fake_model_path(tmp_path))
    with pytest.raises(ImportError, match="hailo_platform"):
        HailoDetector(cfg)


# --------------------------------------------------------------------------
# Output path -- identical Detections to the ONNX path given the same raw
# probabilities, via the fake hailo_platform module.
# --------------------------------------------------------------------------


def test_hailo_detector_output_matches_classifier_path_for_same_probabilities(tmp_path, monkeypatch):
    prob = 0.9
    raw = _one_hot_logits(CLASS_NAMES.index("failure"), len(CLASS_NAMES), prob).astype(np.float32)
    _network_group, _vdevice, pipelines = _install_fake_hailo_platform(monkeypatch, output_value=raw)

    cfg = _hailo_config(_fake_model_path(tmp_path))
    detector = HailoDetector(cfg)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    try:
        result = detector.infer(Frame(image=image, timestamp=0.0, seq=0))
    finally:
        detector.close()

    expected = postprocess_classify(raw, CLASS_NAMES, _THRESHOLDS, 0.50, _SEVERITY, image.shape[:2])
    assert result.detections == expected
    assert result.detections[0].class_name == "failure"
    assert result.detections[0].severity is Severity.CATASTROPHIC
    assert result.detections[0].confidence == pytest.approx(prob, abs=1e-6)
    assert result.p_failure == pytest.approx(prob, abs=1e-6)
    assert len(pipelines) == 1 and pipelines[0].infer_calls, "infer() must call the HailoRT infer pipeline"
    assert list(pipelines[0].infer_calls[0].keys()) == [INPUT_NAME]


def test_hailo_detector_normal_prediction_yields_zero_detections(tmp_path, monkeypatch):
    raw = _one_hot_logits(CLASS_NAMES.index("normal"), len(CLASS_NAMES), 0.99).astype(np.float32)
    _install_fake_hailo_platform(monkeypatch, output_value=raw)

    cfg = _hailo_config(_fake_model_path(tmp_path))
    detector = HailoDetector(cfg)
    try:
        result = detector.infer(Frame(image=np.zeros((64, 64, 3), dtype=np.uint8), timestamp=0.0, seq=0))
    finally:
        detector.close()

    assert result.detections == ()
    assert result.p_failure == 0.0


def test_hailo_detector_below_threshold_yields_zero_detections(tmp_path, monkeypatch):
    # failure's threshold is 0.75; 0.6 should not clear it -- identical rule
    # to the ONNX path's postprocess_classify.
    raw = _one_hot_logits(CLASS_NAMES.index("failure"), len(CLASS_NAMES), 0.6).astype(np.float32)
    _install_fake_hailo_platform(monkeypatch, output_value=raw)

    cfg = _hailo_config(_fake_model_path(tmp_path))
    detector = HailoDetector(cfg)
    try:
        result = detector.infer(Frame(image=np.zeros((64, 64, 3), dtype=np.uint8), timestamp=0.0, seq=0))
    finally:
        detector.close()

    assert result.detections == ()


def test_hailo_detector_output_shape_1xn_accepted(tmp_path, monkeypatch):
    # HailoRT's real output shape for a single-input/single-output
    # classification HEF is unverified (see module docstring) -- exercise a
    # batched (1, N) shape to prove _flatten_hailo_output's reshape covers it.
    raw = _one_hot_logits(CLASS_NAMES.index("failure"), len(CLASS_NAMES), 0.9).astype(np.float32)
    _install_fake_hailo_platform(monkeypatch, output_value=raw.reshape(1, -1))

    cfg = _hailo_config(_fake_model_path(tmp_path))
    detector = HailoDetector(cfg)
    try:
        result = detector.infer(Frame(image=np.zeros((64, 64, 3), dtype=np.uint8), timestamp=0.0, seq=0))
    finally:
        detector.close()

    assert len(result.detections) == 1
    assert result.detections[0].class_name == "failure"


# --------------------------------------------------------------------------
# Lifecycle -- construction wires VDevice/network group/vstreams; close()
# releases all of it and infer() then raises.
# --------------------------------------------------------------------------


def test_hailo_detector_close_releases_hailort_resources(tmp_path, monkeypatch):
    raw = _one_hot_logits(0, len(CLASS_NAMES), 0.9).astype(np.float32)
    network_group, vdevice, pipelines = _install_fake_hailo_platform(monkeypatch, output_value=raw)

    cfg = _hailo_config(_fake_model_path(tmp_path))
    detector = HailoDetector(cfg)

    assert vdevice.configure_calls, "VDevice.configure should be called during construction"
    assert network_group.activations and network_group.activations[-1].entered
    assert pipelines and pipelines[0].entered
    assert vdevice.released is False

    detector.close()

    assert network_group.activations[-1].exited
    assert pipelines[0].exited
    assert vdevice.released is True

    with pytest.raises(RuntimeError, match="closed"):
        detector.infer(Frame(image=np.zeros((64, 64, 3), dtype=np.uint8), timestamp=0.0, seq=0))

    # Idempotent: ArgusService.close() calls detector.close() unconditionally
    # regardless of prior state (see ArgusService.close()'s own docstring).
    detector.close()
