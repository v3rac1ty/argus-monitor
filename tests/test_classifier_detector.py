"""Tests for argus.detectors.classifier: pure pre/post-processing functions,
ONNX class-name metadata reconciliation, and ClassifierDetector construction.

No real model file and no network access -- everything here operates on
synthetic numpy arrays and a lightweight fake onnxruntime session, matching
the style of test_onnx_yolo_detector.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

from argus.config import DetectorConfig
from argus.detectors import classifier as classifier_module
from argus.detectors.classifier import (
    ClassifierDetector,
    class_names_from_onnx_metadata,
    postprocess_classify,
    preprocess_classify,
    probabilities_from_output,
    resolve_class_names,
    softmax,
)
from argus.types import DetectionResult, Severity

CLASS_NAMES = ("normal", "spaghetti", "cracking", "layer_shifting", "stringing", "warping")

_DEFAULT_THRESHOLDS = {
    "spaghetti": 0.75,
    "cracking": 0.75,
    "layer_shifting": 0.75,
    "stringing": 0.50,
    "warping": 0.75,
}

_DEFAULT_SEVERITY = {
    "spaghetti": Severity.CATASTROPHIC,
    "cracking": Severity.CATASTROPHIC,
    "layer_shifting": Severity.CATASTROPHIC,
    "stringing": Severity.COSMETIC,
    "warping": Severity.COSMETIC,
}


def _one_hot_logits(index: int, num_classes: int, prob: float) -> np.ndarray:
    """Build a raw probability-like row (sums to 1, already in [0,1]) with
    `prob` at `index` and the remainder spread evenly over the rest."""
    remainder = (1.0 - prob) / (num_classes - 1)
    row = np.full(num_classes, remainder, dtype=np.float64)
    row[index] = prob
    return row


# --------------------------------------------------------------------------
# preprocess_classify
# --------------------------------------------------------------------------


def test_preprocess_classify_square_shape_dtype_range():
    image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    blob = preprocess_classify(image, 128)
    assert blob.shape == (1, 3, 128, 128)
    assert blob.dtype == np.float32
    assert blob.min() >= 0.0
    assert blob.max() <= 1.0


def test_preprocess_classify_portrait_shape():
    # height > width -- short side is width
    image = np.random.randint(0, 256, (600, 300, 3), dtype=np.uint8)
    blob = preprocess_classify(image, 128)
    assert blob.shape == (1, 3, 128, 128)
    assert blob.dtype == np.float32
    assert 0.0 <= blob.min() and blob.max() <= 1.0


def test_preprocess_classify_landscape_shape():
    # width > height -- short side is height
    image = np.random.randint(0, 256, (300, 600, 3), dtype=np.uint8)
    blob = preprocess_classify(image, 128)
    assert blob.shape == (1, 3, 128, 128)
    assert blob.dtype == np.float32
    assert 0.0 <= blob.min() and blob.max() <= 1.0


def test_preprocess_classify_center_crop_picks_middle_region():
    # Build a landscape image (short side = height = 100) that is a
    # horizontal gradient of distinct column values, so we can tell exactly
    # which columns survived the center crop after the resize-short-side
    # step (which is a no-op here since height already equals input_size).
    width, height, input_size = 300, 100, 100
    columns = np.arange(width, dtype=np.uint8)
    image = np.tile(columns[np.newaxis, :, np.newaxis], (height, 1, 3))  # BGR, all channels equal

    blob = preprocess_classify(image, input_size)
    assert blob.shape == (1, 3, input_size, input_size)

    # Resize is a no-op on the short side (height == input_size already) and
    # width is untouched by the short-side resize, so the crop should keep
    # exactly the middle `input_size` columns: [100, 200).
    # blob channel 0 corresponds to R (post BGR->RGB), which equals the
    # original column index / 255 for this synthetic image.
    recovered_columns = np.round(blob[0, 0, 0, :] * 255).astype(int)
    expected_left = (width - input_size) // 2
    expected = np.arange(expected_left, expected_left + input_size)
    np.testing.assert_array_equal(recovered_columns, expected)


# --------------------------------------------------------------------------
# softmax
# --------------------------------------------------------------------------


def test_softmax_sums_to_one():
    logits = np.array([1.0, 2.0, 3.0, 0.5])
    probs = softmax(logits)
    assert probs.shape == logits.shape
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(probs >= 0.0)


def test_softmax_numerically_stable_on_large_logits():
    # Naive softmax (exp without max-subtraction) overflows/NaNs on values
    # this large; a stable implementation must not.
    logits = np.array([1000.0, 1001.0, 999.0, 500.0])
    probs = softmax(logits)
    assert np.all(np.isfinite(probs))
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)
    # class index 1 has the largest logit -> should dominate
    assert np.argmax(probs) == 1


def test_softmax_uniform_on_equal_logits():
    logits = np.zeros(5)
    probs = softmax(logits)
    np.testing.assert_allclose(probs, np.full(5, 0.2), atol=1e-9)


# --------------------------------------------------------------------------
# probabilities_from_output
# --------------------------------------------------------------------------


def test_probabilities_from_output_leaves_normalized_distribution_untouched():
    probs_in = _one_hot_logits(2, 6, 0.9)
    probs_out = probabilities_from_output(probs_in)
    np.testing.assert_allclose(probs_out, probs_in)


def test_probabilities_from_output_softmaxes_raw_logits():
    logits = np.array([2.0, 8.0, -1.0, 0.0])
    probs = probabilities_from_output(logits)
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)
    # Raw logits don't sum to ~1 and aren't all in [0, 1], so this must have
    # gone through softmax rather than being returned as-is.
    assert not np.allclose(probs, logits)


def test_probabilities_from_output_no_double_softmax():
    # Simulate a classification head that already applied softmax: its
    # output is sharply peaked (e.g. 0.97 on the winning class). Applying
    # softmax a second time would flatten that peak dramatically because
    # softmax over already-small, close-together probabilities compresses
    # them further toward uniform. Assert the peak survives essentially
    # intact, proving no second softmax was applied.
    already_softmaxed = _one_hot_logits(3, 6, 0.97)
    result = probabilities_from_output(already_softmaxed)
    assert result[3] == pytest.approx(0.97, abs=1e-9)

    # Contrast: explicitly double-softmaxing would crush that peak well
    # below the input value.
    double_softmaxed = softmax(already_softmaxed)
    assert double_softmaxed[3] < 0.97 - 0.1


# --------------------------------------------------------------------------
# postprocess_classify
# --------------------------------------------------------------------------


def test_normal_prediction_yields_zero_detections_and_zero_p_failure():
    raw = _one_hot_logits(CLASS_NAMES.index("normal"), len(CLASS_NAMES), 0.99)
    detections = postprocess_classify(
        raw, CLASS_NAMES, _DEFAULT_THRESHOLDS, 0.50, _DEFAULT_SEVERITY, (480, 640)
    )
    assert detections == ()

    result = DetectionResult(detections=detections, inference_ms=1.0)
    assert result.p_failure == 0.0


def test_confident_spaghetti_prediction_yields_one_catastrophic_detection():
    prob = 0.9
    raw = _one_hot_logits(CLASS_NAMES.index("spaghetti"), len(CLASS_NAMES), prob)
    orig_shape = (480, 640)
    detections = postprocess_classify(
        raw, CLASS_NAMES, _DEFAULT_THRESHOLDS, 0.50, _DEFAULT_SEVERITY, orig_shape
    )
    assert len(detections) == 1
    det = detections[0]
    assert det.class_name == "spaghetti"
    assert det.severity is Severity.CATASTROPHIC
    assert det.confidence == pytest.approx(prob, abs=1e-9)
    assert det.bbox == (0.0, 0.0, float(orig_shape[1]), float(orig_shape[0]))

    result = DetectionResult(detections=detections, inference_ms=1.0)
    assert result.p_failure == pytest.approx(prob, abs=1e-9)


def test_spaghetti_below_threshold_yields_zero_detections():
    # spaghetti's threshold is 0.75; 0.6 should not clear it.
    prob = 0.6
    raw = _one_hot_logits(CLASS_NAMES.index("spaghetti"), len(CLASS_NAMES), prob)
    detections = postprocess_classify(
        raw, CLASS_NAMES, _DEFAULT_THRESHOLDS, 0.50, _DEFAULT_SEVERITY, (480, 640)
    )
    assert detections == ()


def test_unknown_class_name_defaults_to_cosmetic():
    names = ("normal", "mystery_defect")
    raw = _one_hot_logits(1, 2, 0.9)
    # mystery_defect has no entry in class_thresholds or severity_map.
    detections = postprocess_classify(raw, names, {}, 0.50, {}, (480, 640))
    assert len(detections) == 1
    det = detections[0]
    assert det.class_name == "mystery_defect"
    assert det.severity is Severity.COSMETIC

    result = DetectionResult(detections=detections, inference_ms=1.0)
    assert result.p_failure == 0.0


def test_cosmetic_class_prediction_gives_zero_p_failure_but_is_in_detections():
    prob = 0.8
    raw = _one_hot_logits(CLASS_NAMES.index("stringing"), len(CLASS_NAMES), prob)
    detections = postprocess_classify(
        raw, CLASS_NAMES, _DEFAULT_THRESHOLDS, 0.50, _DEFAULT_SEVERITY, (480, 640)
    )
    assert len(detections) == 1
    assert detections[0].class_name == "stringing"
    assert detections[0].severity is Severity.COSMETIC

    result = DetectionResult(detections=detections, inference_ms=1.0)
    assert result.p_failure == 0.0
    assert result.detections == detections


def test_normal_case_insensitive():
    names = ("Normal", "spaghetti")
    raw = _one_hot_logits(0, 2, 0.99)
    detections = postprocess_classify(raw, names, {}, 0.50, {}, (480, 640))
    assert detections == ()


def test_wrong_length_output_raises_value_error():
    raw = np.array([0.5, 0.5])  # only 2 classes, but CLASS_NAMES has 6
    with pytest.raises(ValueError):
        postprocess_classify(raw, CLASS_NAMES, _DEFAULT_THRESHOLDS, 0.50, _DEFAULT_SEVERITY, (480, 640))


def test_batched_1xn_shape_accepted():
    prob = 0.9
    raw = _one_hot_logits(CLASS_NAMES.index("spaghetti"), len(CLASS_NAMES), prob)[np.newaxis, :]
    assert raw.shape == (1, len(CLASS_NAMES))
    detections = postprocess_classify(
        raw, CLASS_NAMES, _DEFAULT_THRESHOLDS, 0.50, _DEFAULT_SEVERITY, (480, 640)
    )
    assert len(detections) == 1
    assert detections[0].class_name == "spaghetti"


def test_batch_size_greater_than_one_raises_value_error():
    raw = np.tile(_one_hot_logits(1, len(CLASS_NAMES), 0.9), (2, 1))
    assert raw.shape == (2, len(CLASS_NAMES))
    with pytest.raises(ValueError):
        postprocess_classify(raw, CLASS_NAMES, _DEFAULT_THRESHOLDS, 0.50, _DEFAULT_SEVERITY, (480, 640))


# --------------------------------------------------------------------------
# Fakes -- a minimal onnxruntime.InferenceSession stand-in exposing only the
# surface ClassifierDetector / resolve_class_names actually touch, so the
# metadata-reconciliation logic (and ClassifierDetector construction itself)
# can be exercised with no real .onnx file and no onnxruntime model loading.
# --------------------------------------------------------------------------

#: The real model.names ordering baked into models/argus_cls.onnx, as
#: verified by inspecting its ONNX metadata_props directly (Ultralytics
#: assigns indices alphabetically from the training folder names).
REAL_MODEL_CLASS_ORDER = ("cracking", "layer_shifting", "normal", "spaghetti", "stringing", "warping")


class _FakeInput:
    def __init__(self, name: str = "input", shape: tuple[object, ...] = (1, 3, 512, 512)):
        self.name = name
        self.shape = shape


class _FakeModelMeta:
    def __init__(self, custom_metadata_map: dict[str, str]):
        self.custom_metadata_map = custom_metadata_map


class _FakeSession:
    """Stand-in for onnxruntime.InferenceSession exposing just
    get_modelmeta() and get_inputs() -- everything ClassifierDetector.
    __init__ and resolve_class_names touch before running any real
    inference."""

    def __init__(
        self,
        custom_metadata_map: Optional[dict[str, str]] = None,
        input_shape: tuple[object, ...] = (1, 3, 512, 512),
    ):
        self._modelmeta = _FakeModelMeta(custom_metadata_map or {})
        self._inputs = [_FakeInput(shape=input_shape)]

    def get_modelmeta(self) -> _FakeModelMeta:
        return self._modelmeta

    def get_inputs(self) -> list[_FakeInput]:
        return self._inputs

    def run(self, output_names: Any, input_feed: Any) -> Any:  # pragma: no cover - not exercised here
        raise NotImplementedError("these tests never call infer()")


def _metadata_for(names_by_idx: dict[int, str]) -> dict[str, str]:
    """Build a custom_metadata_map like the one Ultralytics actually writes:
    a 'names' entry holding the Python repr of an {index: name} dict (see
    the real 'names' value read off models/argus_cls.onnx)."""
    return {"names": repr(names_by_idx)}


def _class_config(model_path: Path, class_names: tuple[str, ...] = ()) -> DetectorConfig:
    return DetectorConfig(
        kind="classification",
        model_path=str(model_path),
        input_size=512,
        providers=("CPUExecutionProvider",),
        default_threshold=0.50,
        class_thresholds={},
        severity={},
        class_names=class_names,
    )


def _fake_model_path(tmp_path: Path) -> Path:
    """A path that merely needs to exist (ClassifierDetector only checks
    is_file() before handing it to the -- monkeypatched -- session
    constructor); its content is never actually parsed as ONNX."""
    path = tmp_path / "model.onnx"
    path.write_bytes(b"")
    return path


# --------------------------------------------------------------------------
# class_names_from_onnx_metadata
# --------------------------------------------------------------------------


def test_class_names_from_onnx_metadata_parses_ultralytics_style_dict():
    names_by_idx = {0: "cracking", 1: "layer_shifting", 2: "normal", 3: "spaghetti", 4: "stringing", 5: "warping"}
    session = _FakeSession(custom_metadata_map=_metadata_for(names_by_idx))
    result = class_names_from_onnx_metadata(session)
    assert result == REAL_MODEL_CLASS_ORDER


def test_class_names_from_onnx_metadata_none_when_names_entry_absent():
    session = _FakeSession(custom_metadata_map={"description": "some model"})
    assert class_names_from_onnx_metadata(session) is None


def test_class_names_from_onnx_metadata_none_when_metadata_map_empty():
    session = _FakeSession(custom_metadata_map={})
    assert class_names_from_onnx_metadata(session) is None


def test_class_names_from_onnx_metadata_none_on_unparseable_value():
    session = _FakeSession(custom_metadata_map={"names": "not a python literal {{{"})
    assert class_names_from_onnx_metadata(session) is None


def test_class_names_from_onnx_metadata_none_on_non_dict_value():
    session = _FakeSession(custom_metadata_map={"names": "['cracking', 'normal']"})
    assert class_names_from_onnx_metadata(session) is None


def test_class_names_from_onnx_metadata_none_on_sparse_index_range():
    # Keys 0 and 2 but no 1 -- not a trustworthy dense 0..N-1 mapping.
    session = _FakeSession(custom_metadata_map={"names": "{0: 'a', 2: 'b'}"})
    assert class_names_from_onnx_metadata(session) is None


# --------------------------------------------------------------------------
# resolve_class_names
# --------------------------------------------------------------------------


def test_resolve_class_names_uses_metadata_when_config_empty():
    session = _FakeSession(custom_metadata_map=_metadata_for(dict(enumerate(REAL_MODEL_CLASS_ORDER))))
    result = resolve_class_names(session, ())
    assert result == REAL_MODEL_CLASS_ORDER


def test_resolve_class_names_mismatch_raises_value_error_naming_both_orders():
    session = _FakeSession(custom_metadata_map=_metadata_for(dict(enumerate(REAL_MODEL_CLASS_ORDER))))
    wrong_order = ("normal", "spaghetti", "cracking", "layer_shifting", "stringing", "warping")
    with pytest.raises(ValueError) as excinfo:
        resolve_class_names(session, wrong_order)
    message = str(excinfo.value)
    # Both the configured (wrong) order and the model's real order must be
    # named explicitly -- the whole point is the operator can tell which is
    # which without having to go dig through metadata themselves.
    assert str(list(wrong_order)) in message
    assert str(list(REAL_MODEL_CLASS_ORDER)) in message


def test_resolve_class_names_falls_back_with_warning_when_metadata_missing(caplog):
    session = _FakeSession(custom_metadata_map={})
    configured = ("normal", "spaghetti", "cracking", "layer_shifting", "stringing", "warping")
    with caplog.at_level("WARNING"):
        result = resolve_class_names(session, configured)
    assert result == configured
    assert any("could not read class-name metadata" in rec.message for rec in caplog.records)


def test_resolve_class_names_matching_order_passes_cleanly(caplog):
    session = _FakeSession(custom_metadata_map=_metadata_for(dict(enumerate(REAL_MODEL_CLASS_ORDER))))
    with caplog.at_level("WARNING"):
        result = resolve_class_names(session, REAL_MODEL_CLASS_ORDER)
    assert result == REAL_MODEL_CLASS_ORDER
    # Agreement is the success path -- no warning should be logged.
    assert not any("could not read class-name metadata" in rec.message for rec in caplog.records)


def test_resolve_class_names_raises_when_both_metadata_and_config_missing():
    session = _FakeSession(custom_metadata_map={})
    with pytest.raises(ValueError):
        resolve_class_names(session, ())


# --------------------------------------------------------------------------
# ClassifierDetector.__init__ -- exercised end to end with a fake session
# (via monkeypatching ort.InferenceSession) so the whole reconciliation path
# is proven wired up correctly, not just the helper functions in isolation.
# --------------------------------------------------------------------------


def test_classifier_detector_uses_metadata_names_when_config_empty(tmp_path, monkeypatch):
    fake_session = _FakeSession(custom_metadata_map=_metadata_for(dict(enumerate(REAL_MODEL_CLASS_ORDER))))
    monkeypatch.setattr(classifier_module.ort, "InferenceSession", lambda *a, **kw: fake_session)

    cfg = _class_config(_fake_model_path(tmp_path), class_names=())
    detector = ClassifierDetector(cfg)
    assert detector._class_names == REAL_MODEL_CLASS_ORDER


def test_classifier_detector_raises_on_class_name_mismatch(tmp_path, monkeypatch):
    fake_session = _FakeSession(custom_metadata_map=_metadata_for(dict(enumerate(REAL_MODEL_CLASS_ORDER))))
    monkeypatch.setattr(classifier_module.ort, "InferenceSession", lambda *a, **kw: fake_session)

    wrong_order = ("normal", "spaghetti", "cracking", "layer_shifting", "stringing", "warping")
    cfg = _class_config(_fake_model_path(tmp_path), class_names=wrong_order)
    with pytest.raises(ValueError) as excinfo:
        ClassifierDetector(cfg)
    message = str(excinfo.value)
    assert str(list(wrong_order)) in message
    assert str(list(REAL_MODEL_CLASS_ORDER)) in message


def test_classifier_detector_falls_back_with_warning_when_metadata_absent(tmp_path, monkeypatch, caplog):
    fake_session = _FakeSession(custom_metadata_map={})
    monkeypatch.setattr(classifier_module.ort, "InferenceSession", lambda *a, **kw: fake_session)

    configured = ("normal", "spaghetti", "cracking", "layer_shifting", "stringing", "warping")
    cfg = _class_config(_fake_model_path(tmp_path), class_names=configured)
    with caplog.at_level("WARNING"):
        detector = ClassifierDetector(cfg)
    assert detector._class_names == configured
    assert any("could not read class-name metadata" in rec.message for rec in caplog.records)


def test_classifier_detector_matching_order_constructs_cleanly(tmp_path, monkeypatch):
    fake_session = _FakeSession(custom_metadata_map=_metadata_for(dict(enumerate(REAL_MODEL_CLASS_ORDER))))
    monkeypatch.setattr(classifier_module.ort, "InferenceSession", lambda *a, **kw: fake_session)

    cfg = _class_config(_fake_model_path(tmp_path), class_names=REAL_MODEL_CLASS_ORDER)
    detector = ClassifierDetector(cfg)
    assert detector._class_names == REAL_MODEL_CLASS_ORDER
