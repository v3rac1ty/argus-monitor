"""Tests for training/export_tinygrad_onnx.py: the bridge that turns a
tinygrad-trained binary (failure/normal) safetensors checkpoint into an ONNX
file for a Raspberry Pi running ONNX Runtime only.

Most of this module's logic is pure (`parse_class_names`,
`format_class_names_metadata`, `max_abs_diff`, `arrays_agree`, `parse_args`)
and is tested with no model file at all. The single most load-bearing test
is `TestFormatClassNamesMetadataRoundTrip`: it feeds
`format_class_names_metadata`'s output through the REAL production parser
(`argus.detectors.classifier.class_names_from_onnx_metadata`) via a fake
onnxruntime session shaped like `test_classifier_detector.py`'s
`_FakeSession`, proving the two halves of the metadata contract actually
agree with each other rather than just looking plausible independently.

One true end-to-end integration test (`TestEndToEndExportSmoke`) builds a
tiny random-weight resnet18 with `training.tg_models.build_model`, saves it
as a safetensors checkpoint with the pinned metadata contract, and runs the
whole `main()` pipeline (torch export + tinygrad/ONNX-Runtime parity check)
against it -- exactly the "smoke test without a trained model" the module
docstring calls for, since parity between tinygrad and ONNX Runtime is
exactly as meaningful on random weights as on trained ones. It is skipped
if torch/torchvision/onnx aren't importable.

No network access anywhere in this file.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from export_tinygrad_onnx import (  # noqa: E402
    DEFAULT_ATOL,
    DEFAULT_OPSET,
    DEFAULT_OUT_PATH,
    DEFAULT_SAMPLES_PER_CLASS,
    DEFAULT_TEST_DATA_DIR,
    arrays_agree,
    build_torch_resnet,
    format_class_names_metadata,
    load_checkpoint,
    main,
    max_abs_diff,
    parse_args,
    parse_class_names,
)

CLASS_NAMES = ("failure", "normal")


# --------------------------------------------------------------------------
# parse_args
# --------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self):
        args = parse_args(["--checkpoint", "runs/train_tg/bin_v1/best.safetensors"])
        assert args.checkpoint == Path("runs/train_tg/bin_v1/best.safetensors")
        assert args.out == DEFAULT_OUT_PATH
        assert args.opset == DEFAULT_OPSET == 12
        assert args.test_data == DEFAULT_TEST_DATA_DIR
        assert args.samples_per_class == DEFAULT_SAMPLES_PER_CLASS == 5
        assert args.atol == pytest.approx(DEFAULT_ATOL)
        assert args.atol == pytest.approx(2e-2)

    def test_checkpoint_is_required(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_overrides(self):
        args = parse_args(
            [
                "--checkpoint",
                "best.safetensors",
                "--out",
                "models/custom.onnx",
                "--opset",
                "17",
                "--test-data",
                "datasets/custom/test",
                "--samples-per-class",
                "10",
                "--atol",
                "0.1",
            ]
        )
        assert args.out == Path("models/custom.onnx")
        assert args.opset == 17
        assert args.test_data == Path("datasets/custom/test")
        assert args.samples_per_class == 10
        assert args.atol == pytest.approx(0.1)


# --------------------------------------------------------------------------
# parse_class_names
# --------------------------------------------------------------------------


class TestParseClassNames:
    def test_parses_comma_separated_names_in_order(self):
        result = parse_class_names({"class_names": "failure,normal"})
        assert result == ("failure", "normal")

    def test_strips_surrounding_whitespace(self):
        result = parse_class_names({"class_names": " failure , normal "})
        assert result == ("failure", "normal")

    def test_missing_key_raises_value_error(self):
        with pytest.raises(ValueError, match="class_names"):
            parse_class_names({"arch": "resnet18"})

    def test_empty_value_raises_value_error(self):
        with pytest.raises(ValueError, match="class_names"):
            parse_class_names({"class_names": ""})

    def test_empty_segment_after_split_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_class_names({"class_names": "failure,,normal"})

    def test_single_class_name(self):
        assert parse_class_names({"class_names": "only_one"}) == ("only_one",)


# --------------------------------------------------------------------------
# format_class_names_metadata -- round-trips through the REAL production
# parser in argus.detectors.classifier.
# --------------------------------------------------------------------------


class _FakeModelMeta:
    def __init__(self, custom_metadata_map: dict[str, str]):
        self.custom_metadata_map = custom_metadata_map


class _FakeSession:
    """Minimal stand-in for onnxruntime.InferenceSession exposing only
    get_modelmeta(), matching the shape of test_classifier_detector.py's
    _FakeSession -- the real class_names_from_onnx_metadata only ever
    touches session.get_modelmeta().custom_metadata_map."""

    def __init__(self, custom_metadata_map: Optional[dict[str, str]] = None):
        self._modelmeta = _FakeModelMeta(custom_metadata_map or {})

    def get_modelmeta(self) -> _FakeModelMeta:
        return self._modelmeta


class TestFormatClassNamesMetadataRoundTrip:
    def test_roundtrips_through_real_class_names_from_onnx_metadata_parser(self):
        from argus.detectors.classifier import class_names_from_onnx_metadata

        value = format_class_names_metadata(CLASS_NAMES)
        session = _FakeSession({"names": value})

        result = class_names_from_onnx_metadata(session)
        assert result == CLASS_NAMES

    def test_roundtrips_for_more_than_two_classes(self):
        from argus.detectors.classifier import class_names_from_onnx_metadata

        names = ("cracking", "layer_shifting", "normal", "spaghetti", "stringing", "warping")
        value = format_class_names_metadata(names)
        session = _FakeSession({"names": value})

        assert class_names_from_onnx_metadata(session) == names

    def test_output_is_index_zero_keyed_dict_repr(self):
        value = format_class_names_metadata(("failure", "normal"))
        assert value == "{0: 'failure', 1: 'normal'}"


# --------------------------------------------------------------------------
# max_abs_diff / arrays_agree
# --------------------------------------------------------------------------


class TestMaxAbsDiff:
    def test_zero_for_identical_arrays(self):
        a = np.array([1.0, 2.0, 3.0])
        assert max_abs_diff(a, a.copy()) == pytest.approx(0.0)

    def test_finds_largest_elementwise_difference(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.01, 0.5, -0.2])
        assert max_abs_diff(a, b) == pytest.approx(0.5)

    def test_handles_negative_differences(self):
        a = np.array([5.0])
        b = np.array([5.3])
        assert max_abs_diff(a, b) == pytest.approx(0.3, abs=1e-9)

    def test_works_on_2d_arrays(self):
        a = np.zeros((1, 2))
        b = np.array([[0.0, 0.07]])
        assert max_abs_diff(a, b) == pytest.approx(0.07)


class TestArraysAgree:
    def test_agrees_when_diff_within_atol(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.005, 2.0])
        agree, diff = arrays_agree(a, b, atol=0.01)
        assert agree is True
        assert diff == pytest.approx(0.005)

    def test_disagrees_when_diff_exceeds_atol(self):
        a = np.array([1.0])
        b = np.array([1.5])
        agree, diff = arrays_agree(a, b, atol=0.01)
        assert agree is False
        assert diff == pytest.approx(0.5)

    def test_boundary_diff_exactly_equal_to_atol_agrees(self):
        a = np.array([0.0])
        b = np.array([0.02])
        agree, diff = arrays_agree(a, b, atol=0.02)
        assert agree is True
        assert diff == pytest.approx(0.02)

    def test_boundary_diff_just_over_atol_disagrees(self):
        a = np.array([0.0])
        b = np.array([0.020001])
        agree, diff = arrays_agree(a, b, atol=0.02)
        assert agree is False


# --------------------------------------------------------------------------
# load_checkpoint -- missing/incomplete metadata must fail loudly and
# clearly, not with a bare KeyError deep inside export logic.
# --------------------------------------------------------------------------


def _save_checkpoint(path: Path, metadata: dict[str, str], num_classes: int = 2) -> None:
    """Writes a real, tiny tinygrad safetensors checkpoint (a random-weight
    resnet18) with the given metadata -- used to exercise load_checkpoint's
    metadata validation against an actual checkpoint file, not a hand-rolled
    fake."""
    from tinygrad.nn.state import get_state_dict, safe_save

    from training.tg_models import build_model

    model = build_model("resnet18", num_classes=num_classes, pretrained=False)
    safe_save(get_state_dict(model), str(path), metadata=metadata)


class TestLoadCheckpointMetadataValidation:
    def test_missing_checkpoint_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "does_not_exist.safetensors")

    def test_valid_checkpoint_round_trips_tensors_and_metadata(self, tmp_path):
        path = tmp_path / "model.safetensors"
        metadata = {
            "class_names": "failure,normal",
            "arch": "resnet18",
            "imgsz": "320",
            "num_classes": "2",
            "seed": "1337",
        }
        _save_checkpoint(path, metadata)

        tensors, loaded_metadata = load_checkpoint(path)
        assert loaded_metadata == metadata
        assert "fc.weight" in tensors
        assert "fc.bias" in tensors
        assert len(tensors) == 122  # resnet18 key-parity count, see test_tg_models.py

    @pytest.mark.parametrize("missing_key", ["class_names", "arch", "imgsz", "num_classes"])
    def test_missing_required_key_raises_clear_value_error(self, tmp_path, missing_key):
        metadata = {
            "class_names": "failure,normal",
            "arch": "resnet18",
            "imgsz": "320",
            "num_classes": "2",
            "seed": "1337",
        }
        del metadata[missing_key]
        path = tmp_path / "model.safetensors"
        _save_checkpoint(path, metadata)

        with pytest.raises(ValueError, match=missing_key):
            load_checkpoint(path)

    def test_empty_required_value_raises_clear_value_error(self, tmp_path):
        metadata = {
            "class_names": "failure,normal",
            "arch": "",
            "imgsz": "320",
            "num_classes": "2",
        }
        path = tmp_path / "model.safetensors"
        _save_checkpoint(path, metadata)

        with pytest.raises(ValueError, match="arch"):
            load_checkpoint(path)

    def test_checkpoint_with_no_metadata_at_all_raises_value_error(self, tmp_path):
        from tinygrad.nn.state import get_state_dict, safe_save

        from training.tg_models import build_model

        model = build_model("resnet18", num_classes=2, pretrained=False)
        path = tmp_path / "model.safetensors"
        safe_save(get_state_dict(model), str(path), metadata=None)

        with pytest.raises(ValueError):
            load_checkpoint(path)


# --------------------------------------------------------------------------
# build_torch_resnet
# --------------------------------------------------------------------------


class TestBuildTorchResnet:
    def test_unknown_arch_raises_value_error(self):
        pytest.importorskip("torch")
        pytest.importorskip("torchvision")
        with pytest.raises(ValueError, match="resnet50"):
            build_torch_resnet({}, "resnet50", num_classes=2)

    def test_loads_tinygrad_state_dict_into_torchvision_module_strictly(self):
        pytest.importorskip("torch")
        torchvision = pytest.importorskip("torchvision")
        from tinygrad.nn.state import get_state_dict

        from training.tg_models import build_model

        tg_model = build_model("resnet18", num_classes=2, pretrained=False)
        state_dict = get_state_dict(tg_model)

        torch_model = build_torch_resnet(state_dict, "resnet18", num_classes=2)

        assert isinstance(torch_model, torchvision.models.resnet.ResNet)
        assert torch_model.training is False  # .eval() was called

        # Spot-check one real weight value survived the tinygrad -> numpy ->
        # torch round trip unchanged.
        expected = state_dict["conv1.weight"].numpy()
        actual = torch_model.conv1.weight.detach().numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6)


# --------------------------------------------------------------------------
# End-to-end smoke test: tinygrad checkpoint -> ONNX -> parity, on random
# weights (no trained model needed -- see module docstring).
# --------------------------------------------------------------------------


class TestEndToEndExportSmoke:
    def test_random_weight_resnet18_exports_and_parity_agrees(self, tmp_path, capsys):
        pytest.importorskip("torch")
        pytest.importorskip("torchvision")
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

        from tinygrad.nn.state import get_state_dict, safe_save

        from training.tg_models import build_model

        arch = "resnet18"
        num_classes = 2
        imgsz = 64
        class_names = ("failure", "normal")

        tg_model = build_model(arch, num_classes=num_classes, pretrained=False)
        checkpoint_path = tmp_path / "model.safetensors"
        safe_save(
            get_state_dict(tg_model),
            str(checkpoint_path),
            metadata={
                "class_names": ",".join(class_names),
                "arch": arch,
                "imgsz": str(imgsz),
                "num_classes": str(num_classes),
                "seed": "1337",
            },
        )

        out_path = tmp_path / "argus_bin.onnx"
        main(
            [
                "--checkpoint",
                str(checkpoint_path),
                "--out",
                str(out_path),
                # No test images on disk at this path -- exercises the
                # synthetic-input fallback described in the module docstring.
                "--test-data",
                str(tmp_path / "no_such_test_dir"),
                "--samples-per-class",
                "4",
                "--atol",
                "0.02",
            ]
        )

        assert out_path.is_file()
        assert out_path.stat().st_size > 0

        captured = capsys.readouterr()
        assert "Argmax agreement: 100.0%" in captured.out

        # The exported ONNX model must carry the real class-name metadata
        # ClassifierDetector reads at deploy time, in the right order.
        import onnxruntime as ort

        from argus.detectors.classifier import class_names_from_onnx_metadata

        session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        assert class_names_from_onnx_metadata(session) == class_names

        input_meta = session.get_inputs()[0]
        assert tuple(input_meta.shape) == (1, 3, imgsz, imgsz)
        output_meta = session.get_outputs()[0]
        assert tuple(output_meta.shape) == (1, num_classes)

    def test_argmax_disagreement_raises_assertion_error(self, tmp_path, monkeypatch):
        """If the onnx/tinygrad backends ever disagreed on a prediction,
        main() must fail loudly rather than silently shipping a mismatched
        model -- verified by forcing verify_parity to report < 100%
        agreement."""
        pytest.importorskip("torch")
        pytest.importorskip("torchvision")
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

        from tinygrad.nn.state import get_state_dict, safe_save

        from training.tg_models import build_model

        import export_tinygrad_onnx as export_module

        arch = "resnet18"
        num_classes = 2
        imgsz = 64
        class_names = ("failure", "normal")

        tg_model = build_model(arch, num_classes=num_classes, pretrained=False)
        checkpoint_path = tmp_path / "model.safetensors"
        safe_save(
            get_state_dict(tg_model),
            str(checkpoint_path),
            metadata={
                "class_names": ",".join(class_names),
                "arch": arch,
                "imgsz": str(imgsz),
                "num_classes": str(num_classes),
                "seed": "1337",
            },
        )

        def fake_verify_parity(*args: Any, **kwargs: Any) -> dict:
            return {
                "num_samples": 4,
                "used_synthetic_fallback": True,
                "max_abs_diff": 0.0,
                "atol": 0.02,
                "within_atol": True,
                "argmax_agreement": 0.75,
            }

        monkeypatch.setattr(export_module, "verify_parity", fake_verify_parity)

        with pytest.raises(AssertionError, match="disagree"):
            export_module.main(
                [
                    "--checkpoint",
                    str(checkpoint_path),
                    "--out",
                    str(tmp_path / "argus_bin.onnx"),
                ]
            )
