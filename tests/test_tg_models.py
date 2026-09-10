"""Unit tests for training/tg_models.py: the vendored tinygrad ResNet used to
train the print-failure classifier on hardware only tinygrad can reach (an
RTX 5060 Ti; PyTorch has no CUDA build for this Mac).

Everything here runs on whatever ``Device.DEFAULT`` this machine resolves to
(METAL here) -- nothing asserts a specific device, and nothing touches the
network: every model is built with ``pretrained=False``, so
``ResNet.load_from_pretrained`` (the only network-touching path in
tg_models.py) is never exercised. torchvision is imported lazily inside the
tests that need it, and only ever via ``weights=None`` (random init, no
download) -- it is used purely as an independent state-dict-key oracle to
verify against, matching the style of test_classifier_detector.py (plain
pytest, no real model file, no GPU/network dependency).
"""

from __future__ import annotations

import pytest

from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
from tinygrad.nn.state import get_state_dict

from training.tg_models import ResNet34, build_model


# bool _is_batchnorm_param_key(str key)
# Inputs: str key - a dotted parameter name from get_state_dict(model), e.g.
#         "layer2.0.bn1.weight" or "layer3.0.downsample.1.running_var"
# Outputs: bool - True if key names a BatchNorm-owned tensor (weight, bias, running_mean, or
#          running_var on a "bn1"/"bn2"/"bn3" attribute or a "downsample.1" -- the BatchNorm half
#          of a [Conv2d, BatchNorm] downsample pair, see tg_models.BasicBlock/Bottleneck)
# Description: Identifies, by key shape alone, which state-dict entries belong to this module's
#              BatchNorm layers (as opposed to Conv2d/Linear), so the BN-dtype regression test
#              below can check only the tensors tg_models.BatchNorm actually forces to float32.
# Side Effects: None (pure function of its input).
def _is_batchnorm_param_key(key: str) -> bool:
    parts = key.split(".")
    last = parts[-1]
    if last not in ("weight", "bias", "running_mean", "running_var"):
        return False
    owner = parts[-2]
    return owner in ("bn1", "bn2", "bn3") or (owner == "1" and len(parts) >= 3 and parts[-3] == "downsample")


class TestBuildModelKeyParity:
    """The hard contract: tinygrad's get_state_dict(model) must produce
    exactly the same key set as the corresponding torchvision model's
    state_dict(), so a trained checkpoint can be loaded into a real
    torchvision module via plain load_state_dict(strict=True) for ONNX
    export (see tg_models.py's module docstring)."""

    def test_resnet18_key_parity_with_torchvision(self):
        torchvision = pytest.importorskip("torchvision")
        model = build_model("resnet18", num_classes=2, pretrained=False)
        tg_keys = sorted(get_state_dict(model).keys())

        tv_model = torchvision.models.resnet18(weights=None, num_classes=2)
        tv_keys = sorted(tv_model.state_dict().keys())

        assert tg_keys == tv_keys
        assert len(tg_keys) == 122

    def test_resnet34_key_parity_with_torchvision(self):
        torchvision = pytest.importorskip("torchvision")
        model = build_model("resnet34", num_classes=2, pretrained=False)
        tg_keys = sorted(get_state_dict(model).keys())

        tv_model = torchvision.models.resnet34(weights=None, num_classes=2)
        tv_keys = sorted(tv_model.state_dict().keys())

        assert tg_keys == tv_keys
        assert len(tg_keys) == 218

    def test_resnet34_constructor_matches_build_model(self):
        # ResNet34() is the same construction build_model("resnet34", ...)
        # delegates to -- confirm they produce the identical key set (a
        # sanity check that build_model isn't quietly diverging from the
        # public ResNet34 constructor it wraps).
        via_build_model = build_model("resnet34", num_classes=6, pretrained=False)
        via_constructor = ResNet34(num_classes=6)
        assert sorted(get_state_dict(via_build_model).keys()) == sorted(get_state_dict(via_constructor).keys())


class TestForwardShape:
    def test_resnet18_forward_shape_at_small_resolution(self):
        model = build_model("resnet18", num_classes=2, pretrained=False)
        x = Tensor.rand(2, 3, 96, 96)
        y = model(x).realize()
        assert y.shape == (2, 2)

    def test_resnet34_forward_shape_at_small_resolution(self):
        model = build_model("resnet34", num_classes=2, pretrained=False)
        x = Tensor.rand(2, 3, 96, 96)
        y = model(x).realize()
        assert y.shape == (2, 2)


class TestBatchNormFloat32Dtype:
    """Regression test for tg_models.BatchNorm's one deliberate deviation
    from upstream: weight/bias/running_mean/running_var must stay float32
    even when dtypes.default_float is float16 (fp16 training on the RTX
    5060 Ti) -- see that class's docstring for why."""

    def test_bn_params_are_float32_under_fp16_default(self):
        # DEFAULT_FLOAT is a tinygrad ContextVar (dtypes.default_float has no
        # setter); Context(DEFAULT_FLOAT="HALF") is the standard scoped way
        # to flip it, and it's restored automatically on exit.
        with Context(DEFAULT_FLOAT="HALF"):
            assert dtypes.default_float == dtypes.float16
            model = build_model("resnet18", num_classes=2, pretrained=False)

        state = get_state_dict(model)
        bn_keys = [k for k in state if _is_batchnorm_param_key(k)]
        # Sanity check the selector itself actually found the BN tensors
        # (18 BasicBlocks-worth of bn1/bn2 + the stem bn1 + 3 downsample BNs,
        # 4 tensors each) -- if this is 0 the test below would pass vacuously.
        assert len(bn_keys) > 0

        for key in bn_keys:
            assert state[key].dtype == dtypes.float32, f"{key} is {state[key].dtype}, expected float32"

    def test_non_batchnorm_params_follow_default_float_under_fp16(self):
        # Contrast case: conv/fc weights are NOT forced to float32 -- only
        # BatchNorm's own tensors are (see tg_models.BatchNorm docstring).
        # This proves the fp16 default is actually taking effect on the rest
        # of the model, not merely untested.
        with Context(DEFAULT_FLOAT="HALF"):
            model = build_model("resnet18", num_classes=2, pretrained=False)

        state = get_state_dict(model)
        assert state["conv1.weight"].dtype == dtypes.float16
        assert state["layer1.0.conv1.weight"].dtype == dtypes.float16

    def test_bn_params_are_float32_under_default_float32_too(self):
        # With no fp16 override, default_float is already float32 -- confirm
        # tg_models.BatchNorm's explicit dtype doesn't accidentally depend on
        # dtypes.default_float being non-default to "happen" to be right.
        assert dtypes.default_float == dtypes.float32
        model = build_model("resnet18", num_classes=2, pretrained=False)
        state = get_state_dict(model)
        bn_keys = [k for k in state if _is_batchnorm_param_key(k)]
        assert len(bn_keys) > 0
        for key in bn_keys:
            assert state[key].dtype == dtypes.float32


class TestBuildModelValidation:
    def test_unsupported_arch_resnet50_raises_value_error(self):
        # Bottleneck/resnet50 is vendored (ResNet.__init__ needs it for
        # resnet50+) but deliberately not exposed via build_model -- see
        # tg_models.py's module docstring.
        with pytest.raises(ValueError) as excinfo:
            build_model("resnet50", num_classes=2, pretrained=False)
        message = str(excinfo.value)
        assert "resnet50" in message
        assert "resnet18" in message
        assert "resnet34" in message

    def test_bogus_arch_raises_value_error_naming_bad_value_and_options(self):
        with pytest.raises(ValueError) as excinfo:
            build_model("bogus", num_classes=2, pretrained=False)
        message = str(excinfo.value)
        assert "bogus" in message
        assert "resnet18" in message
        assert "resnet34" in message

    def test_build_model_never_silently_defaults_on_bad_arch(self):
        # A bad arch must raise, not fall back to some default architecture.
        with pytest.raises(ValueError):
            build_model("", num_classes=2, pretrained=False)
