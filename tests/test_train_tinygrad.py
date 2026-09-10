"""Unit tests for training/train_tinygrad.py's pure/CLI logic: `parse_args` defaults and
its --batch>64 rejection, `assert_nv_device`, the `cosine_lr_with_warmup` and
`ema_decay_at_step` schedule functions, `macro_f1`, `build_checkpoint_metadata`, and a
regression guard on the ~100 MB single-GPU-allocation ceiling (see the module docstring's
"never allocate a single GPU tensor larger than ~100 MB" rule and the PTE-already-mapped
eGPU crash it exists to prevent).

No GPU and no dataset required -- everything here is plain Python/argparse or tiny
hand-computed examples. Following training/train.py's and training/train_cli tests'
precedent, the actual training loop (train()/main()) and the streamed per-batch/per-chunk
Tensor construction inside it are exercised only by a real run (see the task
instructions), never by this suite.
"""

from __future__ import annotations

import math

import pytest

from training.train_tinygrad import (
    DATASET_IMAGE_SIZE,
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_PATH,
    MAX_BATCH_SIZE,
    assert_nv_device,
    build_checkpoint_metadata,
    cosine_lr_with_warmup,
    ema_decay_at_step,
    macro_f1,
    parse_args,
)


# --------------------------------------------------------------------------
# parse_args
# --------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.data == DEFAULT_DATA_DIR
        assert args.arch == "resnet18"
        assert args.epochs == 40
        assert args.batch == 32
        assert args.imgsz == 320
        assert args.seed == 1337
        assert args.patience == 8
        assert args.backbone_lr == pytest.approx(3e-4)
        assert args.head_lr == pytest.approx(3e-3)
        assert args.weight_decay == pytest.approx(0.01)
        assert args.label_smoothing == pytest.approx(0.1)
        assert args.warmup_epochs == 1
        assert args.ema_decay == pytest.approx(0.999)
        assert args.out == DEFAULT_OUT_PATH
        assert args.pretrained is True

    def test_no_pretrained_flag(self):
        args = parse_args(["--no-pretrained"])
        assert args.pretrained is False

    def test_batch_at_max_is_allowed(self):
        args = parse_args(["--batch", str(MAX_BATCH_SIZE)])
        assert args.batch == MAX_BATCH_SIZE

    def test_batch_above_max_is_rejected(self):
        with pytest.raises(ValueError):
            parse_args(["--batch", str(MAX_BATCH_SIZE + 1)])

    def test_batch_well_above_max_is_rejected(self):
        # The exact scenario the module docstring warns about: 128 previously wedged the
        # eGPU driver hard enough to require physically unplugging the Thunderbolt cable.
        with pytest.raises(ValueError) as excinfo:
            parse_args(["--batch", "128"])
        assert "64" in str(excinfo.value)

    def test_overrides(self):
        args = parse_args(
            [
                "--arch", "resnet34",
                "--epochs", "5",
                "--batch", "16",
                "--imgsz", "224",
                "--seed", "7",
                "--patience", "3",
                "--backbone-lr", "1e-5",
                "--head-lr", "1e-2",
                "--weight-decay", "0.05",
                "--label-smoothing", "0.0",
                "--warmup-epochs", "2",
                "--ema-decay", "0.9",
            ]
        )
        assert args.arch == "resnet34"
        assert args.epochs == 5
        assert args.batch == 16
        assert args.imgsz == 224
        assert args.seed == 7
        assert args.patience == 3
        assert args.backbone_lr == pytest.approx(1e-5)
        assert args.head_lr == pytest.approx(1e-2)
        assert args.weight_decay == pytest.approx(0.05)
        assert args.label_smoothing == pytest.approx(0.0)
        assert args.warmup_epochs == 2
        assert args.ema_decay == pytest.approx(0.9)


# --------------------------------------------------------------------------
# assert_nv_device
# --------------------------------------------------------------------------


class TestAssertNvDevice:
    def test_passes_on_nv(self):
        assert_nv_device("NV")  # must not raise

    def test_raises_and_names_the_device_on_metal(self):
        with pytest.raises(RuntimeError) as excinfo:
            assert_nv_device("METAL")
        message = str(excinfo.value)
        assert "METAL" in message
        assert "NV" in message

    def test_raises_and_names_the_device_on_cpu(self):
        with pytest.raises(RuntimeError) as excinfo:
            assert_nv_device("CPU")
        assert "CPU" in str(excinfo.value)

    def test_error_mentions_dev_nv_env_var(self):
        with pytest.raises(RuntimeError) as excinfo:
            assert_nv_device("METAL")
        assert "DEV=NV" in str(excinfo.value)


# --------------------------------------------------------------------------
# cosine_lr_with_warmup
# --------------------------------------------------------------------------


class TestCosineLrWithWarmup:
    def test_zero_at_step_zero_with_warmup(self):
        lr = cosine_lr_with_warmup(step=0, total_steps=100, warmup_steps=10, peak_lr=1.0)
        assert lr == pytest.approx(0.0)

    def test_peak_at_end_of_warmup(self):
        lr = cosine_lr_with_warmup(step=10, total_steps=100, warmup_steps=10, peak_lr=1.0)
        assert lr == pytest.approx(1.0)

    def test_linear_ramp_during_warmup(self):
        lr = cosine_lr_with_warmup(step=5, total_steps=100, warmup_steps=10, peak_lr=2.0)
        assert lr == pytest.approx(1.0)  # halfway through warmup -> half of peak

    def test_decays_to_near_zero_at_total_steps(self):
        lr = cosine_lr_with_warmup(step=100, total_steps=100, warmup_steps=10, peak_lr=1.0)
        assert lr == pytest.approx(0.0, abs=1e-9)

    def test_monotonically_decreasing_after_warmup(self):
        lrs = [cosine_lr_with_warmup(s, total_steps=100, warmup_steps=10, peak_lr=1.0) for s in range(10, 101)]
        assert all(a >= b - 1e-12 for a, b in zip(lrs, lrs[1:]))

    def test_no_warmup_starts_at_peak(self):
        lr = cosine_lr_with_warmup(step=0, total_steps=100, warmup_steps=0, peak_lr=3.0)
        assert lr == pytest.approx(3.0)

    def test_step_past_total_steps_clamped_to_zero(self):
        lr = cosine_lr_with_warmup(step=1000, total_steps=100, warmup_steps=10, peak_lr=1.0)
        assert lr == pytest.approx(0.0, abs=1e-9)

    def test_matches_hand_computed_midpoint(self):
        # progress = 0.5 through the cosine span -> peak_lr * 0.5 * (1 + cos(pi/2)) = peak_lr * 0.5
        lr = cosine_lr_with_warmup(step=55, total_steps=100, warmup_steps=10, peak_lr=4.0)
        expected = 4.0 * 0.5 * (1.0 + math.cos(math.pi * 0.5))
        assert lr == pytest.approx(expected)

    def test_never_negative_or_above_peak(self):
        for s in range(0, 101, 3):
            lr = cosine_lr_with_warmup(s, total_steps=100, warmup_steps=10, peak_lr=1.0)
            assert -1e-9 <= lr <= 1.0 + 1e-9


# --------------------------------------------------------------------------
# ema_decay_at_step
# --------------------------------------------------------------------------


class TestEmaDecayAtStep:
    def test_matches_formula_at_step_zero(self):
        assert ema_decay_at_step(0, target=0.999) == pytest.approx((1 + 0) / (10 + 0))

    def test_matches_formula_at_a_later_step(self):
        assert ema_decay_at_step(90, target=0.999) == pytest.approx((1 + 90) / (10 + 90))

    def test_monotonically_increasing(self):
        decays = [ema_decay_at_step(s, target=0.999) for s in range(0, 2000, 17)]
        assert all(a <= b + 1e-12 for a, b in zip(decays, decays[1:]))

    def test_bounded_by_target(self):
        for s in (0, 1, 10, 1000, 1_000_000):
            assert ema_decay_at_step(s, target=0.999) <= 0.999

    def test_approaches_but_does_not_reach_target_for_a_short_run(self):
        # This project's real run is ~1900 total steps -- confirm the adaptive schedule is
        # still meaningfully below the fixed target by then (the whole point of the schedule).
        decay = ema_decay_at_step(1900, target=0.999)
        assert decay < 0.999
        assert decay > 0.99

    def test_saturates_at_target_for_a_very_long_run(self):
        decay = ema_decay_at_step(10_000_000, target=0.999)
        assert decay == pytest.approx(0.999, abs=1e-6)

    def test_default_target_is_0999(self):
        assert ema_decay_at_step(0) == pytest.approx(ema_decay_at_step(0, target=0.999))

    def test_custom_target_is_respected(self):
        assert ema_decay_at_step(10_000_000, target=0.5) == pytest.approx(0.5, abs=1e-6)


# --------------------------------------------------------------------------
# macro_f1
# --------------------------------------------------------------------------


class TestMacroF1:
    def test_perfect_predictions_score_one(self):
        y_true = ["failure", "failure", "normal", "normal"]
        y_pred = ["failure", "failure", "normal", "normal"]
        assert macro_f1(y_true, y_pred, ("failure", "normal")) == pytest.approx(1.0)

    def test_all_wrong_scores_zero(self):
        y_true = ["failure", "failure", "normal", "normal"]
        y_pred = ["normal", "normal", "failure", "failure"]
        assert macro_f1(y_true, y_pred, ("failure", "normal")) == pytest.approx(0.0)

    def test_hand_computed_imbalanced_example(self):
        # 3 failure (2 correctly predicted failure, 1 predicted normal), 5 normal (all correct).
        y_true = ["failure", "failure", "failure", "normal", "normal", "normal", "normal", "normal"]
        y_pred = ["failure", "failure", "normal", "normal", "normal", "normal", "normal", "normal"]
        # failure: precision = 2/2 = 1.0, recall = 2/3 -> f1 = 2*1*(2/3)/(1+2/3) = 0.8
        # normal:  precision = 5/6, recall = 5/5 = 1.0 -> f1 = 2*(5/6)*1/((5/6)+1) = 10/11
        failure_f1 = 2 * 1.0 * (2 / 3) / (1.0 + 2 / 3)
        normal_f1 = 2 * (5 / 6) * 1.0 / ((5 / 6) + 1.0)
        expected = (failure_f1 + normal_f1) / 2.0
        assert macro_f1(y_true, y_pred, ("failure", "normal")) == pytest.approx(expected)

    def test_macro_averages_equally_regardless_of_class_support(self):
        # Minority class ("failure", 1 sample) gets equal weight to majority ("normal", 9
        # samples) in a MACRO average -- this is the whole reason macro (not micro/weighted
        # accuracy) is used for early stopping on this imbalanced dataset.
        y_true = ["failure"] + ["normal"] * 9
        y_pred = ["normal"] + ["normal"] * 9  # the one failure sample is missed entirely
        # failure: precision=0 (never predicted) -> f1=0; normal: precision=9/10, recall=1 -> f1=18/19
        normal_f1 = 2 * (9 / 10) * 1.0 / ((9 / 10) + 1.0)
        expected = (0.0 + normal_f1) / 2.0
        assert macro_f1(y_true, y_pred, ("failure", "normal")) == pytest.approx(expected)


# --------------------------------------------------------------------------
# build_checkpoint_metadata
# --------------------------------------------------------------------------


class TestBuildCheckpointMetadata:
    def test_all_values_are_strings(self):
        metadata = build_checkpoint_metadata(("failure", "normal"), "resnet18", 320, 1337)
        for value in metadata.values():
            assert isinstance(value, str)

    def test_expected_keys_present(self):
        metadata = build_checkpoint_metadata(("failure", "normal"), "resnet18", 320, 1337)
        assert set(metadata.keys()) == {"class_names", "arch", "imgsz", "num_classes", "seed"}

    def test_class_names_comma_separated_in_order(self):
        metadata = build_checkpoint_metadata(("failure", "normal"), "resnet18", 320, 1337)
        assert metadata["class_names"] == "failure,normal"

    def test_num_classes_matches_class_names_length(self):
        metadata = build_checkpoint_metadata(("a", "b", "c"), "resnet34", 224, 42)
        assert metadata["num_classes"] == "3"
        assert metadata["class_names"] == "a,b,c"

    def test_arch_imgsz_seed_stringified(self):
        metadata = build_checkpoint_metadata(("failure", "normal"), "resnet18", 320, 1337)
        assert metadata["arch"] == "resnet18"
        assert metadata["imgsz"] == "320"
        assert metadata["seed"] == "1337"


# --------------------------------------------------------------------------
# Single-GPU-allocation ceiling (see module docstring fact 8): the streaming rewrite
# exists specifically because a single resident (whole-split) GPU tensor crashed this
# eGPU's allocator. These are pure-arithmetic regression guards -- no GPU, no tinygrad --
# that fail loudly if MAX_BATCH_SIZE or DATASET_IMAGE_SIZE is ever raised without
# re-checking that the largest tensor this file builds (one training batch, or one
# validation chunk, both uint8 (batch, size, size, 3)) still stays well under ~100 MB.
# --------------------------------------------------------------------------


class TestSingleBatchAllocationStaysUnderCeiling:
    #: The hard rule from the module docstring / task: never allocate a single GPU tensor
    #: larger than ~100 MB (this eGPU's Thunderbolt BAR1 window is only 256 MB).
    _ALLOCATION_CEILING_BYTES = 100 * 1024 * 1024

    def _uint8_batch_bytes(self, batch_size: int) -> int:
        return batch_size * DATASET_IMAGE_SIZE * DATASET_IMAGE_SIZE * 3

    def test_max_batch_uint8_image_tensor_is_well_under_100mb(self):
        worst_case_bytes = self._uint8_batch_bytes(MAX_BATCH_SIZE)
        assert worst_case_bytes < self._ALLOCATION_CEILING_BYTES

    def test_max_batch_uint8_image_tensor_matches_hand_computed_size(self):
        # batch 64 @ 512x512x3 uint8 -- the largest single image Tensor this file will ever
        # build (one training batch, or one validation chunk, both capped by MAX_BATCH_SIZE).
        assert self._uint8_batch_bytes(MAX_BATCH_SIZE) == 64 * 512 * 512 * 3

    def test_default_batch_uint8_image_tensor_is_roughly_25mb(self):
        # The task's own sizing claim: batch 32 @ 512x512x3 uint8 is ~25 MB.
        default_batch_bytes = self._uint8_batch_bytes(32)
        assert default_batch_bytes == pytest.approx(25 * 1024 * 1024, rel=0.05)
