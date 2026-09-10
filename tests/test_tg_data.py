"""Unit tests for training/tg_data.py: the GPU-resident dataset pipeline for
fine-tuning the whole-frame classifier on tinygrad's NV backend.

No GPU and no real dataset required -- everything here builds tiny synthetic
image fixtures under `tmp_path` (via cv2.imwrite) and pure-numpy arrays.
Tensor-touching tests run on whatever `Device.DEFAULT` resolves to on this
machine (METAL here); they assert shapes/dtypes (and, where explicitly
required by the correctness contract this module documents, exact values of
simple backend-independent arithmetic like `/255.0`) rather than exact
pixel values coming out of backend-sensitive ops like crop/interpolate.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tinygrad import Tensor, dtypes

import training.evaluate_classifier as evaluate_classifier_module
from training.tg_data import (
    AugConfig,
    ClassBalancedSampler,
    apply_augmentation,
    discover_class_names,
    list_split_images,
    load_split_to_arrays,
    move_to_device_resident,
    normalize_to_model_input,
    sample_augmentation_params,
)


def _write_jpeg(path: Path, color_bgr: tuple[int, int, int], size: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:] = color_bgr
    ok = cv2.imwrite(str(path), image)
    assert ok


def _build_dataset(root: Path, split: str = "train", size: int = 8) -> None:
    """failure/ gets 2 images, normal/ gets 3 -- enough to check sorted class
    order, deterministic listing, and label alignment."""
    _write_jpeg(root / split / "failure" / "f_001.jpg", (10, 20, 30), size)
    _write_jpeg(root / split / "failure" / "f_002.jpg", (11, 21, 31), size)
    _write_jpeg(root / split / "normal" / "n_001.jpg", (40, 50, 60), size)
    _write_jpeg(root / split / "normal" / "n_002.jpg", (41, 51, 61), size)
    _write_jpeg(root / split / "normal" / "n_003.jpg", (42, 52, 62), size)


# --------------------------------------------------------------------------
# discover_class_names
# --------------------------------------------------------------------------


class TestDiscoverClassNames:
    def test_returns_sorted_names(self, tmp_path: Path) -> None:
        _build_dataset(tmp_path)
        names = discover_class_names(tmp_path, "train")
        assert names == ("failure", "normal")

    def test_order_is_alphabetical_not_insertion(self, tmp_path: Path) -> None:
        # Create "normal" before "failure" on disk -- sorted() must still win.
        (tmp_path / "train" / "normal").mkdir(parents=True)
        (tmp_path / "train" / "failure").mkdir(parents=True)
        names = discover_class_names(tmp_path, "train")
        assert names == ("failure", "normal")

    def test_missing_split_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discover_class_names(tmp_path, "train")

    def test_split_dir_with_no_subdirectories_raises(self, tmp_path: Path) -> None:
        (tmp_path / "train").mkdir(parents=True)
        (tmp_path / "train" / "not_a_class.jpg").write_bytes(b"")
        with pytest.raises(FileNotFoundError):
            discover_class_names(tmp_path, "train")

    def test_default_split_is_train(self, tmp_path: Path) -> None:
        _build_dataset(tmp_path)
        assert discover_class_names(tmp_path) == ("failure", "normal")


# --------------------------------------------------------------------------
# list_split_images -- reused (imported) from training.evaluate_classifier,
# not duplicated. These tests confirm the reuse is real and the contract
# still holds, not re-derive evaluate_classifier's own test coverage.
# --------------------------------------------------------------------------


class TestListSplitImagesReuse:
    def test_is_the_same_function_object_as_evaluate_classifier(self) -> None:
        assert list_split_images is evaluate_classifier_module.list_split_images

    def test_missing_split_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            list_split_images(tmp_path, "train", ("failure", "normal"))

    def test_sorted_order_by_class_then_filename(self, tmp_path: Path) -> None:
        _build_dataset(tmp_path)
        records = list_split_images(tmp_path, "train", ("failure", "normal"))
        assert [p.name for p, _ in records] == ["f_001.jpg", "f_002.jpg", "n_001.jpg", "n_002.jpg", "n_003.jpg"]
        assert [c for _, c in records] == ["failure", "failure", "normal", "normal", "normal"]


# --------------------------------------------------------------------------
# load_split_to_arrays
# --------------------------------------------------------------------------


class TestLoadSplitToArrays:
    def test_shapes_dtypes_and_label_alignment(self, tmp_path: Path) -> None:
        _build_dataset(tmp_path, size=8)
        class_names = discover_class_names(tmp_path, "train")
        images, labels, filenames = load_split_to_arrays(tmp_path, "train", class_names, size=8)

        assert images.shape == (5, 8, 8, 3)
        assert images.dtype == np.uint8
        assert labels.shape == (5,)
        assert labels.dtype == np.int32
        assert len(filenames) == 5

        # First two are "failure" (class index 0), remaining three "normal" (index 1).
        assert list(labels) == [0, 0, 1, 1, 1]

    def test_deterministic_file_ordering(self, tmp_path: Path) -> None:
        _build_dataset(tmp_path, size=8)
        class_names = discover_class_names(tmp_path, "train")
        _, _, filenames = load_split_to_arrays(tmp_path, "train", class_names, size=8)
        assert filenames == ["f_001.jpg", "f_002.jpg", "n_001.jpg", "n_002.jpg", "n_003.jpg"]

    def test_pixel_values_are_bgr_as_cv2_wrote_them(self, tmp_path: Path) -> None:
        _write_jpeg(tmp_path / "train" / "failure" / "a.jpg", (10, 20, 30), size=4)
        _write_jpeg(tmp_path / "train" / "normal" / "b.jpg", (40, 50, 60), size=4)
        class_names = discover_class_names(tmp_path, "train")
        images, _, _ = load_split_to_arrays(tmp_path, "train", class_names, size=4)
        # JPEG is lossy -- allow a small tolerance rather than exact equality.
        np.testing.assert_allclose(images[0].mean(axis=(0, 1)), [10, 20, 30], atol=3)

    def test_wrong_size_raises(self, tmp_path: Path) -> None:
        _build_dataset(tmp_path, size=8)
        class_names = discover_class_names(tmp_path, "train")
        with pytest.raises(ValueError):
            load_split_to_arrays(tmp_path, "train", class_names, size=16)

    def test_missing_split_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_split_to_arrays(tmp_path, "train", ("failure", "normal"), size=8)

    def test_empty_split_raises(self, tmp_path: Path) -> None:
        (tmp_path / "train" / "failure").mkdir(parents=True)
        (tmp_path / "train" / "normal").mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            load_split_to_arrays(tmp_path, "train", ("failure", "normal"), size=8)


# --------------------------------------------------------------------------
# ClassBalancedSampler
# --------------------------------------------------------------------------


class TestClassBalancedSampler:
    def test_every_batch_has_exact_batch_size(self) -> None:
        labels = np.array([0] * 10 + [1] * 100)
        sampler = ClassBalancedSampler(labels, num_classes=2, batch_size=7, seed=0)
        for batch in sampler:
            assert batch.shape == (7,)
            assert batch.dtype == np.int64

    def test_lopsided_counts_converge_to_uniform_per_class_frequency(self) -> None:
        # 10 vs 100 -- heavily imbalanced. Draw many batches and check the
        # empirical per-class draw frequency approaches 1/num_classes = 0.5.
        labels = np.array([0] * 10 + [1] * 100)
        sampler = ClassBalancedSampler(labels, num_classes=2, batch_size=8, seed=0)
        counts = {0: 0, 1: 0}
        total = 0
        for _ in range(20):  # several "epochs" worth of draws
            for batch in sampler:
                for idx in batch:
                    counts[int(labels[idx])] += 1
                    total += 1
        assert total > 0
        assert counts[0] / total == pytest.approx(0.5, abs=0.02)
        assert counts[1] / total == pytest.approx(0.5, abs=0.02)

    def test_three_classes_converge_to_one_third_each(self) -> None:
        labels = np.array([0] * 5 + [1] * 50 + [2] * 500)
        sampler = ClassBalancedSampler(labels, num_classes=3, batch_size=9, seed=1)
        counts = {0: 0, 1: 0, 2: 0}
        total = 0
        for _ in range(30):
            for batch in sampler:
                for idx in batch:
                    counts[int(labels[idx])] += 1
                    total += 1
        for c in range(3):
            assert counts[c] / total == pytest.approx(1.0 / 3.0, abs=0.02)

    def test_identical_seed_gives_identical_index_sequence(self) -> None:
        labels = np.array([0] * 10 + [1] * 100)
        sampler_a = ClassBalancedSampler(labels, num_classes=2, batch_size=8, seed=42)
        sampler_b = ClassBalancedSampler(labels, num_classes=2, batch_size=8, seed=42)

        batches_a = [batch.copy() for _, batch in zip(range(5), sampler_a)]
        batches_b = [batch.copy() for _, batch in zip(range(5), sampler_b)]

        assert len(batches_a) == 5
        for a, b in zip(batches_a, batches_b):
            np.testing.assert_array_equal(a, b)

    def test_different_seed_gives_different_index_sequence(self) -> None:
        labels = np.array([0] * 10 + [1] * 100)
        sampler_a = ClassBalancedSampler(labels, num_classes=2, batch_size=8, seed=1)
        sampler_b = ClassBalancedSampler(labels, num_classes=2, batch_size=8, seed=2)

        batch_a = next(iter(sampler_a))
        batch_b = next(iter(sampler_b))
        assert not np.array_equal(batch_a, batch_b)

    def test_steps_per_epoch_is_dataset_size_over_batch_size(self) -> None:
        labels = np.array([0] * 10 + [1] * 100)
        sampler = ClassBalancedSampler(labels, num_classes=2, batch_size=10, seed=0)
        assert sampler.steps_per_epoch() == 110 // 10

    def test_steps_per_epoch_floors_at_one_for_tiny_datasets(self) -> None:
        labels = np.array([0, 1])
        sampler = ClassBalancedSampler(labels, num_classes=2, batch_size=64, seed=0)
        assert sampler.steps_per_epoch() == 1

    def test_reshuffles_minority_class_pool_on_exhaustion(self) -> None:
        # class 0 has only 2 examples; draw far more than 2 class-0 samples
        # and confirm both original indices keep reappearing (pool reshuffle
        # rather than raising/stalling once the pool is exhausted).
        labels = np.array([0, 0] + [1] * 20)
        sampler = ClassBalancedSampler(labels, num_classes=2, batch_size=2, seed=0)
        seen_class_0 = set()
        for _ in range(10):
            for batch in sampler:
                for idx in batch:
                    if labels[idx] == 0:
                        seen_class_0.add(int(idx))
        assert seen_class_0 == {0, 1}

    def test_raises_on_class_with_zero_examples(self) -> None:
        labels = np.array([0, 0, 0])
        with pytest.raises(ValueError):
            ClassBalancedSampler(labels, num_classes=2, batch_size=2, seed=0)

    def test_raises_on_non_positive_batch_size(self) -> None:
        labels = np.array([0, 1])
        with pytest.raises(ValueError):
            ClassBalancedSampler(labels, num_classes=2, batch_size=0, seed=0)

    def test_raises_on_non_positive_num_classes(self) -> None:
        labels = np.array([0, 1])
        with pytest.raises(ValueError):
            ClassBalancedSampler(labels, num_classes=0, batch_size=2, seed=0)


# --------------------------------------------------------------------------
# AugConfig
# --------------------------------------------------------------------------


class TestAugConfig:
    def test_defaults_construct_without_error(self) -> None:
        cfg = AugConfig()
        assert cfg.crop_scale_range[0] <= cfg.crop_scale_range[1]

    def test_rejects_reversed_range(self) -> None:
        with pytest.raises(ValueError):
            AugConfig(brightness_range=(1.2, 0.8))

    def test_rejects_probability_above_one(self) -> None:
        with pytest.raises(ValueError):
            AugConfig(hflip_prob=1.5)

    def test_rejects_negative_probability(self) -> None:
        with pytest.raises(ValueError):
            AugConfig(cutout_prob=-0.1)


# --------------------------------------------------------------------------
# sample_augmentation_params -- PURE NUMPY, no GPU/tinygrad involved.
# --------------------------------------------------------------------------


class TestSampleAugmentationParams:
    def test_all_values_inside_configured_ranges(self) -> None:
        cfg = AugConfig()
        rng = np.random.default_rng(0)
        params = sample_augmentation_params(rng, batch_size=32, cfg=cfg)

        assert cfg.crop_scale_range[0] <= params.crop_scale <= cfg.crop_scale_range[1]
        assert 0.0 <= params.crop_top_frac <= 1.0
        assert 0.0 <= params.crop_left_frac <= 1.0

        assert np.all(params.brightness >= cfg.brightness_range[0])
        assert np.all(params.brightness <= cfg.brightness_range[1])
        assert np.all(params.contrast >= cfg.contrast_range[0])
        assert np.all(params.contrast <= cfg.contrast_range[1])
        assert np.all(params.saturation >= cfg.saturation_range[0])
        assert np.all(params.saturation <= cfg.saturation_range[1])

        assert np.all(params.cutout_cy_frac >= 0.0) and np.all(params.cutout_cy_frac <= 1.0)
        assert np.all(params.cutout_cx_frac >= 0.0) and np.all(params.cutout_cx_frac <= 1.0)
        assert np.all(params.cutout_half_frac >= cfg.cutout_size_range[0] / 2.0)
        assert np.all(params.cutout_half_frac <= cfg.cutout_size_range[1] / 2.0)

        assert np.all(params.noise_std >= cfg.noise_std_range[0])
        assert np.all(params.noise_std <= cfg.noise_std_range[1])

    def test_boolean_fields_have_correct_shape_and_dtype(self) -> None:
        cfg = AugConfig()
        rng = np.random.default_rng(0)
        params = sample_augmentation_params(rng, batch_size=16, cfg=cfg)
        for arr in (params.hflip, params.cutout_apply, params.blur_apply, params.noise_apply):
            assert arr.shape == (16,)
            assert arr.dtype == np.bool_

    def test_identical_seed_reproduces_identical_params(self) -> None:
        cfg = AugConfig()
        params_a = sample_augmentation_params(np.random.default_rng(123), batch_size=8, cfg=cfg)
        params_b = sample_augmentation_params(np.random.default_rng(123), batch_size=8, cfg=cfg)

        assert params_a.crop_scale == params_b.crop_scale
        assert params_a.crop_top_frac == params_b.crop_top_frac
        assert params_a.crop_left_frac == params_b.crop_left_frac
        np.testing.assert_array_equal(params_a.hflip, params_b.hflip)
        np.testing.assert_array_equal(params_a.brightness, params_b.brightness)
        np.testing.assert_array_equal(params_a.contrast, params_b.contrast)
        np.testing.assert_array_equal(params_a.saturation, params_b.saturation)
        np.testing.assert_array_equal(params_a.cutout_apply, params_b.cutout_apply)
        np.testing.assert_array_equal(params_a.noise_std, params_b.noise_std)

    def test_different_seed_gives_different_params(self) -> None:
        cfg = AugConfig()
        params_a = sample_augmentation_params(np.random.default_rng(1), batch_size=8, cfg=cfg)
        params_b = sample_augmentation_params(np.random.default_rng(2), batch_size=8, cfg=cfg)
        assert not np.array_equal(params_a.brightness, params_b.brightness)


# --------------------------------------------------------------------------
# normalize_to_model_input -- the regression test for the "no ImageNet
# mean/std normalization" correctness contract described in the module
# docstring. Simple, backend-independent arithmetic (a scalar divide), so
# exact-value assertions are appropriate here (unlike apply_augmentation's
# crop/interpolate/blur, whose exact output can be backend-sensitive).
# --------------------------------------------------------------------------


class TestNormalizeToModelInput:
    def test_all_255_maps_to_all_1(self) -> None:
        images = Tensor(np.full((2, 3, 4, 4), 255, dtype=np.uint8))
        out = normalize_to_model_input(images).realize()
        arr = out.numpy()
        np.testing.assert_allclose(arr, 1.0)

    def test_zero_maps_to_zero_no_mean_subtracted(self) -> None:
        # If ImageNet mean/std were (wrongly) applied, an all-zero input
        # would come out negative (mean subtraction) rather than exactly 0.
        images = Tensor(np.zeros((2, 3, 4, 4), dtype=np.uint8))
        out = normalize_to_model_input(images).realize()
        arr = out.numpy()
        np.testing.assert_allclose(arr, 0.0)

    def test_nothing_negative(self) -> None:
        images = Tensor(np.random.randint(0, 256, size=(4, 3, 8, 8)).astype(np.uint8))
        out = normalize_to_model_input(images).realize()
        assert out.numpy().min() >= 0.0

    def test_output_is_float32_and_scaled_by_255(self) -> None:
        raw = np.array([[[[0, 127, 255]]]], dtype=np.uint8)  # shape (1,1,1,3)
        images = Tensor(raw)
        out = normalize_to_model_input(images).realize()
        assert out.dtype == dtypes.float32
        np.testing.assert_allclose(out.numpy().flatten(), np.array([0, 127, 255]) / 255.0, atol=1e-6)

    def test_accepts_already_float_input(self) -> None:
        images = Tensor(np.full((1, 3, 4, 4), 255.0, dtype=np.float32))
        out = normalize_to_model_input(images).realize()
        np.testing.assert_allclose(out.numpy(), 1.0)


# --------------------------------------------------------------------------
# apply_augmentation -- tensor-touching; per the module's testing policy we
# assert output shape/dtype/finiteness here, not exact pixel values (those
# depend on crop/interpolate/blur internals that can vary across tinygrad
# backends -- this suite never assumes Device.DEFAULT == "NV").
# --------------------------------------------------------------------------


class TestApplyAugmentation:
    def _random_batch(self, batch_size: int, size: int) -> Tensor:
        arr = np.random.randint(0, 256, size=(batch_size, size, size, 3)).astype(np.uint8)
        return move_to_device_resident(arr)

    def test_output_shape_and_dtype(self) -> None:
        batch_size, source_size, train_size = 4, 32, 16
        images = self._random_batch(batch_size, source_size)
        cfg = AugConfig()
        params = sample_augmentation_params(np.random.default_rng(0), batch_size, cfg)

        out = apply_augmentation(images, params, train_size).realize()

        assert out.shape == (batch_size, 3, train_size, train_size)
        assert out.dtype == dtypes.float32

    def test_output_shape_when_train_size_upsamples(self) -> None:
        batch_size, source_size, train_size = 2, 16, 64
        images = self._random_batch(batch_size, source_size)
        cfg = AugConfig()
        params = sample_augmentation_params(np.random.default_rng(1), batch_size, cfg)

        out = apply_augmentation(images, params, train_size).realize()
        assert out.shape == (batch_size, 3, train_size, train_size)

    def test_output_is_finite_and_bounded(self) -> None:
        batch_size, source_size, train_size = 4, 32, 16
        images = self._random_batch(batch_size, source_size)
        cfg = AugConfig()
        params = sample_augmentation_params(np.random.default_rng(2), batch_size, cfg)

        out = apply_augmentation(images, params, train_size).realize()
        arr = out.numpy()
        assert np.all(np.isfinite(arr))
        # apply_augmentation clips back to pixel-value range at the end.
        assert arr.min() >= 0.0
        assert arr.max() <= 255.0

    def test_rejects_wrong_input_rank(self) -> None:
        images = Tensor(np.zeros((4, 3, 8, 8), dtype=np.uint8))  # already CHW, wrong contract
        cfg = AugConfig()
        params = sample_augmentation_params(np.random.default_rng(0), 4, cfg)
        with pytest.raises(ValueError):
            apply_augmentation(images, params, 8)

    def test_batch_size_one_works(self) -> None:
        images = self._random_batch(1, 20)
        cfg = AugConfig()
        params = sample_augmentation_params(np.random.default_rng(0), 1, cfg)
        out = apply_augmentation(images, params, 12).realize()
        assert out.shape == (1, 3, 12, 12)


# --------------------------------------------------------------------------
# move_to_device_resident
# --------------------------------------------------------------------------


class TestMoveToDeviceResident:
    def test_returns_realized_tensor_with_same_shape(self) -> None:
        arr = np.random.randint(0, 256, size=(3, 8, 8, 3)).astype(np.uint8)
        tensor = move_to_device_resident(arr)
        assert tensor.shape == (3, 8, 8, 3)
        assert tensor.dtype == dtypes.uint8

    def test_rejects_non_uint8_input(self) -> None:
        arr = np.random.rand(3, 8, 8, 3).astype(np.float32)
        with pytest.raises(ValueError):
            move_to_device_resident(arr)
