"""Tests for argus.quality: the pre-inference frame quality gate.

Uses synthetic numpy images only -- no real camera or files needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from argus.config import QualityConfig
from argus.quality import blur_variance, evaluate_frame, mean_luma
from argus.types import Frame

CFG = QualityConfig()  # defaults: min_luma=20, max_luma=240, min_blur_var=45, max_age=5.0


def _frame(image: np.ndarray, timestamp: float = 1000.0, seq: int = 0) -> Frame:
    return Frame(image=image, timestamp=timestamp, seq=seq)


def _black(size: int = 64) -> np.ndarray:
    return np.zeros((size, size, 3), dtype=np.uint8)


def _white(size: int = 64) -> np.ndarray:
    return np.full((size, size, 3), 255, dtype=np.uint8)


def _flat_grey(size: int = 64, value: int = 128) -> np.ndarray:
    """Uniform mid-grey image: passes luma checks but has zero edge energy,
    so it fails the blur check."""
    return np.full((size, size, 3), value, dtype=np.uint8)


def _sharp_noise(size: int = 64, seed: int = 42) -> np.ndarray:
    """High-frequency random noise: mid luminance, high Laplacian variance."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


# --------------------------------------------------------------------------
# metric helpers
# --------------------------------------------------------------------------


def test_mean_luma_black_is_zero():
    assert mean_luma(_black()) == pytest.approx(0.0)


def test_mean_luma_white_is_255():
    assert mean_luma(_white()) == pytest.approx(255.0)


def test_blur_variance_flat_image_is_zero():
    assert blur_variance(_flat_grey()) == pytest.approx(0.0)


def test_blur_variance_noise_is_high():
    assert blur_variance(_sharp_noise()) > 45.0


# --------------------------------------------------------------------------
# evaluate_frame gate checks
# --------------------------------------------------------------------------


def test_black_image_is_too_dark():
    result = evaluate_frame(_frame(_black()), CFG, now=1000.0)
    assert result.passed is False
    assert result.reason == "too_dark"


def test_white_image_is_too_bright():
    result = evaluate_frame(_frame(_white()), CFG, now=1000.0)
    assert result.passed is False
    assert result.reason == "too_bright"


def test_flat_midtone_image_is_blurred():
    result = evaluate_frame(_frame(_flat_grey()), CFG, now=1000.0)
    assert result.passed is False
    assert result.reason == "blurred"


def test_sharp_noise_image_passes_blur_check():
    result = evaluate_frame(_frame(_sharp_noise()), CFG, now=1000.0)
    assert result.passed is True
    assert result.reason is None


def test_old_timestamp_is_stale():
    frame = _frame(_sharp_noise(), timestamp=1000.0)
    now = 1000.0 + CFG.max_frame_age_s + 1.0
    result = evaluate_frame(frame, CFG, now=now)
    assert result.passed is False
    assert result.reason == "stale"


def test_fresh_frame_within_max_age_is_not_stale():
    frame = _frame(_sharp_noise(), timestamp=1000.0)
    now = 1000.0 + CFG.max_frame_age_s - 0.5
    result = evaluate_frame(frame, CFG, now=now)
    assert result.passed is True
    assert result.reason is None


def test_good_frame_passes():
    result = evaluate_frame(_frame(_sharp_noise(), timestamp=1000.0), CFG, now=1001.0)
    assert result.passed is True
    assert result.reason is None


def test_staleness_checked_before_pixel_metrics():
    # A stale AND too-dark frame should report "stale" -- staleness is
    # checked first and is cheaper (no pixel access required).
    frame = _frame(_black(), timestamp=1000.0)
    now = 1000.0 + CFG.max_frame_age_s + 1.0
    result = evaluate_frame(frame, CFG, now=now)
    assert result.reason == "stale"
