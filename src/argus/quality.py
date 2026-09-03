"""Pre-inference frame quality gate.

This exists purely to reduce false positives: a dark, blown-out, blurred, or
stale frame produces garbage model scores, so we refuse to score it at all
rather than feed noise into the DecisionEngine. `evaluate_frame` runs the
staleness, luminance, and blur checks (converting to greyscale exactly once
and reusing it), returning the first failing `GateResult.blocked(reason)` or
`GateResult.ok()` if the frame passes everything.
"""

from __future__ import annotations

import cv2
import numpy as np

from argus.config import QualityConfig
from argus.types import Frame, GateResult


def _to_grey(image: np.ndarray) -> np.ndarray:
    """Convert a BGR uint8 image to single-channel greyscale (a no-op if
    `image` is already single-channel)."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def mean_luma(image: np.ndarray) -> float:
    """Mean luminance of a BGR (or already-greyscale) image, in 0..255."""
    return float(_to_grey(image).mean())


def blur_variance(image: np.ndarray) -> float:
    """Variance of the Laplacian of a BGR (or already-greyscale) image.

    Low variance indicates a smooth/blurred image (little edge energy);
    high variance indicates a sharp, in-focus image.
    """
    grey = _to_grey(image)
    laplacian = cv2.Laplacian(grey, cv2.CV_64F)
    return float(laplacian.var())


def evaluate_frame(frame: Frame, cfg: QualityConfig, now: float) -> GateResult:
    """Run all pre-inference quality checks against `frame`.

    Order: staleness first (cheapest -- no pixels touched), then luminance,
    then blur, converting to greyscale once and reusing it for both pixel
    checks.
    """
    age_s = now - frame.timestamp
    if age_s > cfg.max_frame_age_s:
        return GateResult.blocked("stale")

    grey = _to_grey(frame.image)

    luma = float(grey.mean())
    if luma < cfg.min_mean_luma:
        return GateResult.blocked("too_dark")
    if luma > cfg.max_mean_luma:
        return GateResult.blocked("too_bright")

    blur = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    if blur < cfg.min_blur_var:
        return GateResult.blocked("blurred")

    return GateResult.ok()
