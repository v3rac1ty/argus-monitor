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


# np.ndarray _to_grey(np.ndarray image)
# Inputs: np.ndarray image - BGR uint8 image, or already single-channel greyscale
# Outputs: np.ndarray - single-channel greyscale image
# Description: Converts a BGR image to greyscale via cv2, passing single-channel input through
#              unchanged.
# Side Effects: None
def _to_grey(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# float mean_luma(np.ndarray image)
# Inputs: np.ndarray image - BGR uint8 image, or already single-channel greyscale
# Outputs: float - mean luminance in 0..255
# Description: Converts to greyscale (if needed) and averages pixel intensity.
# Side Effects: None
def mean_luma(image: np.ndarray) -> float:
    return float(_to_grey(image).mean())


# float blur_variance(np.ndarray image)
# Inputs: np.ndarray image - BGR uint8 image, or already single-channel greyscale
# Outputs: float - variance of the Laplacian; low means smooth/blurred, high means sharp/in-focus
# Description: Converts to greyscale (if needed), applies a Laplacian edge filter via cv2, and
#              returns the variance of the result as a focus/sharpness measure.
# Side Effects: None
def blur_variance(image: np.ndarray) -> float:
    grey = _to_grey(image)
    laplacian = cv2.Laplacian(grey, cv2.CV_64F)
    return float(laplacian.var())


# GateResult evaluate_frame(Frame frame, QualityConfig cfg, float now)
# Inputs: Frame frame - the captured frame to evaluate
#         QualityConfig cfg - staleness/luminance/blur thresholds
#         float now - current unix time, used to compute frame staleness
# Outputs: GateResult - GateResult.ok() if the frame passes every check, else
#                       GateResult.blocked(reason) for the first check that fails
#                       ("stale", "too_dark", "too_bright", or "blurred")
# Description: Runs the pre-inference quality gate in cheapest-first order: staleness, then
#              luminance, then blur (converting to greyscale once and reusing it for both pixel
#              checks), short-circuiting on the first failure.
# Side Effects: None
def evaluate_frame(frame: Frame, cfg: QualityConfig, now: float) -> GateResult:
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
