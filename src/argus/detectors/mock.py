"""Scripted detector for testing the service and DecisionEngine with no model.

`MockDetector` is driven by a fixed sequence of `p_failure` floats supplied
at construction time. Each call to `infer` consumes the next entry: a
positive value produces a single synthetic CATASTROPHIC `spaghetti`
detection at that confidence; a value <= 0 produces an empty result. This
lets the rest of the pipeline (decision engine, service loop, notifications)
be exercised deterministically end-to-end without onnxruntime or a trained
model file.
"""

from __future__ import annotations

from typing import Sequence

from argus.detectors.base import Detector
from argus.types import Detection, DetectionResult, Frame, Severity

#: Kept in sync with the trained class order in `onnx_yolo.DEFAULT_CLASS_NAMES`
#: (index 1 == "spaghetti").
_SPAGHETTI_CLASS_ID = 1
_SPAGHETTI_CLASS_NAME = "spaghetti"
_MOCK_BBOX: tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0)


class MockDetector(Detector):
    """Detector driven by a scripted sequence of `p_failure` values.

    Once the script is exhausted, behavior is controlled by `cycle`:
    `cycle=False` (default) clamps on the last scripted value forever;
    `cycle=True` wraps back around to the start of the script.
    """

    def __init__(
        self,
        script: Sequence[float],
        cycle: bool = False,
        inference_ms: float = 0.0,
    ) -> None:
        if len(script) == 0:
            raise ValueError("MockDetector script must not be empty")
        self._script: tuple[float, ...] = tuple(script)
        self._cycle = cycle
        self._inference_ms = inference_ms
        self._call_count = 0

    @property
    def call_count(self) -> int:
        """Number of times `infer` has been called so far."""
        return self._call_count

    def infer(self, frame: Frame) -> DetectionResult:
        index = self._call_count
        if self._cycle:
            p_failure = self._script[index % len(self._script)]
        else:
            p_failure = self._script[min(index, len(self._script) - 1)]
        self._call_count += 1

        if p_failure <= 0:
            return DetectionResult.empty(inference_ms=self._inference_ms)

        detection = Detection(
            class_id=_SPAGHETTI_CLASS_ID,
            class_name=_SPAGHETTI_CLASS_NAME,
            confidence=float(p_failure),
            bbox=_MOCK_BBOX,
            severity=Severity.CATASTROPHIC,
        )
        return DetectionResult(detections=(detection,), inference_ms=self._inference_ms)
