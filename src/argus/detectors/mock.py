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

    # None __init__(Sequence[float] script, bool cycle, float inference_ms)
    # Inputs: Sequence[float] script - ordered p_failure values to play back, one per `infer`
    #                 call; must be non-empty
    #         bool cycle - default False. If False, once `script` is exhausted the last value is
    #                 repeated forever; if True, playback wraps back to the start of `script`
    #         float inference_ms - default 0.0. Fixed `inference_ms` value stamped onto every
    #                 returned `DetectionResult`
    # Outputs: None
    # Description: Constructs a `MockDetector` that will replay `script` deterministically
    #              across successive `infer` calls, per the class docstring's `cycle` semantics.
    # Side Effects: Mutates the new instance's state (`_script`, `_cycle`, `_inference_ms`,
    #               `_call_count`); raises `ValueError` if `script` is empty.
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

    # int call_count()
    # Inputs: None
    # Outputs: int - number of times `infer` has been called so far
    # Description: Read-only accessor exposing the detector's internal call counter, primarily
    #              useful for tests asserting how many frames were processed.
    # Side Effects: None
    @property
    def call_count(self) -> int:
        return self._call_count

    # DetectionResult infer(Frame frame)
    # Inputs: Frame frame - the captured camera frame (its contents are ignored -- this
    #                 detector is scripted, not model-driven)
    # Outputs: DetectionResult - an empty result if the scripted value for this call is <= 0,
    #          otherwise a result containing a single synthetic CATASTROPHIC `spaghetti`
    #          `Detection` whose confidence is that scripted value
    # Description: Consumes the next scripted `p_failure` value (per `cycle`'s wraparound vs.
    #              clamp-at-end rule) and turns it into a `DetectionResult`, letting tests drive
    #              the decision engine/service loop deterministically without a real model.
    # Side Effects: Advances (increments) the instance's `_call_count`, consuming one entry
    #               from the script.
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
