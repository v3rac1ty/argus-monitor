"""Abstract base class for Argus detectors.

Kept intentionally tiny and free of any heavy dependency (onnxruntime, cv2)
so it can be imported anywhere `argus.types` can be imported.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Optional

from argus.types import DetectionResult, Frame


class Detector(ABC):
    """Common interface every detector implementation must satisfy."""

    # DetectionResult infer(Frame frame)
    # Inputs: Frame frame - the captured camera frame to run inference on
    # Outputs: DetectionResult - the detections produced for this frame
    # Description: Abstract method every concrete `Detector` must implement to run inference on
    #              a single frame. The base-class body itself is never meant to execute -- it
    #              only exists to raise if a subclass somehow fails to override it.
    # Side Effects: None (the base body only raises NotImplementedError; concrete
    #               implementations may have their own side effects)
    @abstractmethod
    def infer(self, frame: Frame) -> DetectionResult:
        raise NotImplementedError

    # None close()
    # Inputs: None
    # Outputs: None
    # Description: Releases any underlying resources held by the detector. Default
    #              implementation is a no-op; subclasses that own real resources (e.g. an
    #              onnxruntime InferenceSession) override this to release them.
    # Side Effects: None in this base implementation.
    def close(self) -> None:
        return None

    # Detector __enter__()
    # Inputs: None
    # Outputs: Detector - this same instance, enabling `with SomeDetector(...) as d:` usage
    # Description: Context-manager entry point; simply returns self so the detector can be
    #              used directly inside the `with` block.
    # Side Effects: None
    def __enter__(self) -> Detector:
        return self

    # None __exit__(Optional[type[BaseException]] exc_type, Optional[BaseException] exc_val, Optional[TracebackType] exc_tb)
    # Inputs: Optional[type[BaseException]] exc_type - exception class raised in the `with`
    #                 block, or None if it exited normally
    #         Optional[BaseException] exc_val - the exception instance, or None
    #         Optional[TracebackType] exc_tb - the exception's traceback, or None
    # Outputs: None - returning None (falsy) means any exception raised in the `with` block is
    #          not suppressed and propagates normally
    # Description: Context-manager exit point; delegates resource cleanup to `close()`
    #              regardless of whether the block exited normally or via an exception.
    # Side Effects: Calls `self.close()`, which in concrete subclasses may release resources
    #               such as an ONNX runtime session.
    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()
        return None
