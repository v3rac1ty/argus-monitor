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

    @abstractmethod
    def infer(self, frame: Frame) -> DetectionResult:
        """Run inference on a single frame and return its detections."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any underlying resources. Default is a no-op."""
        return None

    def __enter__(self) -> Detector:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()
        return None
