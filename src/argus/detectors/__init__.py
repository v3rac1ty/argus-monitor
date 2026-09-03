"""Detector implementations: model-agnostic interface plus concrete backends."""

from __future__ import annotations

from argus.detectors.base import Detector
from argus.detectors.classifier import ClassifierDetector
from argus.detectors.mock import MockDetector
from argus.detectors.onnx_yolo import DEFAULT_CLASS_NAMES, OnnxYoloDetector

__all__ = [
    "Detector",
    "MockDetector",
    "OnnxYoloDetector",
    "DEFAULT_CLASS_NAMES",
    "ClassifierDetector",
]
