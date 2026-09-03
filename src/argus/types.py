"""Frozen shared data contracts for Argus Monitor.

These types flow between subsystems: camera -> detector -> decision engine ->
moonraker / notify / storage. This module intentionally has no dependency on
onnxruntime, opencv, requests, or yaml so it can be imported by any component
(including lightweight tooling and tests) without pulling in the full runtime
stack. The only non-stdlib dependency is numpy, used purely for typing the
raw frame buffer.

Contract stability: this module is consumed by parallel downstream work.
Do not change field names, types, or semantics without coordinating.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class Severity(str, Enum):
    """Severity classification of a detected print defect."""

    CATASTROPHIC = "catastrophic"
    COSMETIC = "cosmetic"


class PrinterState(str, Enum):
    """Coarse printer state as reported by Moonraker."""

    PRINTING = "printing"
    PAUSED = "paused"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    ERROR = "error"
    STANDBY = "standby"
    UNKNOWN = "unknown"


class DecisionState(str, Enum):
    """State of the temporal DecisionEngine's internal state machine."""

    IDLE = "idle"
    ARMED = "armed"
    WARNING = "warning"
    TRIGGERED = "triggered"
    COOLDOWN = "cooldown"


class Action(str, Enum):
    """Action emitted by the DecisionEngine for a given tick."""

    NONE = "none"
    NOTIFY = "notify"
    PAUSE = "pause"
    CANCEL = "cancel"


class ActionMode(str, Enum):
    """Configured ceiling on what automated action the system is allowed to take."""

    NOTIFY_ONLY = "notify_only"
    PAUSE = "pause"
    CANCEL = "cancel"


@dataclass(frozen=True)
class Detection:
    """A single bounding-box detection from the model.

    bbox coordinates are xyxy, in ORIGINAL frame pixel coordinates (i.e. already
    rescaled from model input size back to the source frame — not letterboxed
    or normalized).
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    severity: Severity


@dataclass(frozen=True)
class DetectionResult:
    """All detections produced for a single frame, plus inference timing."""

    detections: tuple[Detection, ...]
    inference_ms: float

    @classmethod
    def empty(cls, inference_ms: float = 0.0) -> DetectionResult:
        """Build a result with no detections (e.g. inference was skipped)."""
        return cls(detections=(), inference_ms=inference_ms)

    @property
    def catastrophic(self) -> tuple[Detection, ...]:
        """Detections classified as catastrophic severity."""
        return tuple(d for d in self.detections if d.severity is Severity.CATASTROPHIC)

    @property
    def cosmetic(self) -> tuple[Detection, ...]:
        """Detections classified as cosmetic severity."""
        return tuple(d for d in self.detections if d.severity is Severity.COSMETIC)

    @property
    def p_failure(self) -> float:
        """Max confidence among catastrophic detections, or 0.0 if there are none.

        This is the primary signal fed into the DecisionEngine; cosmetic
        detections do not contribute to it.
        """
        catastrophic = self.catastrophic
        if not catastrophic:
            return 0.0
        return max(d.confidence for d in catastrophic)

    @property
    def top(self) -> Optional[Detection]:
        """Highest-confidence catastrophic detection, or None if there isn't one."""
        catastrophic = self.catastrophic
        if not catastrophic:
            return None
        return max(catastrophic, key=lambda d: d.confidence)


@dataclass(frozen=True, eq=False)
class Frame:
    """A single captured camera frame.

    eq=False because the image is a numpy array: arrays are unhashable and
    `==` on them returns an elementwise array rather than a bool, which would
    break the default dataclass-generated __eq__/__hash__.
    """

    image: np.ndarray  # BGR, HxWx3, uint8
    timestamp: float  # unix time the frame was captured
    seq: int  # monotonically increasing capture sequence number


@dataclass(frozen=True)
class PrintState:
    """Printer/print-job state as polled from Moonraker."""

    state: PrinterState
    filename: Optional[str]
    elapsed_s: float
    progress: float  # 0..1
    fetched_at: float  # unix time this snapshot was fetched

    @property
    def is_printing(self) -> bool:
        """True if the printer is actively printing (and thus eligible for detection)."""
        return self.state is PrinterState.PRINTING


@dataclass(frozen=True)
class GateResult:
    """Result of a pre-inference quality/eligibility gate (blur, luma, staleness, etc.)."""

    passed: bool
    reason: Optional[str]

    @classmethod
    def ok(cls) -> GateResult:
        """A passing gate result."""
        return cls(passed=True, reason=None)

    @classmethod
    def blocked(cls, reason: str) -> GateResult:
        """A blocking gate result with a human-readable explanation."""
        return cls(passed=False, reason=reason)


@dataclass(frozen=True)
class Decision:
    """Output of the DecisionEngine for a single tick."""

    action: Action
    state: DecisionState
    score: float  # smoothed EMA score
    p_raw: float  # raw (unsmoothed) p_failure for this tick
    votes: int  # ticks within `window` whose score exceeded the vote threshold
    window: int  # size of the voting window this tick was evaluated against
    consecutive: int  # consecutive ticks currently exceeding the active threshold
    reason: str  # human-readable explanation of the decision
    detections: tuple[Detection, ...]  # detections observed on this tick
    gate: GateResult  # quality gate outcome for the frame behind this tick


@dataclass(frozen=True)
class Event:
    """A loggable/notifiable event: one row in the JSONL event log and the basis
    for the Discord notification payload."""

    timestamp: float
    action: Action
    state: DecisionState
    score: float
    p_raw: float
    votes: int
    reason: str
    class_name: Optional[str]
    confidence: Optional[float]
    frame_path: Optional[str]
    print_filename: Optional[str]
    elapsed_s: Optional[float]
    detections: tuple[Detection, ...]

    def to_dict(self) -> dict:
        """Fully JSON-serializable representation of this event.

        Enums are reduced to their `.value`, detections to a list of plain
        dicts, and bboxes to lists of floats -- safe to pass straight to
        `json.dumps` or a Discord embed payload.
        """
        return {
            "timestamp": self.timestamp,
            "action": self.action.value,
            "state": self.state.value,
            "score": self.score,
            "p_raw": self.p_raw,
            "votes": self.votes,
            "reason": self.reason,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "frame_path": self.frame_path,
            "print_filename": self.print_filename,
            "elapsed_s": self.elapsed_s,
            "detections": [
                {
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "bbox": list(d.bbox),
                    "severity": d.severity.value,
                }
                for d in self.detections
            ],
        }
