"""Configuration loading, validation, and env-var secret substitution.

Frozen dataclasses mirror config.example.yaml section-for-section and are
composed into a top-level `Config`. Use `load_config()` (or `Config.from_yaml`
/ `Config.from_dict`) to obtain a validated instance -- direct dataclass
construction skips validation.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from argus.types import ActionMode, Severity

# Repo root, assuming this file lives at <repo>/src/argus/config.py in a
# source checkout (this project is run in place on the Pi, not installed
# from a built wheel, so config.example.yaml is always a sibling of src/).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config.example.yaml"

_ENV_REF = re.compile(r"^env:([A-Za-z_][A-Za-z0-9_]*)$")

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

_VALID_DETECTOR_LAYOUTS = {"auto", "yolov8", "end2end"}

_VALID_DETECTOR_KINDS = {"detection", "classification"}


class ConfigError(Exception):
    """Raised when a config file is malformed or fails validation."""


# --------------------------------------------------------------------------
# Section dataclasses
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraConfig:
    """Camera capture source and framing."""

    source: str = "0"
    width: int = 1280
    height: int = 720
    fps: int = 5
    reconnect_backoff_s: float = 2.0
    max_reconnect_backoff_s: float = 30.0


@dataclass(frozen=True)
class DetectorConfig:
    """ONNX detector model, preprocessing, and per-class thresholds.

    `kind` selects which `Detector` implementation `build_service` would
    construct: `"detection"` (default) for the ONNX object detector
    (`OnnxYoloDetector`, supporting both the legacy YOLOv8 raw-prediction
    output contract and YOLO26's NMS-free end-to-end contract -- see
    `layout`), or `"classification"` for a whole-frame ONNX classifier
    (`ClassifierDetector`). `class_names` is the ordered class list matching
    the model's output index order; it is required (must be non-empty) when
    `kind == "classification"` since a classifier has no other way to know
    what its output indices mean.
    """

    kind: str = "detection"
    model_path: str = "models/argus.onnx"
    input_size: int = 640
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    nms_iou: float = 0.45
    default_threshold: float = 0.50
    class_thresholds: dict[str, float] = field(default_factory=dict)
    severity: dict[str, Severity] = field(default_factory=dict)
    layout: str = "auto"
    class_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionConfig:
    """Temporal DecisionEngine tuning: smoothing, voting windows, and action thresholds."""

    tick_interval_s: float = 1.0
    warmup_s: float = 180
    ema_alpha: float = 0.35
    window: int = 12
    vote_threshold: float = 0.55
    warn_score: float = 0.60
    warn_votes: int = 4
    pause_score: float = 0.75
    pause_votes: int = 6
    pause_consecutive: int = 3
    cancel_score: float = 0.92
    cancel_votes: int = 9
    cancel_consecutive: int = 2
    cancel_enabled: bool = False
    clear_score: float = 0.40
    clear_ticks: int = 5
    cooldown_s: float = 300
    action_mode: ActionMode = ActionMode.NOTIFY_ONLY


@dataclass(frozen=True)
class QualityConfig:
    """Pre-inference frame quality gate thresholds."""

    min_mean_luma: float = 20.0
    max_mean_luma: float = 240.0
    min_blur_var: float = 45.0
    max_frame_age_s: float = 5.0


@dataclass(frozen=True)
class MoonrakerConfig:
    """Moonraker (Klipper API) connection settings."""

    base_url: str = "http://localhost:7125"
    timeout_s: float = 5.0
    poll_interval_s: float = 1.0
    pause_macro: Optional[str] = None


@dataclass(frozen=True)
class NotifyConfig:
    """Discord webhook notification settings."""

    discord_webhook_url: Optional[str] = None
    attach_frame: bool = True
    min_interval_s: float = 60


@dataclass(frozen=True)
class StorageConfig:
    """Local storage for the JSONL event log and captured frames."""

    event_log: str = "logs/events.jsonl"
    frame_dir: str = "captures"
    max_frames: int = 500
    retention_days: int = 14


@dataclass(frozen=True)
class LoggingConfig:
    """Process-wide logging settings."""

    level: str = "INFO"


@dataclass(frozen=True)
class Config:
    """Top-level, validated Argus Monitor configuration."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    moonraker: MoonrakerConfig = field(default_factory=MoonrakerConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> Config:
        """Load, parse, and validate a YAML config file."""
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"could not read config file '{path}': {exc}") from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"could not parse config file '{path}': {exc}") from exc
        if data is None:
            data = {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        """Build and validate a Config from a plain (already-parsed) dict.

        Applies `env:VAR_NAME` substitution to every string leaf before
        constructing section dataclasses, then runs full validation.
        """
        if not isinstance(d, dict):
            raise ConfigError(f"config root must be a mapping, got {type(d).__name__}")

        resolved = _apply_env_overrides(d)

        config = cls(
            camera=_build_section(resolved.get("camera"), CameraConfig, "camera"),
            detector=_build_detector(resolved.get("detector")),
            decision=_build_decision(resolved.get("decision")),
            quality=_build_section(resolved.get("quality"), QualityConfig, "quality"),
            moonraker=_build_section(resolved.get("moonraker"), MoonrakerConfig, "moonraker"),
            notify=_build_section(resolved.get("notify"), NotifyConfig, "notify"),
            storage=_build_section(resolved.get("storage"), StorageConfig, "storage"),
            logging=_build_section(resolved.get("logging"), LoggingConfig, "logging"),
        )
        _validate(config)
        return config


def load_config(path: Optional[Union[str, Path]] = None) -> Config:
    """Load a validated Config from `path`, or from config.example.yaml if omitted."""
    if path is None:
        path = DEFAULT_CONFIG_PATH
    return Config.from_yaml(path)


# --------------------------------------------------------------------------
# env:VAR_NAME substitution
# --------------------------------------------------------------------------


def _apply_env_overrides(value: Any) -> Any:
    """Recursively replace any string of the exact form 'env:VAR_NAME' with
    os.environ['VAR_NAME'] (or None if that variable is unset)."""
    if isinstance(value, dict):
        return {k: _apply_env_overrides(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_apply_env_overrides(v) for v in value]
    if isinstance(value, str):
        match = _ENV_REF.match(value)
        if match:
            return os.environ.get(match.group(1))
    return value


# --------------------------------------------------------------------------
# Section construction
# --------------------------------------------------------------------------


def _build_section(data: Any, section_cls: type, name: str) -> Any:
    """Construct a simple section dataclass from a dict, rejecting unknown keys."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"config section '{name}' must be a mapping, got {type(data).__name__}")
    known = {f.name for f in fields(section_cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"unknown key(s) in '{name}': {unknown}")
    try:
        return section_cls(**data)
    except TypeError as exc:
        raise ConfigError(f"invalid '{name}' section: {exc}") from exc


def _build_detector(data: Any) -> DetectorConfig:
    """Construct DetectorConfig, converting providers to a tuple and severity
    values from plain strings to Severity enum members."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"config section 'detector' must be a mapping, got {type(data).__name__}")
    data = dict(data)
    known = {f.name for f in fields(DetectorConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"unknown key(s) in 'detector': {unknown}")

    if data.get("providers") is not None:
        data["providers"] = tuple(data["providers"])

    if data.get("class_names") is not None:
        data["class_names"] = tuple(data["class_names"])

    severity_raw = data.get("severity")
    if severity_raw is not None:
        if not isinstance(severity_raw, dict):
            raise ConfigError("detector.severity must be a mapping of class name to severity")
        converted: dict[str, Severity] = {}
        for class_name, value in severity_raw.items():
            try:
                converted[class_name] = Severity(value)
            except ValueError as exc:
                valid = [s.value for s in Severity]
                raise ConfigError(
                    f"detector.severity['{class_name}'] has invalid value {value!r}; must be one of {valid}"
                ) from exc
        data["severity"] = converted

    try:
        return DetectorConfig(**data)
    except TypeError as exc:
        raise ConfigError(f"invalid 'detector' section: {exc}") from exc


def _build_decision(data: Any) -> DecisionConfig:
    """Construct DecisionConfig, converting action_mode from a plain string to
    an ActionMode enum member."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"config section 'decision' must be a mapping, got {type(data).__name__}")
    data = dict(data)
    known = {f.name for f in fields(DecisionConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"unknown key(s) in 'decision': {unknown}")

    if data.get("action_mode") is not None:
        try:
            data["action_mode"] = ActionMode(data["action_mode"])
        except ValueError as exc:
            valid = [m.value for m in ActionMode]
            raise ConfigError(
                f"decision.action_mode has invalid value {data['action_mode']!r}; must be one of {valid}"
            ) from exc

    try:
        return DecisionConfig(**data)
    except TypeError as exc:
        raise ConfigError(f"invalid 'decision' section: {exc}") from exc


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _check_unit_interval(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ConfigError(f"{name} must be in [0, 1], got {value}")


def _validate(config: Config) -> None:
    """Validate cross-field and range invariants. Raises ConfigError on the
    first violation found."""
    det = config.detector
    d = config.decision
    cam = config.camera
    q = config.quality
    mr = config.moonraker
    n = config.notify
    s = config.storage

    # --- detector ---
    if det.kind not in _VALID_DETECTOR_KINDS:
        raise ConfigError(
            f"detector.kind must be one of {sorted(_VALID_DETECTOR_KINDS)}, got {det.kind!r}"
        )
    if det.kind == "classification" and not det.class_names:
        raise ConfigError(
            "detector.class_names is required (and must be non-empty) when "
            "detector.kind == 'classification'"
        )
    if det.layout not in _VALID_DETECTOR_LAYOUTS:
        raise ConfigError(
            f"detector.layout must be one of {sorted(_VALID_DETECTOR_LAYOUTS)}, got {det.layout!r}"
        )
    _check_unit_interval("detector.default_threshold", det.default_threshold)
    _check_unit_interval("detector.nms_iou", det.nms_iou)
    for class_name, threshold in det.class_thresholds.items():
        _check_unit_interval(f"detector.class_thresholds['{class_name}']", threshold)

    threshold_classes = set(det.class_thresholds)
    severity_classes = set(det.severity)
    if threshold_classes != severity_classes:
        problems = []
        missing_severity = sorted(threshold_classes - severity_classes)
        missing_threshold = sorted(severity_classes - threshold_classes)
        if missing_severity:
            problems.append(f"missing 'severity' entry for: {missing_severity}")
        if missing_threshold:
            problems.append(f"missing 'class_thresholds' entry for: {missing_threshold}")
        raise ConfigError(
            "detector.class_thresholds and detector.severity must define exactly the same "
            "class names; " + "; ".join(problems)
        )

    # --- decision: action_mode validity (also enforced at parse time; kept
    # here so a Config built by hand is still checked by _validate) ---
    if not isinstance(d.action_mode, ActionMode):
        try:
            ActionMode(d.action_mode)
        except ValueError as exc:
            valid = [m.value for m in ActionMode]
            raise ConfigError(
                f"decision.action_mode has invalid value {d.action_mode!r}; must be one of {valid}"
            ) from exc

    _check_unit_interval("decision.vote_threshold", d.vote_threshold)
    _check_unit_interval("decision.warn_score", d.warn_score)
    _check_unit_interval("decision.pause_score", d.pause_score)
    _check_unit_interval("decision.cancel_score", d.cancel_score)
    _check_unit_interval("decision.clear_score", d.clear_score)

    if d.window < 1:
        raise ConfigError(f"decision.window must be >= 1, got {d.window}")

    for votes_field in ("warn_votes", "pause_votes", "cancel_votes"):
        votes = getattr(d, votes_field)
        if votes < 0:
            raise ConfigError(f"decision.{votes_field} must be >= 0, got {votes}")
        if votes > d.window:
            raise ConfigError(
                f"decision.{votes_field} ({votes}) must be <= decision.window ({d.window})"
            )

    if not (0 < d.ema_alpha <= 1):
        raise ConfigError(f"decision.ema_alpha must be in (0, 1], got {d.ema_alpha}")

    if d.warmup_s < 0:
        raise ConfigError(f"decision.warmup_s must be >= 0, got {d.warmup_s}")

    if d.pause_score < d.warn_score:
        raise ConfigError(
            f"decision.pause_score ({d.pause_score}) must be >= decision.warn_score ({d.warn_score})"
        )
    if d.cancel_score < d.pause_score:
        raise ConfigError(
            f"decision.cancel_score ({d.cancel_score}) must be >= decision.pause_score ({d.pause_score})"
        )

    if d.tick_interval_s <= 0:
        raise ConfigError(f"decision.tick_interval_s must be > 0, got {d.tick_interval_s}")
    if d.cooldown_s < 0:
        raise ConfigError(f"decision.cooldown_s must be >= 0, got {d.cooldown_s}")
    if d.clear_ticks < 1:
        raise ConfigError(f"decision.clear_ticks must be >= 1, got {d.clear_ticks}")
    if d.pause_consecutive < 1:
        raise ConfigError(f"decision.pause_consecutive must be >= 1, got {d.pause_consecutive}")
    if d.cancel_consecutive < 1:
        raise ConfigError(f"decision.cancel_consecutive must be >= 1, got {d.cancel_consecutive}")

    # --- camera ---
    if cam.width <= 0 or cam.height <= 0:
        raise ConfigError(f"camera.width and camera.height must be > 0, got {cam.width}x{cam.height}")
    if cam.fps <= 0:
        raise ConfigError(f"camera.fps must be > 0, got {cam.fps}")
    if cam.reconnect_backoff_s <= 0:
        raise ConfigError(f"camera.reconnect_backoff_s must be > 0, got {cam.reconnect_backoff_s}")
    if cam.max_reconnect_backoff_s < cam.reconnect_backoff_s:
        raise ConfigError(
            "camera.max_reconnect_backoff_s must be >= camera.reconnect_backoff_s"
        )

    # --- quality ---
    if q.min_mean_luma >= q.max_mean_luma:
        raise ConfigError("quality.min_mean_luma must be < quality.max_mean_luma")
    if q.min_blur_var < 0:
        raise ConfigError(f"quality.min_blur_var must be >= 0, got {q.min_blur_var}")
    if q.max_frame_age_s <= 0:
        raise ConfigError(f"quality.max_frame_age_s must be > 0, got {q.max_frame_age_s}")

    # --- moonraker ---
    if mr.timeout_s <= 0:
        raise ConfigError(f"moonraker.timeout_s must be > 0, got {mr.timeout_s}")
    if mr.poll_interval_s <= 0:
        raise ConfigError(f"moonraker.poll_interval_s must be > 0, got {mr.poll_interval_s}")

    # --- notify ---
    if n.min_interval_s < 0:
        raise ConfigError(f"notify.min_interval_s must be >= 0, got {n.min_interval_s}")

    # --- storage ---
    if s.max_frames < 0:
        raise ConfigError(f"storage.max_frames must be >= 0, got {s.max_frames}")
    if s.retention_days < 0:
        raise ConfigError(f"storage.retention_days must be >= 0, got {s.retention_days}")

    # --- logging ---
    if config.logging.level.upper() not in _VALID_LOG_LEVELS:
        raise ConfigError(
            f"logging.level must be one of {sorted(_VALID_LOG_LEVELS)}, got {config.logging.level!r}"
        )
