"""Tests for argus.config: loading, defaults, env substitution, and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from argus.config import Config, ConfigError, DEFAULT_CONFIG_PATH, load_config
from argus.types import ActionMode, Severity

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG_PATH = REPO_ROOT / "config.example.yaml"


def _base_dict() -> dict:
    """A fresh, mutable copy of the parsed example YAML, for tests that tweak
    a single field to exercise validation."""
    with open(EXAMPLE_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure env-substituted vars used by tests start unset, regardless of
    what the host environment happens to have."""
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("SOME_VAR", raising=False)
    yield


# --------------------------------------------------------------------------
# Loading & defaults
# --------------------------------------------------------------------------


def test_example_yaml_loads():
    config = Config.from_yaml(EXAMPLE_CONFIG_PATH)
    assert isinstance(config, Config)


def test_default_config_path_is_the_example_file():
    assert DEFAULT_CONFIG_PATH == EXAMPLE_CONFIG_PATH
    assert DEFAULT_CONFIG_PATH.exists()


def test_load_config_with_no_path_falls_back_to_example():
    config = load_config()
    assert config.camera.width == 1280
    assert config.camera.height == 720


def test_camera_defaults():
    config = load_config()
    c = config.camera
    assert c.source == "0"
    assert c.width == 1280
    assert c.height == 720
    assert c.fps == 5
    assert c.reconnect_backoff_s == 2.0
    assert c.max_reconnect_backoff_s == 30.0


def test_detector_defaults():
    config = load_config()
    det = config.detector
    assert det.kind == "detection"
    assert det.model_path == "models/argus.onnx"
    assert det.input_size == 512
    assert det.providers == ("CPUExecutionProvider",)
    assert det.layout == "auto"
    assert det.class_names == ()
    assert det.nms_iou == 0.45
    assert det.default_threshold == 0.50
    assert det.class_thresholds == {
        "spaghetti": 0.75,
        "warping": 0.76,
        "stringing": 0.30,
        "zits": 0.75,
        "error extrusion": 0.64,
    }
    assert det.severity == {
        "spaghetti": Severity.CATASTROPHIC,
        "warping": Severity.COSMETIC,
        "stringing": Severity.COSMETIC,
        "zits": Severity.COSMETIC,
        "error extrusion": Severity.COSMETIC,
    }


def test_decision_defaults():
    config = load_config()
    d = config.decision
    assert d.tick_interval_s == 1.0
    assert d.warmup_s == 180
    assert d.ema_alpha == 0.35
    assert d.window == 12
    assert d.vote_threshold == 0.55
    assert d.warn_score == 0.60
    assert d.warn_votes == 4
    assert d.pause_score == 0.75
    assert d.pause_votes == 6
    assert d.pause_consecutive == 3
    assert d.cancel_score == 0.92
    assert d.cancel_votes == 9
    assert d.cancel_consecutive == 2
    assert d.cancel_enabled is False
    assert d.clear_score == 0.40
    assert d.clear_ticks == 5
    assert d.cooldown_s == 300
    assert d.action_mode == ActionMode.NOTIFY_ONLY


def test_quality_defaults():
    q = load_config().quality
    assert q.min_mean_luma == 20.0
    assert q.max_mean_luma == 240.0
    assert q.min_blur_var == 45.0
    assert q.max_frame_age_s == 5.0


def test_moonraker_defaults():
    m = load_config().moonraker
    assert m.base_url == "http://localhost:7125"
    assert m.timeout_s == 5.0
    assert m.poll_interval_s == 1.0
    assert m.pause_macro is None


def test_storage_and_logging_defaults():
    config = load_config()
    assert config.storage.event_log == "logs/events.jsonl"
    assert config.storage.frame_dir == "captures"
    assert config.storage.max_frames == 500
    assert config.storage.retention_days == 14
    assert config.logging.level == "INFO"


# --------------------------------------------------------------------------
# env:VAR_NAME substitution
# --------------------------------------------------------------------------


def test_env_substitution_when_var_is_set(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/webhook/123")
    config = load_config()
    assert config.notify.discord_webhook_url == "https://discord.example/webhook/123"


def test_env_substitution_when_var_is_unset():
    # DISCORD_WEBHOOK_URL is guaranteed unset by the autouse fixture.
    config = load_config()
    assert config.notify.discord_webhook_url is None


def test_env_substitution_requires_exact_form(monkeypatch):
    monkeypatch.setenv("SOME_VAR", "resolved-value")
    data = _base_dict()
    data["notify"]["discord_webhook_url"] = "env:SOME_VAR"
    data["moonraker"]["base_url"] = "not-env:SOME_VAR"  # must NOT be substituted
    config = Config.from_dict(data)
    assert config.notify.discord_webhook_url == "resolved-value"
    assert config.moonraker.base_url == "not-env:SOME_VAR"


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_default_threshold_out_of_range_rejected():
    data = _base_dict()
    data["detector"]["default_threshold"] = 1.5
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_class_threshold_out_of_range_rejected():
    data = _base_dict()
    data["detector"]["class_thresholds"]["spaghetti"] = -0.1
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_nms_iou_out_of_range_rejected():
    data = _base_dict()
    data["detector"]["nms_iou"] = 1.2
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_detector_layout_valid_values_accepted():
    for value in ("auto", "yolov8", "end2end"):
        data = _base_dict()
        data["detector"]["layout"] = value
        config = Config.from_dict(data)
        assert config.detector.layout == value


def test_detector_layout_invalid_value_rejected():
    data = _base_dict()
    data["detector"]["layout"] = "yolov5"
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_detector_kind_valid_values_accepted():
    for value in ("detection", "classification"):
        data = _base_dict()
        data["detector"]["kind"] = value
        if value == "classification":
            data["detector"]["class_names"] = ["normal", "spaghetti"]
        config = Config.from_dict(data)
        assert config.detector.kind == value


def test_detector_kind_invalid_value_rejected():
    data = _base_dict()
    data["detector"]["kind"] = "segmentation"
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_classification_kind_requires_class_names():
    data = _base_dict()
    data["detector"]["kind"] = "classification"
    # class_names deliberately left unset (absent from the example config)
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_classification_kind_with_class_names_accepted():
    data = _base_dict()
    data["detector"]["kind"] = "classification"
    data["detector"]["class_names"] = [
        "normal",
        "spaghetti",
        "cracking",
        "layer_shifting",
        "stringing",
        "warping",
    ]
    config = Config.from_dict(data)
    assert config.detector.class_names == (
        "normal",
        "spaghetti",
        "cracking",
        "layer_shifting",
        "stringing",
        "warping",
    )


def test_detection_kind_does_not_require_class_names():
    data = _base_dict()
    data["detector"]["kind"] = "detection"
    # class_names absent -- detection path doesn't need it
    config = Config.from_dict(data)
    assert config.detector.class_names == ()


def test_pause_score_below_warn_score_rejected():
    data = _base_dict()
    data["decision"]["warn_score"] = 0.6
    data["decision"]["pause_score"] = 0.5
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_cancel_score_below_pause_score_rejected():
    data = _base_dict()
    data["decision"]["pause_score"] = 0.75
    data["decision"]["cancel_score"] = 0.5
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_window_below_one_rejected():
    data = _base_dict()
    data["decision"]["window"] = 0
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_votes_exceeding_window_rejected():
    data = _base_dict()
    data["decision"]["window"] = 5
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_ema_alpha_zero_rejected():
    data = _base_dict()
    data["decision"]["ema_alpha"] = 0.0
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_ema_alpha_above_one_rejected():
    data = _base_dict()
    data["decision"]["ema_alpha"] = 1.5
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_negative_warmup_rejected():
    data = _base_dict()
    data["decision"]["warmup_s"] = -1
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_invalid_action_mode_rejected():
    data = _base_dict()
    data["decision"]["action_mode"] = "destroy_printer"
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_class_thresholds_missing_severity_entry_rejected():
    data = _base_dict()
    data["detector"]["class_thresholds"]["new_defect"] = 0.5
    # no corresponding entry added to severity
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_severity_missing_class_threshold_entry_rejected():
    data = _base_dict()
    data["detector"]["severity"]["new_defect"] = "cosmetic"
    # no corresponding entry added to class_thresholds
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_invalid_severity_value_rejected():
    data = _base_dict()
    data["detector"]["severity"]["spaghetti"] = "extremely_bad"
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_unknown_key_rejected():
    data = _base_dict()
    data["camera"]["unknown_field"] = 123
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_valid_config_round_trips():
    data = _base_dict()
    config = Config.from_dict(data)
    assert config.decision.action_mode == ActionMode.NOTIFY_ONLY
    assert config.detector.severity["spaghetti"] == Severity.CATASTROPHIC


def test_spaghetti_is_the_only_catastrophic_class():
    """Pins the current safety posture: on the measured evaluation results,
    only spaghetti has earned the right to be treated as catastrophic (able
    to drive pause/cancel). Every other class -- including warping, which
    was demoted after its high apparent precision turned out to be ~2 true
    detections on a 31-instance sample -- must stay cosmetic until a
    retrained model demonstrates genuine precision on a larger sample. This
    guards against a future edit silently re-arming an unproven class."""
    config = load_config()
    catastrophic = [
        name
        for name, severity in config.detector.severity.items()
        if severity == Severity.CATASTROPHIC
    ]
    assert catastrophic == ["spaghetti"]
