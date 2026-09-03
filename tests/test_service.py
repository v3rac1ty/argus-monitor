"""Tests for argus.service.ArgusService.

No camera, no network, no model file: every collaborator is a fake or a
`unittest.mock.MagicMock` injected via the constructor, plus the real
`MockDetector` and real `DecisionEngine` (never reimplemented -- the whole
point of these tests is exercising the code that actually ships).

Coverage mirrors the safety properties the service is responsible for:
  - a nominal tick does nothing and never touches Moonraker
  - a sustained failure eventually PAUSEs, exactly once
  - dry_run is a hard ceiling on side effects (no Moonraker call, no
    notification) -- this is the safety guarantee the CLI's --dry-run flag
    is supposed to provide
  - the detector is skipped (not just ignored) when the quality gate fails
    or the printer isn't printing -- proves the inference-skip optimisation
  - a None frame read does not advance the engine
  - print-state polling is throttled to poll_interval_s
  - a tick that raises is caught and logged, not propagated, by run()
"""

from __future__ import annotations

import logging
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from argus.config import (
    CameraConfig,
    Config,
    DecisionConfig,
    DetectorConfig,
    LoggingConfig,
    MoonrakerConfig,
    NotifyConfig,
    QualityConfig,
    StorageConfig,
)
from argus.decision import DecisionEngine
from argus.detectors.mock import MockDetector
from argus.service import ArgusService, _parse_mock_score, main
from argus.types import (
    Action,
    ActionMode,
    DecisionState,
    Frame,
    PrinterState,
    PrintState,
)

T0 = 1_700_000_000.0


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeFrameSource:
    """Yields a fixed sequence of synthetic frames, then None forever.

    A `None` anywhere in `frames` is yielded as-is (simulates "no frame
    available this tick" without ending the sequence).
    """

    def __init__(self, frames: list[Optional[np.ndarray]]) -> None:
        self._frames = frames
        self._index = 0
        self._seq = 0
        self.closed = False

    def read(self) -> Optional[Frame]:
        if self._index >= len(self._frames):
            return None
        image = self._frames[self._index]
        self._index += 1
        if image is None:
            return None
        # Timestamp tracks `seq` (1 real frame per tick in every test that
        # drives `now` forward by 1s/tick) so the quality gate's staleness
        # check ("now - frame.timestamp") never spuriously trips as a long
        # test loop advances `now` far past a fixed T0.
        frame = Frame(image=image, timestamp=T0 + self._seq, seq=self._seq)
        self._seq += 1
        return frame

    def close(self) -> None:
        self.closed = True


def _make_frame_array() -> np.ndarray:
    """A mid-grey, high-texture-enough image that clears the quality gate
    (bright/dark and blur thresholds) under the default QualityConfig."""
    rng = np.random.RandomState(0)
    return rng.randint(60, 200, size=(64, 64, 3)).astype(np.uint8)


def make_cfg(**decision_overrides: object) -> Config:
    """A full Config with a fast, deterministic decision section (no
    warmup, action_mode=PAUSE so PAUSE isn't clamped to NOTIFY) and quality
    thresholds loose enough for the synthetic test frame to pass."""
    decision_kwargs = dict(
        tick_interval_s=1.0,
        warmup_s=0.0,
        ema_alpha=0.35,
        window=12,
        vote_threshold=0.55,
        warn_score=0.60,
        warn_votes=4,
        pause_score=0.75,
        pause_votes=6,
        pause_consecutive=3,
        cancel_score=0.92,
        cancel_votes=9,
        cancel_consecutive=2,
        cancel_enabled=False,
        clear_score=0.40,
        clear_ticks=5,
        cooldown_s=300.0,
        action_mode=ActionMode.PAUSE,
    )
    decision_kwargs.update(decision_overrides)
    return Config(
        camera=CameraConfig(),
        detector=DetectorConfig(),
        decision=DecisionConfig(**decision_kwargs),
        quality=QualityConfig(),
        moonraker=MoonrakerConfig(poll_interval_s=1.0),
        notify=NotifyConfig(),
        storage=StorageConfig(),
        logging=LoggingConfig(),
    )


def make_print_state(
    state: PrinterState = PrinterState.PRINTING,
    filename: str = "benchy.gcode",
    elapsed_s: float = 999.0,
) -> PrintState:
    return PrintState(state=state, filename=filename, elapsed_s=elapsed_s, progress=0.5, fetched_at=T0)


def make_moonraker(print_state: Optional[PrintState] = None) -> MagicMock:
    mock = MagicMock()
    mock.get_print_state.return_value = print_state if print_state is not None else make_print_state()
    mock.pause.return_value = True
    mock.cancel.return_value = True
    return mock


def make_notifier() -> MagicMock:
    mock = MagicMock()
    mock.send.return_value = True
    return mock


def make_store(tmp_path) -> MagicMock:
    mock = MagicMock()
    mock.save_frame.return_value = str(tmp_path / "frame.jpg")
    mock.prune.return_value = 0
    return mock


def make_service(
    cfg: Config,
    *,
    source: FakeFrameSource,
    detector: MockDetector,
    moonraker: MagicMock,
    notifier: MagicMock,
    store: MagicMock,
    dry_run: bool = False,
) -> ArgusService:
    engine = DecisionEngine(cfg.decision)
    return ArgusService(
        cfg,
        source=source,
        detector=detector,
        engine=engine,
        moonraker=moonraker,
        notifier=notifier,
        store=store,
        dry_run=dry_run,
    )


# --------------------------------------------------------------------------
# 1. Nominal tick: no action, no Moonraker call
# --------------------------------------------------------------------------


def test_nominal_tick_produces_no_action_and_no_moonraker_call(tmp_path):
    cfg = make_cfg()
    source = FakeFrameSource([_make_frame_array()])
    detector = MockDetector([0.0], cycle=True)
    moonraker = make_moonraker()
    notifier = make_notifier()
    store = make_store(tmp_path)
    service = make_service(cfg, source=source, detector=detector, moonraker=moonraker, notifier=notifier, store=store)

    decision = service.run_once(T0)

    assert decision is not None
    assert decision.action is Action.NONE
    moonraker.pause.assert_not_called()
    moonraker.cancel.assert_not_called()
    notifier.send.assert_not_called()


# --------------------------------------------------------------------------
# 2. Sustained failure -> exactly one PAUSE, exactly one moonraker.pause()
# --------------------------------------------------------------------------


def test_sustained_failure_eventually_pauses_exactly_once(tmp_path):
    cfg = make_cfg()
    frames = [_make_frame_array() for _ in range(60)]
    source = FakeFrameSource(frames)
    detector = MockDetector([0.95], cycle=True)
    moonraker = make_moonraker()
    notifier = make_notifier()
    store = make_store(tmp_path)
    service = make_service(cfg, source=source, detector=detector, moonraker=moonraker, notifier=notifier, store=store)

    actions = []
    for i in range(60):
        decision = service.run_once(T0 + i)
        actions.append(decision.action if decision is not None else None)

    assert Action.PAUSE in actions
    assert actions.count(Action.PAUSE) == 1
    moonraker.pause.assert_called_once()
    moonraker.cancel.assert_not_called()


# --------------------------------------------------------------------------
# 3. dry_run is a hard ceiling: never calls moonraker or notifier
# --------------------------------------------------------------------------


def test_dry_run_never_calls_moonraker_or_notifier(tmp_path):
    cfg = make_cfg()
    frames = [_make_frame_array() for _ in range(60)]
    source = FakeFrameSource(frames)
    detector = MockDetector([0.95], cycle=True)
    moonraker = make_moonraker()
    notifier = make_notifier()
    store = make_store(tmp_path)
    service = make_service(
        cfg, source=source, detector=detector, moonraker=moonraker, notifier=notifier, store=store, dry_run=True,
    )

    actions = []
    for i in range(60):
        decision = service.run_once(T0 + i)
        actions.append(decision.action if decision is not None else None)

    assert Action.PAUSE in actions
    moonraker.pause.assert_not_called()
    moonraker.cancel.assert_not_called()
    notifier.send.assert_not_called()


# --------------------------------------------------------------------------
# 4. Detector is not invoked when the quality gate fails
# --------------------------------------------------------------------------


def test_detector_not_invoked_when_quality_gate_fails(tmp_path):
    cfg = make_cfg()
    # An all-black frame fails the quality gate's min_mean_luma check.
    black_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    source = FakeFrameSource([black_frame])
    detector = MockDetector([0.9], cycle=True)
    moonraker = make_moonraker()
    notifier = make_notifier()
    store = make_store(tmp_path)
    service = make_service(cfg, source=source, detector=detector, moonraker=moonraker, notifier=notifier, store=store)

    decision = service.run_once(T0)

    assert decision is not None
    assert decision.gate.passed is False
    assert detector.call_count == 0


# --------------------------------------------------------------------------
# 5. Detector is not invoked when the printer isn't printing
# --------------------------------------------------------------------------


def test_detector_not_invoked_when_not_printing(tmp_path):
    cfg = make_cfg()
    source = FakeFrameSource([_make_frame_array()])
    detector = MockDetector([0.9], cycle=True)
    moonraker = make_moonraker(print_state=make_print_state(state=PrinterState.STANDBY))
    notifier = make_notifier()
    store = make_store(tmp_path)
    service = make_service(cfg, source=source, detector=detector, moonraker=moonraker, notifier=notifier, store=store)

    decision = service.run_once(T0)

    assert decision is not None
    assert decision.reason == "not_printing"
    assert detector.call_count == 0


# --------------------------------------------------------------------------
# 6. source.read() returning None returns None and does not advance the engine
# --------------------------------------------------------------------------


def test_no_frame_returns_none_and_does_not_advance_engine(tmp_path):
    cfg = make_cfg()
    source = FakeFrameSource([None, None, _make_frame_array()])
    detector = MockDetector([0.4], cycle=True)
    moonraker = make_moonraker()
    notifier = make_notifier()
    store = make_store(tmp_path)
    engine = DecisionEngine(cfg.decision)
    service = ArgusService(
        cfg, source=source, detector=detector, engine=engine, moonraker=moonraker,
        notifier=notifier, store=store,
    )

    d1 = service.run_once(T0)
    d2 = service.run_once(T0 + 1)
    assert d1 is None
    assert d2 is None
    assert detector.call_count == 0
    assert engine.state is DecisionState.IDLE

    d3 = service.run_once(T0 + 2)
    assert d3 is not None
    assert engine.state is not DecisionState.IDLE


# --------------------------------------------------------------------------
# 7. Print-state polling is throttled to poll_interval_s
# --------------------------------------------------------------------------


def test_print_state_polling_is_throttled(tmp_path):
    cfg = make_cfg()
    cfg = Config(
        camera=cfg.camera, detector=cfg.detector, decision=cfg.decision, quality=cfg.quality,
        moonraker=MoonrakerConfig(poll_interval_s=5.0), notify=cfg.notify, storage=cfg.storage,
        logging=cfg.logging,
    )
    frames = [_make_frame_array() for _ in range(10)]
    source = FakeFrameSource(frames)
    detector = MockDetector([0.0], cycle=True)
    moonraker = make_moonraker()
    notifier = make_notifier()
    store = make_store(tmp_path)
    service = make_service(cfg, source=source, detector=detector, moonraker=moonraker, notifier=notifier, store=store)

    # Ticks at T0, T0+1, ..., T0+9 with poll_interval_s=5: expect polls at
    # T0, T0+5 -> 2 calls, not 10.
    for i in range(10):
        service.run_once(T0 + i)

    assert moonraker.get_print_state.call_count == 2


# --------------------------------------------------------------------------
# 8. An exception inside a tick is caught and logged, not propagated, by run()
# --------------------------------------------------------------------------


def test_run_catches_and_logs_exception_from_tick(tmp_path, monkeypatch, caplog):
    cfg = make_cfg(tick_interval_s=0.01)
    source = FakeFrameSource([_make_frame_array()])
    detector = MockDetector([0.0], cycle=True)
    moonraker = make_moonraker()
    notifier = make_notifier()
    store = make_store(tmp_path)
    service = make_service(cfg, source=source, detector=detector, moonraker=moonraker, notifier=notifier, store=store)

    call_count = {"n": 0}

    def _boom(now: float):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated tick failure")
        service._running = False
        return None

    monkeypatch.setattr(service, "run_once", _boom)

    with caplog.at_level(logging.ERROR):
        service.run()  # must not raise

    assert call_count["n"] == 2
    assert any("simulated tick failure" in r.message or r.exc_info for r in caplog.records)


# --------------------------------------------------------------------------
# 9. --mock-score parsing and CLI wiring
# --------------------------------------------------------------------------


def test_parse_mock_score_single_value():
    assert _parse_mock_score("0.9") == [0.9]


def test_parse_mock_score_comma_sequence():
    assert _parse_mock_score("0.1,0.9,0.95,0.95") == [0.1, 0.9, 0.95, 0.95]


def test_parse_mock_score_out_of_range_rejected():
    with pytest.raises(ValueError, match="out of range"):
        _parse_mock_score("1.5")
    with pytest.raises(ValueError, match="out of range"):
        _parse_mock_score("-0.1")
    with pytest.raises(ValueError, match="out of range"):
        _parse_mock_score("0.5,1.1")


def test_parse_mock_score_non_numeric_rejected():
    with pytest.raises(ValueError, match="not a valid float"):
        _parse_mock_score("nope")


def test_main_mock_score_without_mock_errors(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--mock-score", "0.9"])
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "--mock-score requires --mock" in captured.err


def test_main_mock_score_out_of_range_errors(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--mock", "--mock-score", "1.5"])
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "out of range" in captured.err


# --------------------------------------------------------------------------
# 10. End-to-end: sustained high --mock-score drives a non-NONE action;
#     0.0 never does.
# --------------------------------------------------------------------------


def test_mock_score_end_to_end_high_score_triggers_action(tmp_path):
    cfg = make_cfg()
    script = _parse_mock_score("0.95")
    frames = [_make_frame_array() for _ in range(60)]
    source = FakeFrameSource(frames)
    detector = MockDetector(script, cycle=True)
    moonraker = make_moonraker()
    notifier = make_notifier()
    store = make_store(tmp_path)
    service = make_service(cfg, source=source, detector=detector, moonraker=moonraker, notifier=notifier, store=store)

    actions = [service.run_once(T0 + i).action for i in range(60)]

    assert any(action is not Action.NONE for action in actions)


def test_mock_score_end_to_end_zero_score_never_triggers_action(tmp_path):
    cfg = make_cfg()
    script = _parse_mock_score("0.0")
    frames = [_make_frame_array() for _ in range(60)]
    source = FakeFrameSource(frames)
    detector = MockDetector(script, cycle=True)
    moonraker = make_moonraker()
    notifier = make_notifier()
    store = make_store(tmp_path)
    service = make_service(cfg, source=source, detector=detector, moonraker=moonraker, notifier=notifier, store=store)

    actions = [service.run_once(T0 + i).action for i in range(60)]

    assert all(action is Action.NONE for action in actions)
    moonraker.pause.assert_not_called()
    moonraker.cancel.assert_not_called()
    notifier.send.assert_not_called()
