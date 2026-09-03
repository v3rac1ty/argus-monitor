"""Top-level orchestration: wires camera -> quality gate -> detector ->
DecisionEngine -> Moonraker / notify / storage into one runnable service.

`ArgusService` takes every collaborator by dependency injection (a
`FrameSource`, a `Detector`, a `DecisionEngine`, a `MoonrakerClient`, a
`Notifier`, an `EventStore`) so `run_once` can be exercised in tests with
mocks and a `MockDetector` -- no camera, no network, no model file. Building
the real collaborators from a `Config` is a separate, thin concern handled
by `build_service`.

This module is also the `argus` console-script entry point (see
`pyproject.toml`'s `[project.scripts]`); `main()` is a thin CLI wrapper
around `build_service` + `ArgusService.run`/`run_once`.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import signal
import sys
import time
from types import FrameType
from typing import Optional

from argus.camera import FrameSource, build_source
from argus.config import Config, load_config
from argus.decision import DecisionEngine
from argus.detectors.base import Detector
from argus.detectors.mock import MockDetector
from argus.detectors.onnx_yolo import OnnxYoloDetector
from argus.moonraker import MoonrakerClient
from argus.notify import Notifier, build_notifier
from argus.quality import evaluate_frame
from argus.storage import EventStore
from argus.types import (
    Action,
    Decision,
    DetectionResult,
    Event,
    Frame,
    PrintState,
)

logger = logging.getLogger(__name__)

#: How often (in successfully-scored ticks) to call `EventStore.prune()`.
#: Pruning on every tick would mean stat'ing the whole frame directory once
#: per second for no benefit; every 100 ticks (~100s at the default 1s tick
#: interval) keeps disk use bounded without meaningful overhead.
_PRUNE_EVERY_N_TICKS = 100


class ArgusService:
    """Wires one tick of the monitoring pipeline together.

    Every collaborator is injected so `run_once` is fully unit-testable
    with fakes/mocks (see tests/test_service.py) -- this class contains no
    construction logic of its own. Use `build_service` to assemble the real
    implementations from a `Config`.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        source: FrameSource,
        detector: Detector,
        engine: DecisionEngine,
        moonraker: MoonrakerClient,
        notifier: Notifier,
        store: EventStore,
        dry_run: bool = False,
    ) -> None:
        self._cfg = cfg
        self._source = source
        self._detector = detector
        self._engine = engine
        self._moonraker = moonraker
        self._notifier = notifier
        self._store = store
        self._dry_run = dry_run

        self._print_state: Optional[PrintState] = None
        self._last_poll_at: Optional[float] = None
        self._tick_count = 0
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_once(self, now: float) -> Optional[Decision]:
        """Run exactly one tick of the pipeline. Returns the `Decision` for
        this tick, or `None` if no frame was available (the engine is not
        advanced in that case)."""
        self._refresh_print_state(now)
        print_state = self._print_state

        frame = self._source.read()
        if frame is None:
            return None

        gate = evaluate_frame(frame, self._cfg.quality, now)

        # Inference is the expensive step: only pay for it when the gate
        # passed AND the printer is actually printing -- there is no reason
        # to score a frame the engine's own gate would refuse anyway.
        should_infer = gate.passed and print_state is not None and print_state.is_printing
        if should_infer:
            result = self._detector.infer(frame)
        else:
            result = DetectionResult.empty()

        decision = self._engine.tick(
            p_raw=result.p_failure,
            detections=result.detections,
            print_state=print_state,
            quality=gate,
            now=now,
        )

        if decision.action is not Action.NONE:
            self._handle_action(decision, result, frame, print_state, now)

        self._tick_count += 1
        if self._tick_count % _PRUNE_EVERY_N_TICKS == 0:
            removed = self._store.prune()
            if removed:
                logger.debug("pruned %d old frame(s)/log entries", removed)

        return decision

    def run(self) -> None:
        """Run `run_once` on a fixed cadence of `cfg.decision.tick_interval_s`
        until interrupted (SIGINT/SIGTERM) or an unrecoverable error.

        Sleep time is computed from a running target timestamp (not a fixed
        `time.sleep(tick_interval_s)`) so time spent inside `run_once` does
        not accumulate as drift. Each iteration is wrapped so a transient
        error is logged and the loop continues -- a monitoring daemon must
        survive a bad tick, not die mid-print.
        """
        self._running = True
        self._install_signal_handlers()

        tick_interval = self._cfg.decision.tick_interval_s
        next_tick = time.time()
        logger.info(
            "ArgusService starting (tick_interval_s=%.2f, dry_run=%s, action_mode=%s)",
            tick_interval,
            self._dry_run,
            self._cfg.decision.action_mode.value,
        )
        try:
            while self._running:
                now = time.time()
                try:
                    decision = self.run_once(now)
                    if decision is not None:
                        logger.debug(
                            "heartbeat: score=%.3f p_raw=%.3f state=%s votes=%d/%d action=%s",
                            decision.score,
                            decision.p_raw,
                            decision.state.value,
                            decision.votes,
                            decision.window,
                            decision.action.value,
                        )
                except Exception:
                    logger.exception("unhandled error during tick; continuing")

                next_tick += tick_interval
                sleep_for = next_tick - time.time()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                elif -sleep_for > tick_interval:
                    # We've fallen behind by more than a full tick (e.g. a
                    # slow inference or a hiccup) -- resync to "now" rather
                    # than firing a burst of immediate catch-up ticks.
                    logger.warning("tick loop falling behind by %.2fs; resyncing", -sleep_for)
                    next_tick = time.time()
        finally:
            logger.info("ArgusService stopping")

    def close(self) -> None:
        """Release every injected collaborator's resources. Safe to call
        even if some collaborators were never used."""
        for name, obj in (
            ("source", self._source),
            ("detector", self._detector),
            ("notifier", self._notifier),
            ("moonraker", self._moonraker),
        ):
            try:
                obj.close()
            except Exception:
                logger.exception("error closing %s", name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_print_state(self, now: float) -> None:
        """Poll Moonraker for print state, throttled to
        `cfg.moonraker.poll_interval_s` -- the camera ticks faster than we
        should hammer Moonraker with HTTP requests."""
        poll_interval = self._cfg.moonraker.poll_interval_s
        if (
            self._print_state is None
            or self._last_poll_at is None
            or (now - self._last_poll_at) >= poll_interval
        ):
            self._print_state = self._moonraker.get_print_state()
            self._last_poll_at = now

    def _handle_action(
        self,
        decision: Decision,
        result: DetectionResult,
        frame: Frame,
        print_state: Optional[PrintState],
        now: float,
    ) -> None:
        """Persist and act on a tick whose action is not `Action.NONE`.

        Always saves the frame and writes the event -- that record is what
        `tools/calibrate.py` and post-hoc review rely on, dry-run or not.
        Only the consequential parts (the notification and the Moonraker
        pause/cancel call) are skipped in dry-run mode.
        """
        logger.info(
            "action=%s state=%s score=%.3f votes=%d/%d reason=%s",
            decision.action.value,
            decision.state.value,
            decision.score,
            decision.votes,
            decision.window,
            decision.reason,
        )

        frame_path = self._store.save_frame(frame, now)
        top = result.top
        event = Event(
            timestamp=now,
            action=decision.action,
            state=decision.state,
            score=decision.score,
            p_raw=decision.p_raw,
            votes=decision.votes,
            reason=decision.reason,
            class_name=top.class_name if top is not None else None,
            confidence=top.confidence if top is not None else None,
            frame_path=frame_path,
            print_filename=print_state.filename if print_state is not None else None,
            elapsed_s=print_state.elapsed_s if print_state is not None else None,
            detections=decision.detections,
        )
        self._store.write_event(event)

        if self._dry_run:
            logger.info(
                "DRY-RUN: would send notification and take action=%s -- "
                "no Moonraker call made, no notification sent",
                decision.action.value,
            )
            return

        self._notifier.send(event, frame_path)

        if decision.action is Action.PAUSE:
            self._moonraker.pause()
        elif decision.action is Action.CANCEL:
            self._moonraker.cancel()
        # Action.NOTIFY: notification only, already sent above -- no
        # printer call.

    def _install_signal_handlers(self) -> None:
        def _handler(signum: int, frame: Optional[FrameType]) -> None:
            logger.info("received signal %s, shutting down", signum)
            self._running = False

        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                # signal.signal only works from the main thread; harmless
                # to skip installation when called from elsewhere (e.g. a
                # test running the service on a worker thread).
                logger.debug("could not install handler for %s", sig_name)


def build_service(
    cfg: Config,
    *,
    dry_run: bool = False,
    source_override: Optional[FrameSource] = None,
    detector_override: Optional[Detector] = None,
) -> ArgusService:
    """Construct an `ArgusService` with real collaborators from `cfg`.

    `source_override` lets a caller (e.g. `--source` on the CLI, or a test)
    replace the camera source without touching `cfg`. `detector_override`
    does the same for the detector -- used by the CLI's `--mock` flag to
    swap in `MockDetector` without duplicating the rest of this wiring.
    """
    source = source_override if source_override is not None else build_source(cfg.camera)
    detector = detector_override if detector_override is not None else OnnxYoloDetector(cfg.detector)
    engine = DecisionEngine(cfg.decision)
    moonraker = MoonrakerClient(cfg.moonraker)
    notifier = build_notifier(cfg.notify)
    store = EventStore(cfg.storage)

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
# CLI
# --------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="argus",
        description="Argus Monitor: camera-based AI print-failure detection for Klipper/Moonraker.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML (default: config.example.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log what would happen but never call Moonraker or send a notification",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Override camera.source (e.g. a directory of frames for offline replay)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single tick then exit, instead of looping forever",
    )
    parser.add_argument(
        "--log-level", type=str, default=None,
        help="Override logging.level (DEBUG/INFO/WARNING/ERROR/CRITICAL)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override detector.model_path",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use MockDetector instead of the real ONNX detector (no trained model needed)",
    )
    parser.add_argument(
        "--mock-score", type=str, default=None,
        help=(
            "Only applies with --mock: the constant p_failure MockDetector reports "
            "(float in [0.0, 1.0], default 0.0), or a comma-separated sequence "
            "(e.g. 0.1,0.9,0.95,0.95) to script a ramp that is replayed through the "
            "real DecisionEngine, cycling once exhausted."
        ),
    )
    return parser


def _parse_mock_score(raw: str) -> list[float]:
    """Parse `--mock-score` into a non-empty list of floats in [0.0, 1.0].

    Accepts a single value ("0.9") or a comma-separated sequence
    ("0.1,0.9,0.95,0.95"). Raises ValueError with a clear message on any
    malformed or out-of-range entry.
    """
    parts = [p.strip() for p in raw.split(",")]
    if any(not p for p in parts):
        raise ValueError(f"--mock-score has an empty entry in {raw!r}")

    scores: list[float] = []
    for part in parts:
        try:
            value = float(part)
        except ValueError as exc:
            raise ValueError(f"--mock-score entry {part!r} is not a valid float") from exc
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"--mock-score entry {value} out of range; must be in [0.0, 1.0]")
        scores.append(value)
    return scores


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.mock_score is not None and not args.mock:
        parser.error("--mock-score requires --mock")

    mock_script = [0.0]
    if args.mock_score is not None:
        try:
            mock_script = _parse_mock_score(args.mock_score)
        except ValueError as exc:
            parser.error(str(exc))

    cfg = load_config(args.config)

    if args.source is not None:
        cfg = dataclasses.replace(cfg, camera=dataclasses.replace(cfg.camera, source=args.source))
    if args.model is not None:
        cfg = dataclasses.replace(cfg, detector=dataclasses.replace(cfg.detector, model_path=args.model))

    log_level_name = (args.log_level or cfg.logging.level).upper()
    logging.basicConfig(
        level=getattr(logging, log_level_name, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    detector_override = MockDetector(mock_script, cycle=True) if args.mock else None

    service = build_service(cfg, dry_run=args.dry_run, detector_override=detector_override)
    try:
        if args.once:
            decision = service.run_once(time.time())
            if decision is not None:
                logger.info(
                    "tick result: action=%s state=%s score=%.3f p_raw=%.3f votes=%d/%d reason=%s",
                    decision.action.value,
                    decision.state.value,
                    decision.score,
                    decision.p_raw,
                    decision.votes,
                    decision.window,
                    decision.reason,
                )
            else:
                logger.info("tick result: no frame available")
        else:
            service.run()
    finally:
        service.close()


if __name__ == "__main__":
    main()
