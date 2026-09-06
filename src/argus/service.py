"""Top-level orchestration: wires camera -> quality gate -> detector ->
DecisionEngine -> Moonraker / notify / storage into one runnable service.

`ArgusService` takes every collaborator by dependency injection so
`run_once` can be tested with mocks (no camera/network/model file);
`build_service` assembles the real ones from a `Config`. Also the `argus`
console-script entry point (`main()`).
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

#: Ticks between `EventStore.prune()` calls; avoids stat'ing the frame dir every tick.
_PRUNE_EVERY_N_TICKS = 100


class ArgusService:
    """Wires one tick of the monitoring pipeline together. See `build_service`
    to assemble the real implementations from a `Config`."""

    # None __init__(self, Config cfg, FrameSource source, Detector detector, DecisionEngine engine, MoonrakerClient moonraker, Notifier notifier, EventStore store, bool dry_run=False)
    # Inputs: Config cfg - the full validated configuration
    #         FrameSource source - frame acquisition collaborator (injected)
    #         Detector detector - model inference collaborator (injected)
    #         DecisionEngine engine - temporal decision collaborator (injected)
    #         MoonrakerClient moonraker - Klipper/Moonraker REST client (injected)
    #         Notifier notifier - alert-delivery collaborator (injected)
    #         EventStore store - JSONL event log + frame archive collaborator (injected)
    #         bool dry_run - if True, skip Moonraker calls and notifications while still logging
    #                        and persisting events; defaults to False
    # Outputs: None
    # Description: Wires every injected collaborator together and initializes per-run state
    #              (current print state cache, last-poll timestamp, tick counter, running flag).
    #              Contains no construction logic of its own -- see `build_service` for that.
    # Side Effects: None beyond storing references and initializing plain instance attributes.
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

    # Optional[Decision] run_once(self, float now)
    # Inputs: float now - caller-supplied unix timestamp for this tick
    # Outputs: Optional[Decision] - the DecisionEngine's Decision for this tick, or None if no
    #                               frame was available (the engine is not advanced in that case)
    # Description: Runs exactly one tick of the pipeline: refresh print state, read a frame, run
    #              the quality gate, run inference only if the gate passed and the printer is
    #              printing, feed the result into the DecisionEngine, act on any non-NONE
    #              decision, and periodically prune old storage.
    # Side Effects: Polls Moonraker for print state (throttled). Reads a frame from the camera
    #               source (device/network/disk I/O). Runs model inference when eligible. Advances
    #               the DecisionEngine's temporal state. On a non-NONE decision: saves a frame,
    #               writes an event, sends a notification, and/or PAUSES/CANCELS THE PRINT via
    #               Moonraker (see `_handle_action`). Every ~100 ticks, prunes old frames/log
    #               entries from storage. Mutates self._tick_count.
    def run_once(self, now: float) -> Optional[Decision]:
        self._refresh_print_state(now)
        print_state = self._print_state

        frame = self._source.read()
        if frame is None:
            return None

        gate = evaluate_frame(frame, self._cfg.quality, now)

        # Only run inference when the gate passed and the printer is printing.
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

    # None run(self)
    # Inputs: None
    # Outputs: None
    # Description: Runs `run_once` on a fixed cadence of `cfg.decision.tick_interval_s` until
    #              interrupted (SIGINT/SIGTERM) or `self._running` is otherwise cleared. Sleep
    #              time is computed from a running target timestamp (not a fixed sleep) so time
    #              spent inside `run_once` does not accumulate as drift; if the loop falls behind
    #              by more than a full tick it resyncs to "now" rather than firing a burst of
    #              catch-up ticks.
    # Side Effects: Installs SIGINT/SIGTERM handlers. Runs an unbounded loop calling `run_once`
    #               (see its side effects) every tick_interval_s, sleeping via time.sleep in
    #               between. Catches and logs any exception from a single tick so the loop
    #               survives a bad tick rather than dying mid-print. Logs info/debug/warning
    #               lines throughout. Mutates self._running.
    def run(self) -> None:
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
                    # Fell behind by a full tick+; resync to "now" instead of catch-up bursts.
                    logger.warning("tick loop falling behind by %.2fs; resyncing", -sleep_for)
                    next_tick = time.time()
        finally:
            logger.info("ArgusService stopping")

    # None close(self)
    # Inputs: None
    # Outputs: None
    # Description: Releases every injected collaborator's resources (source, detector, notifier,
    #              moonraker) in turn. Safe to call even if some collaborators were never used.
    # Side Effects: Calls .close() on the source, detector, notifier, and moonraker collaborators
    #               (releasing devices/connections/sessions). Logs (via logger.exception) rather
    #               than raising if any individual close() fails, so one failure doesn't prevent
    #               closing the rest.
    def close(self) -> None:
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

    # None _refresh_print_state(self, float now)
    # Inputs: float now - caller-supplied unix timestamp for this tick
    # Outputs: None
    # Description: Polls Moonraker for the current print state, throttled to
    #              `cfg.moonraker.poll_interval_s` -- the camera ticks faster than we should
    #              hammer Moonraker with HTTP requests -- and caches the result.
    # Side Effects: When the poll interval has elapsed (or no state has been fetched yet), calls
    #               MoonrakerClient.get_print_state() (an HTTP GET) and mutates
    #               self._print_state and self._last_poll_at.
    def _refresh_print_state(self, now: float) -> None:
        poll_interval = self._cfg.moonraker.poll_interval_s
        if (
            self._print_state is None
            or self._last_poll_at is None
            or (now - self._last_poll_at) >= poll_interval
        ):
            self._print_state = self._moonraker.get_print_state()
            self._last_poll_at = now

    # None _handle_action(self, Decision decision, DetectionResult result, Frame frame, Optional[PrintState] print_state, float now)
    # Inputs: Decision decision - the non-NONE decision produced for this tick
    #         DetectionResult result - the detector output behind this decision (used for the
    #                                   top detection's class/confidence)
    #         Frame frame - the frame behind this decision, to be archived
    #         Optional[PrintState] print_state - current Moonraker print snapshot, or None
    #         float now - caller-supplied unix timestamp for this tick
    # Outputs: None
    # Description: Persists and acts on a tick whose action is not Action.NONE: always saves the
    #              frame and writes the event record (dry-run or not, since
    #              tools/calibrate.py and post-hoc review rely on it), then -- unless in
    #              dry-run mode -- sends the notification and, for PAUSE/CANCEL, calls the
    #              corresponding Moonraker endpoint.
    # Side Effects: Logs an info line. Writes a JPEG frame to disk and appends a JSON line to the
    #               event log (via EventStore). When not in dry-run mode: sends a notification
    #               (network I/O via Notifier.send), and for Action.PAUSE/Action.CANCEL, PAUSES or
    #               CANCELS THE USER'S PRINT via MoonrakerClient. In dry-run mode, only logs that
    #               it would have done so -- no notification is sent and no Moonraker call is made.
    def _handle_action(
        self,
        decision: Decision,
        result: DetectionResult,
        frame: Frame,
        print_state: Optional[PrintState],
        now: float,
    ) -> None:
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

    # None _install_signal_handlers(self)
    # Inputs: None
    # Outputs: None
    # Description: Installs a handler for SIGINT and SIGTERM (where available on this platform)
    #              that clears self._running so the `run` loop exits cleanly on the next
    #              iteration.
    # Side Effects: Calls signal.signal() to register handlers for SIGINT/SIGTERM -- process-wide
    #               signal disposition changes. Logs at debug level if installation is skipped
    #               (e.g. not running on the main thread); never raises.
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


# ArgusService build_service(Config cfg, bool dry_run=False, Optional[FrameSource] source_override=None, Optional[Detector] detector_override=None)
# Inputs: Config cfg - the full validated configuration
#         bool dry_run - if True, ArgusService skips Moonraker calls and notifications; defaults
#                         to False
#         Optional[FrameSource] source_override - replaces the camera source built from
#                                                   `cfg.camera`; defaults to None (build from
#                                                   cfg); used by the CLI's --source flag / tests
#         Optional[Detector] detector_override - replaces the ONNX detector built from
#                                                 `cfg.detector`; defaults to None (build from
#                                                 cfg); used by the CLI's --mock flag
# Outputs: ArgusService - a fully wired service with real collaborators
# Description: Assembles the real FrameSource, Detector, DecisionEngine, MoonrakerClient,
#              Notifier, and EventStore from `cfg` (honoring any overrides) and injects them into
#              a new ArgusService.
# Side Effects: Constructing the real collaborators may touch hardware/network/filesystem (e.g.
#               OnnxYoloDetector loading a model file, MoonrakerClient/build_notifier creating
#               HTTP sessions) -- see each collaborator's own __init__ for specifics.
def build_service(
    cfg: Config,
    *,
    dry_run: bool = False,
    source_override: Optional[FrameSource] = None,
    detector_override: Optional[Detector] = None,
) -> ArgusService:
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


# argparse.ArgumentParser _build_arg_parser()
# Inputs: None
# Outputs: argparse.ArgumentParser - the configured CLI parser for the `argus` entry point
# Description: Declares every CLI flag (--config, --dry-run, --source, --once, --log-level,
#              --model, --mock, --mock-score) and their help text.
# Side Effects: None
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


# list[float] _parse_mock_score(str raw)
# Inputs: str raw - the raw --mock-score CLI value: a single float string, or a comma-separated
#                    sequence of them
# Outputs: list[float] - the parsed sequence of floats, each in [0.0, 1.0]
# Description: Parses `--mock-score` into a non-empty list of floats in [0.0, 1.0], accepting
#              either a single value ("0.9") or a comma-separated sequence ("0.1,0.9,0.95,0.95").
# Side Effects: Raises ValueError with a descriptive message on any empty, malformed, or
#               out-of-range entry.
def _parse_mock_score(raw: str) -> list[float]:
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


# None main(Optional[list[str]] argv=None)
# Inputs: Optional[list[str]] argv - CLI arguments to parse; defaults to None, in which case
#                                     argparse falls back to sys.argv[1:]
# Outputs: None
# Description: The `argus` console-script entry point: parses CLI flags, loads and optionally
#              overrides the config (source/model), configures logging, builds the service (real
#              or mock detector per --mock), and runs it once (--once) or in a loop, always
#              closing collaborators on exit.
# Side Effects: Reads a config file from disk (load_config). Configures process-wide logging
#               (logging.basicConfig) to stderr. Constructs real collaborators including
#               hardware/network/filesystem access (via build_service). Runs the monitoring loop
#               or a single tick, which can PAUSE/CANCEL THE USER'S PRINT and send notifications
#               (see run_once/run/_handle_action). Calls parser.error (prints to stderr and exits
#               the process) on invalid argument combinations. Always calls service.close() in a
#               finally block.
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
