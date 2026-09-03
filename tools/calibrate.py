"""Measure false-positive rate and detection latency of the REAL
`argus.decision.DecisionEngine` against recorded footage.

This never reimplements the decision logic -- the whole point is measuring
the code that actually ships. Two input modes:

  --frames DIR [DIR ...]   Replay a directory of image frames through a real
                            detector (ONNX or, with --mock, MockDetector).
                            Slow: inference runs once per frame.

  --scores FILE [FILE ...] Replay a JSONL of previously-computed per-tick
                            scores (produced by a prior --frames run with
                            --scores-out). Fast: no inference, no frame
                            decoding -- just DecisionEngine.tick() in a
                            loop, so sweeping thresholds is cheap.

Each input (one directory, or one JSONL file) is treated as one independent
print "trace" (continuous timeline starting at warmup). Report:

  - false positives per print-hour (the project's headline safety metric)
  - total actions fired, broken down by NOTIFY/PAUSE/CANCEL
  - mean time-to-detect, when --failure-start-tick marks a known failure
    onset in every trace

--sweep varies pause_score / pause_votes / ema_alpha over a grid (each
combo re-replays every trace against the SAME recorded scores -- no new
inference), prints a ranked table, and recommends a `decision:` YAML block
-- the evidence needed before graduating from notify_only to pause.

Usage:
    # One-shot: build a score cache from a directory of frames, replay it.
    python tools/calibrate.py --frames datasets/captures/nominal_run1 \\
        --scores-out /tmp/nominal_run1.jsonl

    # Fast re-sweep against a cached score file.
    python tools/calibrate.py --scores /tmp/nominal_run1.jsonl --sweep \\
        --sweep-pause-score 0.65,0.75,0.85 --sweep-pause-votes 4,6,8

    # Time-to-detect against a trace with a known failure onset at tick 400.
    python tools/calibrate.py --scores /tmp/spaghetti_run1.jsonl \\
        --failure-start-tick 400
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import cv2

from argus.config import Config, DecisionConfig, load_config
from argus.decision import DecisionEngine
from argus.detectors.base import Detector
from argus.detectors.mock import MockDetector
from argus.detectors.onnx_yolo import OnnxYoloDetector
from argus.quality import evaluate_frame
from argus.types import (
    Action,
    Decision,
    Detection,
    DetectionResult,
    Frame,
    GateResult,
    PrinterState,
    PrintState,
    Severity,
)

logger = logging.getLogger("argus.tools.calibrate")

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

#: Arbitrary fixed epoch for synthetic tick timestamps -- keeps replay
#: fully deterministic and independent of wall-clock time.
_BASE_TS = 1_700_000_000.0


# --------------------------------------------------------------------------
# TickRecord: one frame's worth of pre-computed detector/gate output
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TickRecord:
    """One tick's detector + quality-gate output, decoupled from any
    particular DecisionConfig -- what gets cached to/from a --scores JSONL
    file so a threshold sweep never has to re-run inference."""

    tick: int
    timestamp: float
    p_raw: float
    gate_passed: bool
    gate_reason: Optional[str]
    detections: tuple[Detection, ...]


def _detection_to_dict(d: Detection) -> dict:
    return {
        "class_id": d.class_id,
        "class_name": d.class_name,
        "confidence": d.confidence,
        "bbox": list(d.bbox),
        "severity": d.severity.value,
    }


def _detection_from_dict(d: dict) -> Detection:
    return Detection(
        class_id=int(d["class_id"]),
        class_name=str(d["class_name"]),
        confidence=float(d["confidence"]),
        bbox=tuple(float(x) for x in d["bbox"]),  # type: ignore[arg-type]
        severity=Severity(d["severity"]),
    )


def _record_to_dict(r: TickRecord) -> dict:
    return {
        "tick": r.tick,
        "timestamp": r.timestamp,
        "p_raw": r.p_raw,
        "gate_passed": r.gate_passed,
        "gate_reason": r.gate_reason,
        "detections": [_detection_to_dict(d) for d in r.detections],
    }


def _record_from_dict(d: dict) -> TickRecord:
    return TickRecord(
        tick=int(d["tick"]),
        timestamp=float(d["timestamp"]),
        p_raw=float(d["p_raw"]),
        gate_passed=bool(d.get("gate_passed", True)),
        gate_reason=d.get("gate_reason"),
        detections=tuple(_detection_from_dict(x) for x in d.get("detections", ())),
    )


def write_scores(records: Sequence[TickRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(_record_to_dict(r)) + "\n")


def load_scores(path: Path) -> list[TickRecord]:
    records: list[TickRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(_record_from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: malformed score record: {exc}") from exc
    return records


# --------------------------------------------------------------------------
# Computing TickRecords from a directory of frames (runs inference once)
# --------------------------------------------------------------------------


def _list_image_files(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )


def compute_scores_from_frames(
    directory: Path,
    detector: Detector,
    cfg: Config,
    tick_interval_s: float,
) -> list[TickRecord]:
    """Run the quality gate + detector once per frame in `directory` (sorted
    filename order) and return one `TickRecord` per frame.

    Frame timestamps are synthetic (`_BASE_TS + tick * tick_interval_s`),
    not file mtimes, so the quality gate's staleness check never spuriously
    trips during replay regardless of how long computing scores takes.
    """
    files = _list_image_files(directory)
    if not files:
        raise ValueError(f"no image files ({_IMAGE_EXTENSIONS}) found in '{directory}'")

    records: list[TickRecord] = []
    for i, path in enumerate(files):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("skipping unreadable image %s", path)
            continue
        ts = _BASE_TS + i * tick_interval_s
        frame = Frame(image=image, timestamp=ts, seq=i)

        gate = evaluate_frame(frame, cfg.quality, now=ts)
        if gate.passed:
            result = detector.infer(frame)
        else:
            result = DetectionResult.empty()

        records.append(
            TickRecord(
                tick=i,
                timestamp=ts,
                p_raw=result.p_failure,
                gate_passed=gate.passed,
                gate_reason=gate.reason,
                detections=result.detections,
            )
        )
        if (i + 1) % 100 == 0:
            logger.info("scored %d/%d frames in %s", i + 1, len(files), directory)

    logger.info("scored %d frame(s) from %s", len(records), directory)
    return records


# --------------------------------------------------------------------------
# Replaying TickRecords through the REAL DecisionEngine
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TraceResult:
    n_ticks: int
    duration_s: float
    actions: tuple[tuple[int, Decision], ...]
    false_positives: int
    time_to_detect_s: Optional[float]
    detected: Optional[bool]  # None if no failure_start_tick was given


def replay_trace(
    records: Sequence[TickRecord],
    decision_cfg: DecisionConfig,
    failure_start_tick: Optional[int] = None,
) -> TraceResult:
    """Feed `records` through a fresh `DecisionEngine(decision_cfg)`, one
    tick at a time, exactly as `ArgusService.run_once` would (synthesizing
    a continuously-PRINTING `PrintState` since recorded footage has no
    Moonraker to poll).

    Action classification when `failure_start_tick` is given: any action at
    a tick before it is a false positive (the failure hadn't started yet);
    the first action at or after it is the true detection (used for
    time-to-detect); any further actions after that first true detection
    are neither -- they're the engine continuing to react to the same
    still-ongoing real failure, not a fresh false alarm. With no
    `failure_start_tick`, the trace is assumed entirely nominal, so every
    fired action is a false positive.
    """
    engine = DecisionEngine(decision_cfg)
    if not records:
        return TraceResult(
            n_ticks=0, duration_s=0.0, actions=(), false_positives=0,
            time_to_detect_s=None, detected=None,
        )

    base_ts = records[0].timestamp
    failure_start_ts: Optional[float] = None

    actions: list[tuple[int, Decision]] = []
    false_positives = 0
    time_to_detect_s: Optional[float] = None
    detected = False

    for rec in records:
        if failure_start_tick is not None and rec.tick == failure_start_tick:
            failure_start_ts = rec.timestamp

        print_state = PrintState(
            state=PrinterState.PRINTING,
            filename="calibration",
            elapsed_s=rec.timestamp - base_ts,
            progress=0.0,
            fetched_at=rec.timestamp,
        )
        gate = GateResult.ok() if rec.gate_passed else GateResult.blocked(rec.gate_reason or "blocked")

        decision = engine.tick(
            p_raw=rec.p_raw,
            detections=rec.detections,
            print_state=print_state,
            quality=gate,
            now=rec.timestamp,
        )

        if decision.action is not Action.NONE:
            actions.append((rec.tick, decision))
            if failure_start_tick is None or rec.tick < failure_start_tick:
                false_positives += 1
            elif not detected:
                detected = True
                if failure_start_ts is not None:
                    time_to_detect_s = rec.timestamp - failure_start_ts

    duration_s = records[-1].timestamp - records[0].timestamp
    return TraceResult(
        n_ticks=len(records),
        duration_s=duration_s,
        actions=tuple(actions),
        false_positives=false_positives,
        time_to_detect_s=time_to_detect_s,
        detected=(detected if failure_start_tick is not None else None),
    )


@dataclasses.dataclass(frozen=True)
class AggregateResult:
    n_traces: int
    total_ticks: int
    total_hours: float
    total_actions: int
    actions_by_type: dict[str, int]
    total_false_positives: int
    fp_per_hour: float
    mean_time_to_detect_s: Optional[float]
    detected_fraction: Optional[float]


def aggregate(results: Sequence[TraceResult]) -> AggregateResult:
    total_ticks = sum(r.n_ticks for r in results)
    total_hours = sum(r.duration_s for r in results) / 3600.0
    total_actions = sum(len(r.actions) for r in results)

    actions_by_type: dict[str, int] = {}
    for r in results:
        for _, decision in r.actions:
            actions_by_type[decision.action.value] = actions_by_type.get(decision.action.value, 0) + 1

    total_fp = sum(r.false_positives for r in results)
    fp_per_hour = (total_fp / total_hours) if total_hours > 0 else 0.0

    ttds = [r.time_to_detect_s for r in results if r.time_to_detect_s is not None]
    mean_ttd = (sum(ttds) / len(ttds)) if ttds else None

    detects = [r.detected for r in results if r.detected is not None]
    detected_fraction = (sum(1 for d in detects if d) / len(detects)) if detects else None

    return AggregateResult(
        n_traces=len(results),
        total_ticks=total_ticks,
        total_hours=total_hours,
        total_actions=total_actions,
        actions_by_type=actions_by_type,
        total_false_positives=total_fp,
        fp_per_hour=fp_per_hour,
        mean_time_to_detect_s=mean_ttd,
        detected_fraction=detected_fraction,
    )


def _fmt_ttd(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    return f"{seconds:.1f}s"


def print_report(agg: AggregateResult, failure_start_tick: Optional[int]) -> None:
    print("=" * 72)
    print("CALIBRATION REPORT")
    print("=" * 72)
    print(f"Traces replayed:       {agg.n_traces}")
    print(f"Total ticks:           {agg.total_ticks}")
    print(f"Total print-time:      {agg.total_hours:.3f} hours")
    print(f"Total actions fired:   {agg.total_actions}  {agg.actions_by_type or {}}")
    print(f"False positives:       {agg.total_false_positives}")
    print(f"FALSE POSITIVES/HOUR:  {agg.fp_per_hour:.4f}")
    if failure_start_tick is not None:
        print(f"Mean time-to-detect:   {_fmt_ttd(agg.mean_time_to_detect_s)}")
        det_pct = "n/a" if agg.detected_fraction is None else f"{agg.detected_fraction * 100:.1f}%"
        print(f"Detected fraction:     {det_pct}")
    print("=" * 72)


# --------------------------------------------------------------------------
# Sweep mode
# --------------------------------------------------------------------------


def _parse_float_grid(spec: Optional[str], fallback: float) -> list[float]:
    if not spec:
        return [fallback]
    return [float(x) for x in spec.split(",") if x.strip()]


def _parse_int_grid(spec: Optional[str], fallback: int) -> list[int]:
    if not spec:
        return [fallback]
    return [int(x) for x in spec.split(",") if x.strip()]


@dataclasses.dataclass(frozen=True)
class SweepRow:
    pause_score: float
    pause_votes: int
    ema_alpha: float
    agg: AggregateResult


def run_sweep(
    traces: Sequence[Sequence[TickRecord]],
    base_decision_cfg: DecisionConfig,
    pause_scores: Sequence[float],
    pause_votes_grid: Sequence[int],
    ema_alphas: Sequence[float],
    failure_start_tick: Optional[int],
) -> list[SweepRow]:
    rows: list[SweepRow] = []
    for pause_score, pause_votes, ema_alpha in itertools.product(pause_scores, pause_votes_grid, ema_alphas):
        decision_cfg = dataclasses.replace(
            base_decision_cfg, pause_score=pause_score, pause_votes=pause_votes, ema_alpha=ema_alpha,
        )
        results = [replay_trace(trace, decision_cfg, failure_start_tick) for trace in traces]
        rows.append(SweepRow(pause_score, pause_votes, ema_alpha, aggregate(results)))
    return rows


def print_sweep_table(rows: Sequence[SweepRow], failure_start_tick: Optional[int]) -> None:
    ranked = sorted(
        rows,
        key=lambda r: (
            r.agg.fp_per_hour,
            r.agg.mean_time_to_detect_s if r.agg.mean_time_to_detect_s is not None else float("inf"),
        ),
    )
    print("=" * 88)
    print("SWEEP RESULTS (sorted: lowest FP/hour first, then fastest time-to-detect)")
    print("=" * 88)
    header = f"{'pause_score':>12}{'pause_votes':>13}{'ema_alpha':>11}{'FP/hour':>12}{'actions':>10}{'time-to-detect':>18}"
    print(header)
    print("-" * len(header))
    for row in ranked:
        print(
            f"{row.pause_score:>12.3f}{row.pause_votes:>13d}{row.ema_alpha:>11.3f}"
            f"{row.agg.fp_per_hour:>12.4f}{row.agg.total_actions:>10d}"
            f"{_fmt_ttd(row.agg.mean_time_to_detect_s):>18}"
        )
    print("=" * 88)

    best = ranked[0]
    print()
    print(f"Recommended (lowest FP/hour = {best.agg.fp_per_hour:.4f}", end="")
    if failure_start_tick is not None:
        print(f", mean time-to-detect = {_fmt_ttd(best.agg.mean_time_to_detect_s)}", end="")
    print("):")
    print("decision:")
    print(f"  pause_score: {best.pause_score}")
    print(f"  pause_votes: {best.pause_votes}")
    print(f"  ema_alpha: {best.ema_alpha}")
    if best.agg.fp_per_hour > 0:
        print(
            "  # NOTE: this grid did not find a zero-FP/hour setting -- widen the sweep, "
            "record more nominal footage, or keep action_mode: notify_only until it does."
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay recorded footage through the real DecisionEngine to measure "
                    "false-positive rate and detection latency.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (default: config.example.yaml)")
    parser.add_argument(
        "--frames", type=Path, nargs="+", default=None,
        help="Directory(ies) of frames to score+replay; each is one independent trace",
    )
    parser.add_argument(
        "--scores", type=Path, nargs="+", default=None,
        help="Pre-computed JSONL score file(s) to replay; each is one independent trace",
    )
    parser.add_argument(
        "--scores-out", type=Path, default=None,
        help="With a single --frames directory, save computed scores to this JSONL for fast re-sweeping",
    )
    parser.add_argument("--model", type=str, default=None, help="Override detector.model_path (--frames mode only)")
    parser.add_argument(
        "--mock", action="store_true",
        help="Use MockDetector (script=[0.0], nominal) instead of the real ONNX detector (--frames mode only)",
    )
    parser.add_argument(
        "--tick-interval-s", type=float, default=None,
        help="Synthetic seconds/tick for --frames mode (default: decision.tick_interval_s from config)",
    )
    parser.add_argument(
        "--failure-start-tick", type=int, default=None,
        help="Tick index (0-based, applied per trace) at which a real failure begins; actions "
             "before it are false positives, the first action at/after it is the true detection",
    )
    parser.add_argument(
        "--action-mode", type=str, default="pause", choices=["notify_only", "pause", "cancel"],
        help="action_mode to simulate during replay (default: pause -- so PAUSE-level firings "
             "are visible even if the live config is still notify_only)",
    )
    parser.add_argument("--cancel-enabled", action="store_true", help="Simulate with decision.cancel_enabled=true")
    parser.add_argument("--sweep", action="store_true", help="Sweep pause_score/pause_votes/ema_alpha over a grid")
    parser.add_argument("--sweep-pause-score", type=str, default=None, help="Comma-separated pause_score grid")
    parser.add_argument("--sweep-pause-votes", type=str, default=None, help="Comma-separated pause_votes grid")
    parser.add_argument("--sweep-ema-alpha", type=str, default=None, help="Comma-separated ema_alpha grid")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def _build_decision_cfg(cfg: Config, args: argparse.Namespace) -> DecisionConfig:
    from argus.types import ActionMode

    overrides: dict[str, object] = {"action_mode": ActionMode(args.action_mode)}
    if args.cancel_enabled:
        overrides["cancel_enabled"] = True
    if args.tick_interval_s is not None:
        overrides["tick_interval_s"] = args.tick_interval_s
    return dataclasses.replace(cfg.decision, **overrides)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    if args.model is not None:
        cfg = dataclasses.replace(cfg, detector=dataclasses.replace(cfg.detector, model_path=args.model))

    if not args.frames and not args.scores:
        raise SystemExit("must pass --frames DIR [DIR ...] and/or --scores FILE [FILE ...]")

    tick_interval_s = args.tick_interval_s or cfg.decision.tick_interval_s

    traces: list[list[TickRecord]] = []

    if args.frames:
        if args.scores_out is not None and len(args.frames) != 1:
            raise SystemExit("--scores-out requires exactly one --frames directory")

        detector: Detector = MockDetector([0.0], cycle=True) if args.mock else OnnxYoloDetector(cfg.detector)
        try:
            for directory in args.frames:
                records = compute_scores_from_frames(directory, detector, cfg, tick_interval_s)
                traces.append(records)
                if args.scores_out is not None:
                    write_scores(records, args.scores_out)
                    logger.info("wrote %d score record(s) to %s", len(records), args.scores_out)
        finally:
            detector.close()

    if args.scores:
        for path in args.scores:
            traces.append(load_scores(path))

    if not traces:
        raise SystemExit("no traces to replay")

    decision_cfg = _build_decision_cfg(cfg, args)

    if args.sweep:
        pause_scores = _parse_float_grid(args.sweep_pause_score, decision_cfg.pause_score)
        pause_votes_grid = _parse_int_grid(args.sweep_pause_votes, decision_cfg.pause_votes)
        ema_alphas = _parse_float_grid(args.sweep_ema_alpha, decision_cfg.ema_alpha)
        rows = run_sweep(traces, decision_cfg, pause_scores, pause_votes_grid, ema_alphas, args.failure_start_tick)
        print_sweep_table(rows, args.failure_start_tick)
    else:
        results = [replay_trace(trace, decision_cfg, args.failure_start_tick) for trace in traces]
        agg = aggregate(results)
        print_report(agg, args.failure_start_tick)


if __name__ == "__main__":
    main()
