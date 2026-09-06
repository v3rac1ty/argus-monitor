"""Record camera frames from a real print, for building a personal dataset
to fine-tune the detector on this printer's actual camera/lighting/bed.

Captures from `cfg.camera` at a fixed interval, writes timestamped JPEGs
into `--out`. `--only-while-printing` gates capture on Moonraker's print
state to skip idle/homing/bed-mesh frames.

Usage:
    python tools/record.py --out datasets/captures/2026-09-01_benchy --interval 5
    python tools/record.py --config config.yaml --out datasets/captures/run1 \\
        --interval 3 --only-while-printing --max-frames 500
"""

from __future__ import annotations

import argparse
import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Optional

import cv2

from argus.camera import build_source
from argus.config import Config, load_config
from argus.moonraker import MoonrakerClient
from argus.types import Frame

logger = logging.getLogger("argus.tools.record")


# str _frame_filename(float ts)
# Inputs: float ts - unix timestamp (seconds) of the frame to name
# Outputs: str - UTC-timestamped JPEG filename, millisecond precision
# Description: Builds the on-disk filename for a captured frame from its capture timestamp,
#              formatting it so filenames sort chronologically by name.
# Side Effects: None
def _frame_filename(ts: float) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return f"frame_{dt.strftime('%Y%m%dT%H%M%S')}_{dt.microsecond // 1000:03d}Z.jpg"


# Optional[Path] _save_frame(Frame frame, Path out_dir)
# Inputs: Frame frame   - captured frame (image + timestamp) to persist
#         Path out_dir  - directory to write the JPEG into
# Outputs: Optional[Path] - path to the written JPEG, or None if the write failed
# Description: Writes a single frame to disk as a timestamped JPEG, tolerating and logging
#              failures so one bad write never aborts a multi-hour recording session.
# Side Effects: Creates `out_dir` (and parents) if missing; writes a JPEG file to disk; logs a
#               warning or exception on failure.
def _save_frame(frame: Frame, out_dir: Path) -> Optional[Path]:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / _frame_filename(frame.timestamp)
        ok = cv2.imwrite(str(path), frame.image)
        if not ok:
            logger.warning("failed to write frame to %s", path)
            return None
        return path
    except OSError:
        logger.exception("failed to save frame")
        return None


class _StopFlag:
    """Mutable holder so the SIGINT/SIGTERM handler can signal the capture
    loop to stop, without module-level global state."""

    # None __init__()
    # Inputs: None
    # Outputs: None
    # Description: Initializes the mutable stop flag to False.
    # Side Effects: None
    def __init__(self) -> None:
        self.stop = False

    # None install_handlers()
    # Inputs: None
    # Outputs: None
    # Description: Registers SIGINT/SIGTERM handlers that set `self.stop` so the capture loop
    #              exits after finishing its current frame instead of terminating abruptly.
    # Side Effects: Installs process-level signal handlers for SIGINT and SIGTERM (where
    #               available); logs a debug message if a handler cannot be installed.
    def install_handlers(self) -> None:
        def _handler(signum: int, frame: Optional[FrameType]) -> None:
            logger.info("received signal %s, stopping after current frame", signum)
            self.stop = True

        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                logger.debug("could not install handler for %s", sig_name)


# argparse.Namespace parse_args(Optional[list[str]] argv=None)
# Inputs: Optional[list[str]] argv - command-line arguments to parse; defaults to None, which
#                                    makes argparse read sys.argv
# Outputs: argparse.Namespace - parsed CLI options (--config, --out, --interval, --max-frames,
#                                --only-while-printing, --log-level)
# Description: Defines and parses the command-line interface for the recording tool.
# Side Effects: None
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record camera frames at a fixed interval for dataset building.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML (default: config.example.yaml). Only camera.* and "
             "moonraker.* sections are used.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory for captured frames")
    parser.add_argument(
        "--interval", type=float, default=5.0,
        help="Seconds between captures (default: 5.0)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Stop after this many frames have been saved (default: unlimited -- run until Ctrl+C)",
    )
    parser.add_argument(
        "--only-while-printing", action="store_true",
        help="Only capture while Moonraker reports the printer is actively printing "
             "(skips homing/bed-mesh/idle frames)",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (default: INFO)")
    return parser.parse_args(argv)


# None main(Optional[list[str]] argv=None)
# Inputs: Optional[list[str]] argv - command-line arguments to parse; defaults to None (reads
#                                    sys.argv)
# Outputs: None
# Description: Entry point that drives the recording loop -- builds the camera source (and
#              optional Moonraker client), captures frames at a fixed interval (gated on print
#              state when requested), saves them to disk, and logs a final summary.
# Side Effects: Configures logging; opens the camera/HTTP video source and, when
#               --only-while-printing is set, an HTTP connection to Moonraker; polls Moonraker
#               for print state; writes JPEG frames to disk (via _save_frame); prints/logs
#               progress and a final summary; sleeps between capture ticks; closes the camera
#               source and Moonraker client on exit.
def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    cfg: Config = load_config(args.config)
    source = build_source(cfg.camera)
    moonraker: Optional[MoonrakerClient] = (
        MoonrakerClient(cfg.moonraker) if args.only_while_printing else None
    )

    stop_flag = _StopFlag()
    stop_flag.install_handlers()

    saved = 0
    skipped_not_printing = 0
    skipped_no_frame = 0

    logger.info(
        "recording to '%s' every %.1fs (max_frames=%s, only_while_printing=%s, source=%s)",
        args.out, args.interval, args.max_frames, args.only_while_printing, cfg.camera.source,
    )

    try:
        while not stop_flag.stop:
            tick_start = time.time()

            eligible = True
            if moonraker is not None:
                print_state = moonraker.get_print_state()
                eligible = print_state.is_printing
                if not eligible:
                    skipped_not_printing += 1
                    logger.debug(
                        "skipping capture: printer state=%s (not printing)", print_state.state.value
                    )

            if eligible:
                frame = source.read()
                if frame is not None:
                    path = _save_frame(frame, args.out)
                    if path is not None:
                        saved += 1
                        logger.info("saved %s (%d total)", path.name, saved)
                else:
                    skipped_no_frame += 1
                    logger.debug("skipping capture: no frame available")

            if args.max_frames is not None and saved >= args.max_frames:
                logger.info("reached --max-frames=%d, stopping", args.max_frames)
                break

            elapsed = time.time() - tick_start
            sleep_for = args.interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        source.close()
        if moonraker is not None:
            moonraker.close()
        logger.info(
            "done: saved=%d skipped_not_printing=%d skipped_no_frame=%d -> '%s'",
            saved, skipped_not_printing, skipped_no_frame, args.out,
        )


if __name__ == "__main__":
    main()
