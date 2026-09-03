"""Record camera frames from a real print, for building a personal dataset.

This is the main lever for real-world accuracy: the shipped model is
trained on a public dataset, but every printer's camera angle, lighting,
and bed/plate look a little different. Running this during a real print
(ideally several, covering both nominal prints and any real failures you
witness) builds up a `datasets/raw`-shaped pile of frames to label and feed
into `training/prepare_dataset.py` / `training/train.py` for fine-tuning.

Frames are captured from `cfg.camera` (same config, same `build_source`
dispatch, as the live service) at a fixed wall-clock interval and written
as timestamped JPEGs into `--out`. Pass `--only-while-printing` to gate
capture on Moonraker's print state so idle/homing/bed-mesh frames -- which
are not what the detector needs to learn from -- don't dilute the dataset.

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


def _frame_filename(ts: float) -> str:
    """UTC-timestamped filename, millisecond precision, sortable by name."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return f"frame_{dt.strftime('%Y%m%dT%H%M%S')}_{dt.microsecond // 1000:03d}Z.jpg"


def _save_frame(frame: Frame, out_dir: Path) -> Optional[Path]:
    """Write `frame` as a JPEG into `out_dir`. Returns the path, or None on
    failure -- a single bad write must never abort a multi-hour recording
    session."""
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
    """Small mutable holder so the SIGINT/SIGTERM handler can signal the
    capture loop to stop after the current frame without module-level
    global state."""

    def __init__(self) -> None:
        self.stop = False

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
