"""Event log (JSONL) and captured-frame archive.

Both are designed to fail soft: a storage hiccup (full disk, permissions,
transient I/O error) must never crash the monitoring loop or corrupt
previously-written data. `prune()` bounds disk use on the Pi, where a full
SD card would take down Klipper itself, not just Argus.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2

from argus.config import StorageConfig
from argus.types import Event, Frame

logger = logging.getLogger(__name__)


def _safe_mtime(path: Path) -> float:
    """`path`'s mtime, or 0.0 (treated as "oldest") if it can't be stat'd."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


class EventStore:
    """Append-only JSONL event log plus a pruned JPEG frame archive."""

    def __init__(self, cfg: StorageConfig) -> None:
        self._cfg = cfg
        self._event_log = Path(cfg.event_log)
        self._frame_dir = Path(cfg.frame_dir)

    def write_event(self, event: Event) -> None:
        """Append one JSON object (one line) to the event log.

        Opens in append mode and writes+flushes exactly once, so a crash
        mid-write can at worst truncate the newest line -- previously
        written lines are never touched.
        """
        self._event_log.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.to_dict()) + "\n"
        with open(self._event_log, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()

    def save_frame(self, frame: Frame, event_ts: float) -> Optional[str]:
        """Write `frame` as a JPEG into `cfg.frame_dir`, named by `event_ts` (UTC).

        Returns the saved path, or None on any failure -- losing a debug
        image must never break the monitoring loop.
        """
        try:
            self._frame_dir.mkdir(parents=True, exist_ok=True)
            dt = datetime.fromtimestamp(event_ts, tz=timezone.utc)
            name = f"frame_{dt.strftime('%Y%m%d_%H%M%S')}_{dt.microsecond // 1000:03d}.jpg"
            path = self._frame_dir / name
            ok = cv2.imwrite(str(path), frame.image)
            if not ok:
                logger.warning("failed to write frame to %s", path)
                return None
            return str(path)
        except Exception:
            logger.exception("failed to save frame")
            return None

    def prune(self) -> int:
        """Delete frames older than `cfg.retention_days` and, of what
        remains, any beyond the newest `cfg.max_frames`. Returns the count
        of files removed."""
        if not self._frame_dir.is_dir():
            return 0

        files = [p for p in self._frame_dir.iterdir() if p.is_file()]
        removed = 0

        cutoff = datetime.now().timestamp() - self._cfg.retention_days * 86400
        kept: list[Path] = []
        for p in files:
            if _safe_mtime(p) < cutoff:
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    logger.warning("failed to prune %s", p)
                    kept.append(p)
            else:
                kept.append(p)
        files = kept

        if len(files) > self._cfg.max_frames:
            newest_first = sorted(files, key=_safe_mtime, reverse=True)
            for p in newest_first[self._cfg.max_frames :]:
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    logger.warning("failed to prune %s", p)

        return removed

    def read_events(self, limit: Optional[int] = None) -> list[dict]:
        """Read parsed events from the log, skipping malformed/partial lines.

        If `limit` is given, returns only the most recent `limit` events.
        """
        if not self._event_log.is_file():
            return []
        events: list[dict] = []
        with open(self._event_log, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("skipping malformed event log line")
                    continue
        if limit is not None:
            events = events[-limit:]
        return events
