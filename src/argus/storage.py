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


# float _safe_mtime(Path path)
# Inputs: Path path - filesystem path to stat
# Outputs: float - path's mtime, or 0.0 (treated as "oldest") if it can't be stat'd
# Description: Best-effort mtime lookup used to sort/prune files by age.
# Side Effects: Reads filesystem metadata for `path` (stat call).
def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


class EventStore:
    """Append-only JSONL event log plus a pruned JPEG frame archive."""

    # None __init__(self, StorageConfig cfg)
    # Inputs: StorageConfig cfg - event log path, frame directory, and pruning limits
    # Outputs: None
    # Description: Stores the config and resolves the event-log and frame-directory paths.
    # Side Effects: None (paths are not created or touched until write_event/save_frame runs).
    def __init__(self, cfg: StorageConfig) -> None:
        self._cfg = cfg
        self._event_log = Path(cfg.event_log)
        self._frame_dir = Path(cfg.frame_dir)

    # None write_event(self, Event event)
    # Inputs: Event event - the event to append to the log
    # Outputs: None
    # Description: Serializes `event` to one JSON line and appends it to the event log.
    # Side Effects: Creates the event log's parent directories if missing. Opens the event log
    #               file in append mode, writes one line, and flushes -- a crash mid-write can at
    #               worst truncate the newest line, never previously written ones.
    def write_event(self, event: Event) -> None:
        self._event_log.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.to_dict()) + "\n"
        with open(self._event_log, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()

    # Optional[str] save_frame(self, Frame frame, float event_ts)
    # Inputs: Frame frame - the frame image to persist
    #         float event_ts - unix timestamp (UTC) used to name the saved file
    # Outputs: Optional[str] - the saved file path as a string, or None on any failure
    # Description: Writes `frame`'s image as a JPEG into cfg.frame_dir, named from `event_ts`.
    # Side Effects: Creates the frame directory if missing. Writes one JPEG file to disk via
    #               cv2.imwrite. Logs a warning/exception on failure. Never raises.
    def save_frame(self, frame: Frame, event_ts: float) -> Optional[str]:
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

    # int prune(self)
    # Inputs: None
    # Outputs: int - count of files removed
    # Description: Deletes frames older than cfg.retention_days and, of what remains, any beyond
    #              the newest cfg.max_frames, keeping disk use bounded on the Pi.
    # Side Effects: Lists the frame directory. Permanently deletes frame files from disk (via
    #               Path.unlink). Logs a warning for any file that fails to delete.
    def prune(self) -> int:
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

    # list[dict] read_events(self, Optional[int] limit=None)
    # Inputs: Optional[int] limit - if given, return only the most recent `limit` events;
    #                                defaults to None (return all events)
    # Outputs: list[dict] - parsed event dicts in file order, most-recent-`limit` if truncated
    # Description: Reads and JSON-parses every line of the event log, skipping malformed/partial
    #              lines, and optionally trims to the most recent `limit` entries.
    # Side Effects: Reads the entire event log file from disk. Logs a warning for each malformed
    #               line encountered.
    def read_events(self, limit: Optional[int] = None) -> list[dict]:
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
