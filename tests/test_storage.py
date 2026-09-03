"""Tests for argus.storage: the JSONL event log and JPEG frame archive.

Uses tmp_path only -- no real camera, network, or shared state.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from argus.config import StorageConfig
from argus.storage import EventStore
from argus.types import Action, Detection, DecisionState, Event, Frame, Severity


def _cfg(tmp_path: Path, **overrides) -> StorageConfig:
    base = StorageConfig(
        event_log=str(tmp_path / "logs" / "events.jsonl"),
        frame_dir=str(tmp_path / "captures"),
    )
    return replace(base, **overrides)


def _event(timestamp: float = 1000.0, reason: str = "ok") -> Event:
    detection = Detection(
        class_id=0,
        class_name="spaghetti",
        confidence=0.91,
        bbox=(1.0, 2.0, 3.0, 4.0),
        severity=Severity.CATASTROPHIC,
    )
    return Event(
        timestamp=timestamp,
        action=Action.NOTIFY,
        state=DecisionState.WARNING,
        score=0.8,
        p_raw=0.9,
        votes=5,
        reason=reason,
        class_name="spaghetti",
        confidence=0.91,
        frame_path=None,
        print_filename="benchy.gcode",
        elapsed_s=120.0,
        detections=(detection,),
    )


def _frame(size: int = 16) -> Frame:
    image = np.random.default_rng(0).integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Frame(image=image, timestamp=time.time(), seq=0)


# --------------------------------------------------------------------------
# write_event / read_events
# --------------------------------------------------------------------------


def test_write_then_read_event_round_trips(tmp_path: Path):
    store = EventStore(_cfg(tmp_path))
    event = _event()
    store.write_event(event)

    events = store.read_events()
    assert len(events) == 1
    assert events[0] == event.to_dict()


def test_write_event_creates_parent_dirs(tmp_path: Path):
    store = EventStore(_cfg(tmp_path))
    log_path = Path(_cfg(tmp_path).event_log)
    assert not log_path.parent.exists()
    store.write_event(_event())
    assert log_path.exists()


def test_multiple_appends_preserve_order(tmp_path: Path):
    store = EventStore(_cfg(tmp_path))
    for i in range(5):
        store.write_event(_event(timestamp=1000.0 + i, reason=f"reason-{i}"))

    events = store.read_events()
    assert [e["reason"] for e in events] == [f"reason-{i}" for i in range(5)]


def test_read_events_respects_limit(tmp_path: Path):
    store = EventStore(_cfg(tmp_path))
    for i in range(5):
        store.write_event(_event(timestamp=1000.0 + i, reason=f"reason-{i}"))

    events = store.read_events(limit=2)
    assert [e["reason"] for e in events] == ["reason-3", "reason-4"]


def test_read_events_on_missing_file_returns_empty_list(tmp_path: Path):
    store = EventStore(_cfg(tmp_path))
    assert store.read_events() == []


def test_read_events_skips_malformed_lines(tmp_path: Path):
    store = EventStore(_cfg(tmp_path))
    store.write_event(_event(reason="good-1"))

    log_path = Path(_cfg(tmp_path).event_log)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("{not valid json\n")
        f.write("\n")  # blank line should also be skipped, not raise

    store.write_event(_event(reason="good-2"))

    events = store.read_events()
    assert [e["reason"] for e in events] == ["good-1", "good-2"]


def test_write_event_output_is_one_json_object_per_line(tmp_path: Path):
    store = EventStore(_cfg(tmp_path))
    store.write_event(_event(reason="a"))
    store.write_event(_event(reason="b"))

    log_path = Path(_cfg(tmp_path).event_log)
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # must not raise


# --------------------------------------------------------------------------
# save_frame
# --------------------------------------------------------------------------


def test_save_frame_writes_readable_jpeg(tmp_path: Path):
    store = EventStore(_cfg(tmp_path))
    frame = _frame()
    path = store.save_frame(frame, event_ts=1735689600.123)  # 2025-01-01T00:00:00.123Z

    assert path is not None
    saved = Path(path)
    assert saved.exists()
    assert saved.suffix == ".jpg"

    loaded = cv2.imread(str(saved), cv2.IMREAD_COLOR)
    assert loaded is not None
    assert loaded.shape == frame.image.shape


def test_save_frame_creates_frame_dir(tmp_path: Path):
    store = EventStore(_cfg(tmp_path))
    frame_dir = Path(_cfg(tmp_path).frame_dir)
    assert not frame_dir.exists()
    store.save_frame(_frame(), event_ts=1735689600.0)
    assert frame_dir.exists()


def test_save_frame_name_encodes_utc_timestamp(tmp_path: Path):
    from datetime import datetime, timezone

    store = EventStore(_cfg(tmp_path))
    dt = datetime(2026, 9, 1, 14, 30, 22, 123000, tzinfo=timezone.utc)
    path = store.save_frame(_frame(), event_ts=dt.timestamp())
    assert path is not None
    assert "20260901_143022_123" in path


# --------------------------------------------------------------------------
# prune
# --------------------------------------------------------------------------


def test_prune_empty_dir_returns_zero(tmp_path: Path):
    cfg = _cfg(tmp_path)
    Path(cfg.frame_dir).mkdir(parents=True)
    store = EventStore(cfg)
    assert store.prune() == 0


def test_prune_missing_dir_returns_zero(tmp_path: Path):
    store = EventStore(_cfg(tmp_path))
    assert store.prune() == 0


def test_prune_respects_max_frames(tmp_path: Path):
    cfg = _cfg(tmp_path, max_frames=3, retention_days=365)
    frame_dir = Path(cfg.frame_dir)
    frame_dir.mkdir(parents=True)

    paths = []
    now = time.time()
    for i in range(5):
        p = frame_dir / f"frame_{i}.jpg"
        cv2.imwrite(str(p), np.full((4, 4, 3), i, dtype=np.uint8))
        # Force distinct, increasing mtimes regardless of filesystem
        # timestamp resolution: oldest is index 0, newest is index 4.
        os.utime(p, (now - (5 - i), now - (5 - i)))
        paths.append(p)

    store = EventStore(cfg)
    removed = store.prune()

    assert removed == 2
    remaining = {p.name for p in frame_dir.iterdir()}
    assert remaining == {"frame_2.jpg", "frame_3.jpg", "frame_4.jpg"}


def test_prune_respects_retention_days(tmp_path: Path):
    cfg = _cfg(tmp_path, max_frames=1000, retention_days=7)
    frame_dir = Path(cfg.frame_dir)
    frame_dir.mkdir(parents=True)

    now = time.time()
    old_path = frame_dir / "old.jpg"
    new_path = frame_dir / "new.jpg"
    cv2.imwrite(str(old_path), np.zeros((4, 4, 3), dtype=np.uint8))
    cv2.imwrite(str(new_path), np.zeros((4, 4, 3), dtype=np.uint8))

    old_mtime = now - 10 * 86400  # 10 days old, older than retention_days=7
    new_mtime = now - 1 * 86400  # 1 day old, within retention
    os.utime(old_path, (old_mtime, old_mtime))
    os.utime(new_path, (new_mtime, new_mtime))

    store = EventStore(cfg)
    removed = store.prune()

    assert removed == 1
    remaining = {p.name for p in frame_dir.iterdir()}
    assert remaining == {"new.jpg"}


def test_prune_returns_count_removed(tmp_path: Path):
    cfg = _cfg(tmp_path, max_frames=0, retention_days=365)
    frame_dir = Path(cfg.frame_dir)
    frame_dir.mkdir(parents=True)

    for i in range(3):
        p = frame_dir / f"f{i}.jpg"
        cv2.imwrite(str(p), np.zeros((4, 4, 3), dtype=np.uint8))

    store = EventStore(cfg)
    removed = store.prune()

    assert removed == 3
    assert list(frame_dir.iterdir()) == []
