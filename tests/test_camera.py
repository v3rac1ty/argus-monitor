"""Tests for argus.camera: source dispatch and directory replay.

Only DirectoryFrameSource is ever `read()` from -- Device/Http dispatch is
checked by isinstance only, never by opening a real device or URL.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from argus.camera import (
    DeviceFrameSource,
    DirectoryFrameSource,
    HttpFrameSource,
    build_source,
)
from argus.config import CameraConfig


def _cfg(source: str) -> CameraConfig:
    return replace(CameraConfig(), source=source)


# --------------------------------------------------------------------------
# build_source dispatch
# --------------------------------------------------------------------------


def test_dispatch_digits_gives_device_source():
    source = build_source(_cfg("0"))
    assert isinstance(source, DeviceFrameSource)


def test_dispatch_multi_digit_string_gives_device_source():
    source = build_source(_cfg("42"))
    assert isinstance(source, DeviceFrameSource)


def test_dispatch_http_url_gives_http_source():
    source = build_source(_cfg("http://192.168.1.50/webcam/?action=stream"))
    assert isinstance(source, HttpFrameSource)


def test_dispatch_https_url_gives_http_source():
    source = build_source(_cfg("https://example.com/webcam/?action=stream"))
    assert isinstance(source, HttpFrameSource)


def test_dispatch_snapshot_url_is_flagged_as_snapshot_mode():
    source = build_source(_cfg("http://192.168.1.50/webcam/?action=snapshot"))
    assert isinstance(source, HttpFrameSource)
    assert source._is_snapshot is True


def test_dispatch_stream_url_is_not_flagged_as_snapshot_mode():
    source = build_source(_cfg("http://192.168.1.50/webcam/?action=stream"))
    assert isinstance(source, HttpFrameSource)
    assert source._is_snapshot is False


def test_dispatch_directory_gives_directory_source(tmp_path: Path):
    source = build_source(_cfg(str(tmp_path)))
    assert isinstance(source, DirectoryFrameSource)


def test_dispatch_invalid_source_raises_value_error():
    with pytest.raises(ValueError):
        build_source(_cfg("not-a-real-path-or-url-or-device-xyz"))


def test_dispatch_negative_number_raises_value_error():
    # "-1" is not a digit string per str.isdigit() (the leading "-" isn't a
    # digit) and is not an existing directory either.
    with pytest.raises(ValueError):
        build_source(_cfg("-1"))


# --------------------------------------------------------------------------
# DirectoryFrameSource replay
# --------------------------------------------------------------------------


def _write_jpeg(path: Path, fill: int) -> None:
    image = np.full((16, 16, 3), fill, dtype=np.uint8)
    ok = cv2.imwrite(str(path), image)
    assert ok


def test_directory_source_replays_in_sorted_filename_order(tmp_path: Path):
    # Deliberately create out of sorted order to prove read() sorts by name,
    # not by creation time.
    _write_jpeg(tmp_path / "b_frame.jpg", fill=100)
    _write_jpeg(tmp_path / "a_frame.jpg", fill=50)
    _write_jpeg(tmp_path / "c_frame.png", fill=150)

    source = build_source(_cfg(str(tmp_path)))
    assert isinstance(source, DirectoryFrameSource)

    frame1 = source.read()
    frame2 = source.read()
    frame3 = source.read()

    assert frame1 is not None and frame2 is not None and frame3 is not None
    # a_frame.jpg < b_frame.jpg < c_frame.png in sorted order
    assert frame1.image[0, 0, 0] == 50
    assert frame2.image[0, 0, 0] == 100
    assert frame3.image[0, 0, 0] == 150


def test_directory_source_increments_seq(tmp_path: Path):
    _write_jpeg(tmp_path / "1.jpg", fill=10)
    _write_jpeg(tmp_path / "2.jpg", fill=20)

    source = build_source(_cfg(str(tmp_path)))
    frame1 = source.read()
    frame2 = source.read()

    assert frame1 is not None and frame2 is not None
    assert frame1.seq == 0
    assert frame2.seq == 1


def test_directory_source_returns_none_when_exhausted(tmp_path: Path):
    _write_jpeg(tmp_path / "only.jpg", fill=10)

    source = build_source(_cfg(str(tmp_path)))
    assert source.read() is not None
    assert source.read() is None
    # Repeated calls after exhaustion keep returning None (no looping).
    assert source.read() is None


def test_directory_source_empty_dir_returns_none_immediately(tmp_path: Path):
    source = build_source(_cfg(str(tmp_path)))
    assert source.read() is None


def test_directory_source_ignores_non_image_files(tmp_path: Path):
    _write_jpeg(tmp_path / "real.jpg", fill=42)
    (tmp_path / "notes.txt").write_text("not an image")

    source = build_source(_cfg(str(tmp_path)))
    frame = source.read()
    assert frame is not None
    assert source.read() is None


def test_directory_source_sets_recent_timestamp(tmp_path: Path):
    import time

    _write_jpeg(tmp_path / "only.jpg", fill=10)
    source = build_source(_cfg(str(tmp_path)))

    before = time.time()
    frame = source.read()
    after = time.time()

    assert frame is not None
    assert before <= frame.timestamp <= after


def test_directory_source_context_manager_closes_cleanly(tmp_path: Path):
    _write_jpeg(tmp_path / "only.jpg", fill=10)
    with build_source(_cfg(str(tmp_path))) as source:
        assert source.read() is not None
    # close() on a directory source is a no-op; just verify no exception.
