"""Frame acquisition: camera device, HTTP (MJPEG/snapshot), and directory replay.

All sources implement the `FrameSource` interface and are constructed via
`build_source(cfg)`, which dispatches on `CameraConfig.source`:

  - a string of digits (e.g. "0")          -> DeviceFrameSource
                                               (cv2.VideoCapture(int(source)))
  - starts with "http://" / "https://"     -> HttpFrameSource
                                               (single snapshot or MJPEG stream)
  - an existing directory path             -> DirectoryFrameSource
                                               (replays *.jpg/*.jpeg/*.png, sorted)
  - anything else                          -> ValueError

Device and HTTP sources reconnect with exponential backoff on failure
(starting at `cfg.reconnect_backoff_s`, doubling up to
`cfg.max_reconnect_backoff_s`, reset on the next successful read) so a
dropped USB camera or a flaky webcam stream never crashes a multi-hour
monitoring run. `read()` never raises on a transient failure -- it logs a
warning and returns None.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Optional

import cv2
import numpy as np
import requests

from argus.config import CameraConfig
from argus.types import Frame

logger = logging.getLogger(__name__)

# Fixed request timeout for HTTP snapshot fetches. CameraConfig has no
# dedicated http-timeout field, so this mirrors MoonrakerConfig's default
# timeout_s (5.0) rather than inventing a new config surface.
_HTTP_SNAPSHOT_TIMEOUT_S = 5.0

# Moonraker/crowsnest convention: a webcam URL's query string carries
# "action=snapshot" for a single still image (e.g. "/webcam/?action=snapshot")
# or "action=stream" (or nothing in particular) for a repeating MJPEG stream
# (e.g. "/webcam/?action=stream"). We key off exactly that substring: if it's
# present, treat the URL as a snapshot endpoint fetched fresh on every
# read() via `requests.get` + `cv2.imdecode`; otherwise treat it as a
# streaming source opened once via `cv2.VideoCapture(url)`, which natively
# understands MJPEG multipart streams. This is a heuristic on the URL text,
# not a content negotiation -- a server using nonstandard query params would
# need its URL adjusted to match.
_SNAPSHOT_MARKER = "action=snapshot"

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class FrameSource(ABC):
    """Common interface every frame source implementation must satisfy."""

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Return the next available frame, or None if none is available right now.

        Must never raise on a transient failure (dropped connection, decode
        error, exhausted replay) -- log and return None instead.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release any underlying resources. Default is a no-op."""
        return None

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()
        return None


class _Backoff:
    """Exponential backoff schedule: starts at `initial_s`, doubles on each
    failure up to `max_s`, and resets to `initial_s` after a success."""

    def __init__(self, initial_s: float, max_s: float) -> None:
        self._initial_s = initial_s
        self._max_s = max_s
        self._current_s = initial_s
        self._retry_at = 0.0

    def ready(self) -> bool:
        """True if enough time has passed since the last failure to retry now."""
        return time.monotonic() >= self._retry_at

    def fail(self) -> None:
        """Record a failure: schedule the next retry and grow the delay."""
        self._retry_at = time.monotonic() + self._current_s
        self._current_s = min(self._current_s * 2, self._max_s)

    def reset(self) -> None:
        """Record a success: clear the schedule and reset the delay to initial."""
        self._retry_at = 0.0
        self._current_s = self._initial_s


class DeviceFrameSource(FrameSource):
    """Captures frames from a local camera device via cv2.VideoCapture(index).

    The device is opened lazily on the first `read()` (not in the
    constructor) so that simply building a source never touches hardware,
    and so the same exponential-backoff/reconnect logic governs the very
    first connection attempt as every subsequent reconnect.
    """

    def __init__(self, cfg: CameraConfig) -> None:
        self._cfg = cfg
        self._device_index = int(cfg.source)
        self._cap: Optional[cv2.VideoCapture] = None
        self._backoff = _Backoff(cfg.reconnect_backoff_s, cfg.max_reconnect_backoff_s)
        self._seq = 0

    def _open(self) -> bool:
        cap = cv2.VideoCapture(self._device_index)
        if not cap.isOpened():
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self._cfg.fps)
        self._cap = cap
        return True

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            if not self._backoff.ready():
                return None
            if not self._open():
                logger.warning(
                    "camera device %d: failed to open, backing off", self._device_index
                )
                self._backoff.fail()
                return None

        assert self._cap is not None
        try:
            ok, image = self._cap.read()
        except cv2.error as exc:
            ok, image = False, None
            logger.warning("camera device %d: read raised %s", self._device_index, exc)

        if not ok or image is None:
            logger.warning(
                "camera device %d: read failed, will reconnect", self._device_index
            )
            self._cap.release()
            self._cap = None
            self._backoff.fail()
            return None

        self._backoff.reset()
        frame = Frame(image=image, timestamp=time.time(), seq=self._seq)
        self._seq += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class HttpFrameSource(FrameSource):
    """Captures frames from an HTTP(S) camera endpoint.

    See `_SNAPSHOT_MARKER` above for the snapshot-vs-stream heuristic. Mode
    is decided once, from the URL, at construction time; nothing is fetched
    or connected until the first `read()`.
    """

    def __init__(self, cfg: CameraConfig) -> None:
        self._cfg = cfg
        self._url = cfg.source
        self._is_snapshot = _SNAPSHOT_MARKER in self._url
        self._cap: Optional[cv2.VideoCapture] = None
        self._backoff = _Backoff(cfg.reconnect_backoff_s, cfg.max_reconnect_backoff_s)
        self._seq = 0

    def read(self) -> Optional[Frame]:
        if self._is_snapshot:
            return self._read_snapshot()
        return self._read_stream()

    def _emit(self, image: np.ndarray) -> Frame:
        self._backoff.reset()
        frame = Frame(image=image, timestamp=time.time(), seq=self._seq)
        self._seq += 1
        return frame

    def _read_snapshot(self) -> Optional[Frame]:
        if not self._backoff.ready():
            return None
        try:
            resp = requests.get(self._url, timeout=_HTTP_SNAPSHOT_TIMEOUT_S)
            resp.raise_for_status()
            buf = np.frombuffer(resp.content, dtype=np.uint8)
            image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except requests.RequestException as exc:
            logger.warning("http snapshot %s: request failed (%s), backing off", self._url, exc)
            self._backoff.fail()
            return None

        if image is None:
            logger.warning("http snapshot %s: failed to decode image, backing off", self._url)
            self._backoff.fail()
            return None

        return self._emit(image)

    def _read_stream(self) -> Optional[Frame]:
        if self._cap is None:
            if not self._backoff.ready():
                return None
            cap = cv2.VideoCapture(self._url)
            if not cap.isOpened():
                cap.release()
                logger.warning("http stream %s: failed to open, backing off", self._url)
                self._backoff.fail()
                return None
            self._cap = cap

        assert self._cap is not None
        try:
            ok, image = self._cap.read()
        except cv2.error as exc:
            ok, image = False, None
            logger.warning("http stream %s: read raised %s", self._url, exc)

        if not ok or image is None:
            logger.warning("http stream %s: read failed, will reconnect", self._url)
            self._cap.release()
            self._cap = None
            self._backoff.fail()
            return None

        return self._emit(image)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class DirectoryFrameSource(FrameSource):
    """Replays image files from a directory, one per `read()`, in sorted
    filename order. Powers offline replay, --dry-run, and calibration.

    Does not loop: once every file has been served, `read()` returns None
    on every subsequent call.
    """

    def __init__(self, cfg: CameraConfig) -> None:
        directory = Path(cfg.source)
        self._files = sorted(
            (
                p
                for p in directory.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
            ),
            key=lambda p: p.name,
        )
        self._index = 0
        self._seq = 0

    def read(self) -> Optional[Frame]:
        while self._index < len(self._files):
            path = self._files[self._index]
            self._index += 1
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                logger.warning("directory replay: failed to read %s, skipping", path)
                continue
            frame = Frame(image=image, timestamp=time.time(), seq=self._seq)
            self._seq += 1
            return frame
        return None

    def close(self) -> None:
        return None


def build_source(cfg: CameraConfig) -> FrameSource:
    """Construct the `FrameSource` implementation matching `cfg.source`.

    See the module docstring for the full dispatch table.
    """
    source = cfg.source
    if source.isdigit():
        return DeviceFrameSource(cfg)
    if source.startswith("http://") or source.startswith("https://"):
        return HttpFrameSource(cfg)
    if Path(source).is_dir():
        return DirectoryFrameSource(cfg)
    raise ValueError(
        f"invalid camera.source {source!r}: expected a digit string (device index), "
        "an http(s):// URL, or an existing directory path"
    )
