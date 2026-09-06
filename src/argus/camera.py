"""Frame acquisition: camera device, HTTP (MJPEG/snapshot), and directory replay.

All sources implement `FrameSource` and are built via `build_source(cfg)`,
which dispatches on `CameraConfig.source`: digit string -> device index,
http(s):// URL -> snapshot/stream, existing directory -> replay, else
ValueError. Device/HTTP sources reconnect with exponential backoff so a
dropped camera never crashes a multi-hour run; `read()` never raises.
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

# Moonraker/crowsnest convention: "action=snapshot" in the URL means a single
# still image (fetched fresh per read via requests+cv2.imdecode); otherwise
# treated as an MJPEG stream (opened once via cv2.VideoCapture). Text
# heuristic, not content negotiation.
_SNAPSHOT_MARKER = "action=snapshot"

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class FrameSource(ABC):
    """Common interface every frame source implementation must satisfy."""

    # Optional[Frame] read(self)
    # Inputs: None
    # Outputs: Optional[Frame] - the next available frame, or None if none is available right now
    # Description: Abstract interface method: return the next captured/replayed frame.
    #              Implementations must never raise on a transient failure (dropped connection,
    #              decode error, exhausted replay) -- log and return None instead.
    # Side Effects: Implementation-defined (typically device/network/file I/O); this base
    #               declaration has none of its own.
    @abstractmethod
    def read(self) -> Optional[Frame]:
        raise NotImplementedError

    # None close(self)
    # Inputs: None
    # Outputs: None
    # Description: Default no-op resource-release hook; subclasses override to release real
    #              devices/connections.
    # Side Effects: None
    def close(self) -> None:
        return None

    # FrameSource __enter__(self)
    # Inputs: None
    # Outputs: FrameSource - self, for use as a context manager
    # Description: Enables `with build_source(cfg) as source:` usage.
    # Side Effects: None
    def __enter__(self) -> FrameSource:
        return self

    # None __exit__(self, Optional[type[BaseException]] exc_type, Optional[BaseException] exc_val, Optional[TracebackType] exc_tb)
    # Inputs: Optional[type[BaseException]] exc_type - exception type from the `with` block, if any
    #         Optional[BaseException] exc_val - exception instance from the `with` block, if any
    #         Optional[TracebackType] exc_tb - traceback from the `with` block, if any
    #         (none of the three are inspected; they exist only to satisfy the context-manager
    #         protocol)
    # Outputs: None
    # Description: Context-manager exit hook that ensures resources are released.
    # Side Effects: Calls self.close() (effects depend on the concrete subclass).
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

    # None __init__(self, float initial_s, float max_s)
    # Inputs: float initial_s - starting/reset backoff delay in seconds
    #         float max_s - ceiling the backoff delay doubles up to
    # Outputs: None
    # Description: Initializes the backoff schedule to its starting delay with no pending retry.
    # Side Effects: None
    def __init__(self, initial_s: float, max_s: float) -> None:
        self._initial_s = initial_s
        self._max_s = max_s
        self._current_s = initial_s
        self._retry_at = 0.0

    # bool ready(self)
    # Inputs: None
    # Outputs: bool - True if enough time has passed since the last failure to retry now
    # Description: Checks the backoff schedule against the current monotonic clock.
    # Side Effects: Reads the monotonic clock (time.monotonic()).
    def ready(self) -> bool:
        return time.monotonic() >= self._retry_at

    # None fail(self)
    # Inputs: None
    # Outputs: None
    # Description: Records a failure by scheduling the next retry and doubling the delay (capped
    #              at max_s).
    # Side Effects: Reads the monotonic clock (time.monotonic()). Mutates self._retry_at and
    #               self._current_s.
    def fail(self) -> None:
        self._retry_at = time.monotonic() + self._current_s
        self._current_s = min(self._current_s * 2, self._max_s)

    # None reset(self)
    # Inputs: None
    # Outputs: None
    # Description: Records a success by clearing the pending retry and resetting the delay to
    #              initial_s.
    # Side Effects: Mutates self._retry_at and self._current_s.
    def reset(self) -> None:
        self._retry_at = 0.0
        self._current_s = self._initial_s


class DeviceFrameSource(FrameSource):
    """Captures frames from a local camera device via cv2.VideoCapture(index).

    The device is opened lazily on the first `read()` (not in the
    constructor) so that simply building a source never touches hardware,
    and so the same exponential-backoff/reconnect logic governs the very
    first connection attempt as every subsequent reconnect.
    """

    # None __init__(self, CameraConfig cfg)
    # Inputs: CameraConfig cfg - source (device index as digit string), width/height/fps, and
    #                             reconnect backoff bounds
    # Outputs: None
    # Description: Stores config, resolves the device index, and sets up the backoff schedule.
    #              Does not open the device -- that happens lazily on the first read().
    # Side Effects: None (no hardware access at construction time).
    def __init__(self, cfg: CameraConfig) -> None:
        self._cfg = cfg
        self._device_index = int(cfg.source)
        self._cap: Optional[cv2.VideoCapture] = None
        self._backoff = _Backoff(cfg.reconnect_backoff_s, cfg.max_reconnect_backoff_s)
        self._seq = 0

    # bool _open(self)
    # Inputs: None
    # Outputs: bool - True if the device opened successfully, else False
    # Description: Opens the configured camera device index and applies the configured
    #              width/height/fps, storing the resulting capture handle on success.
    # Side Effects: Opens (and, on failure, releases) a cv2.VideoCapture device handle -- real
    #               hardware access. Mutates self._cap on success.
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

    # Optional[Frame] read(self)
    # Inputs: None
    # Outputs: Optional[Frame] - the next captured frame, or None if the device is not yet ready
    #                            to retry, failed to open, or a read failed
    # Description: Lazily opens the device on first use (subject to backoff), reads one frame,
    #              and reconnects on failure per the exponential-backoff schedule.
    # Side Effects: Opens/releases the camera device handle on (re)connect. Reads a frame from
    #               real hardware (self._cap.read()). Reads the wall clock (time.time()) to
    #               timestamp the frame. Mutates self._cap, self._seq, and the backoff state.
    #               Logs a warning on any failure. Never raises.
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

    # None close(self)
    # Inputs: None
    # Outputs: None
    # Description: Releases the camera device handle, if one is open.
    # Side Effects: Releases the cv2.VideoCapture device handle (real hardware access). Mutates
    #               self._cap to None.
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

    # None __init__(self, CameraConfig cfg)
    # Inputs: CameraConfig cfg - source (an http(s):// URL) and reconnect backoff bounds
    # Outputs: None
    # Description: Stores config, decides snapshot-vs-stream mode once from the URL text (see
    #              `_SNAPSHOT_MARKER`), and sets up the backoff schedule. Nothing is fetched or
    #              connected yet -- that happens lazily on the first read().
    # Side Effects: None (no network access at construction time).
    def __init__(self, cfg: CameraConfig) -> None:
        self._cfg = cfg
        self._url = cfg.source
        self._is_snapshot = _SNAPSHOT_MARKER in self._url
        self._cap: Optional[cv2.VideoCapture] = None
        self._backoff = _Backoff(cfg.reconnect_backoff_s, cfg.max_reconnect_backoff_s)
        self._seq = 0

    # Optional[Frame] read(self)
    # Inputs: None
    # Outputs: Optional[Frame] - the next available frame, or None if none is available right now
    # Description: Dispatches to `_read_snapshot` or `_read_stream` based on the mode decided at
    #              construction time.
    # Side Effects: Delegates to `_read_snapshot`/`_read_stream` (see those for their effects).
    def read(self) -> Optional[Frame]:
        if self._is_snapshot:
            return self._read_snapshot()
        return self._read_stream()

    # Frame _emit(self, np.ndarray image)
    # Inputs: np.ndarray image - decoded BGR image to wrap as a Frame
    # Outputs: Frame - the wrapped frame, timestamped now with the next sequence number
    # Description: Shared "successful read" tail: resets the backoff and stamps a new Frame.
    # Side Effects: Reads the wall clock (time.time()). Mutates self._seq and resets the backoff
    #               state.
    def _emit(self, image: np.ndarray) -> Frame:
        self._backoff.reset()
        frame = Frame(image=image, timestamp=time.time(), seq=self._seq)
        self._seq += 1
        return frame

    # Optional[Frame] _read_snapshot(self)
    # Inputs: None
    # Outputs: Optional[Frame] - the fetched-and-decoded frame, or None if not yet ready to
    #                            retry, the HTTP request failed, or the response could not be
    #                            decoded as an image
    # Description: Fetches a single still image via HTTP GET and decodes it, per the
    #              snapshot-endpoint convention (subject to the backoff schedule).
    # Side Effects: Issues an HTTP GET to `self._url`. Mutates the backoff state (fail/reset) and
    #               self._seq. Logs a warning on any failure. Never raises.
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

    # Optional[Frame] _read_stream(self)
    # Inputs: None
    # Outputs: Optional[Frame] - the next frame read from the MJPEG stream, or None if not yet
    #                            ready to retry, the stream failed to open, or a read failed
    # Description: Lazily opens the MJPEG stream on first use (subject to backoff), reads one
    #              frame, and reconnects on failure per the exponential-backoff schedule.
    # Side Effects: Opens/releases a cv2.VideoCapture stream connection (network I/O). Reads a
    #               frame from the stream (self._cap.read()). Mutates self._cap, self._seq, and
    #               the backoff state. Logs a warning on any failure. Never raises.
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

    # None close(self)
    # Inputs: None
    # Outputs: None
    # Description: Releases the stream connection, if one is open (no-op in snapshot mode, which
    #              never holds a persistent connection).
    # Side Effects: Releases the cv2.VideoCapture stream handle (network resource). Mutates
    #               self._cap to None.
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

    # None __init__(self, CameraConfig cfg)
    # Inputs: CameraConfig cfg - source (an existing directory path)
    # Outputs: None
    # Description: Enumerates and sorts (by filename) every *.jpg/*.jpeg/*.png file in
    #              `cfg.source` up front, to be replayed one per read().
    # Side Effects: Lists the contents of the source directory on disk.
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

    # Optional[Frame] read(self)
    # Inputs: None
    # Outputs: Optional[Frame] - the next image file's contents as a Frame, or None once every
    #                            file has been served (replay does not loop)
    # Description: Serves image files from the pre-enumerated list in sorted order, skipping any
    #              that fail to decode.
    # Side Effects: Reads image files from disk (cv2.imread). Reads the wall clock (time.time())
    #               to timestamp each frame. Mutates self._index and self._seq. Logs a warning
    #               for any file that fails to decode.
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

    # None close(self)
    # Inputs: None
    # Outputs: None
    # Description: No-op -- this source holds no resources to release (no open device/connection).
    # Side Effects: None
    def close(self) -> None:
        return None


# FrameSource build_source(CameraConfig cfg)
# Inputs: CameraConfig cfg - camera config; `cfg.source` selects the concrete implementation
# Outputs: FrameSource - a DeviceFrameSource, HttpFrameSource, or DirectoryFrameSource, matching
#                        `cfg.source`
# Description: Factory that dispatches on `cfg.source`'s form (digit string -> device index,
#              http(s):// URL -> HTTP source, existing directory -> replay source) per the module
#              docstring's dispatch table.
# Side Effects: Calls Path(source).is_dir(), which touches the filesystem. Raises ValueError if
#               `cfg.source` matches none of the recognised forms. Constructing the chosen
#               subclass has its own side effects (see that class's __init__).
def build_source(cfg: CameraConfig) -> FrameSource:
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
