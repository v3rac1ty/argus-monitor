"""Moonraker (Klipper) REST API client.

This client is the only thing standing between an automated decision and a
consequential, user-visible action -- pausing or cancelling someone's print.
Two safety properties follow directly from that:

  1. `get_print_state` must NEVER raise. On any failure (connection error,
     timeout, non-2xx status, malformed JSON, missing keys) it returns a
     `PrintState` with `state=PrinterState.UNKNOWN`. Downstream, the
     DecisionEngine's gate treats UNKNOWN as "cannot confirm we are
     printing" and fails closed -- no action fires. When in doubt, do
     nothing.
  2. `pause`/`cancel`/`health` also never raise; they return False on any
     failure so a flaky Moonraker connection can never crash the monitoring
     loop.
"""

from __future__ import annotations

import logging
import time
from typing import Optional
from urllib.parse import quote

import requests

from argus.config import MoonrakerConfig
from argus.types import PrinterState, PrintState

logger = logging.getLogger(__name__)

# Moonraker's print_stats.state values -> our coarse PrinterState. Anything
# not in this map (including states Moonraker adds in the future) resolves
# to PrinterState.UNKNOWN rather than being guessed at.
_STATE_MAP: dict[str, PrinterState] = {
    "printing": PrinterState.PRINTING,
    "paused": PrinterState.PAUSED,
    "complete": PrinterState.COMPLETE,
    "cancelled": PrinterState.CANCELLED,
    "error": PrinterState.ERROR,
    "standby": PrinterState.STANDBY,
}


def _unknown_print_state() -> PrintState:
    """The safe fallback returned whenever we cannot confirm real state."""
    return PrintState(
        state=PrinterState.UNKNOWN,
        filename=None,
        elapsed_s=0.0,
        progress=0.0,
        fetched_at=time.time(),
    )


class MoonrakerClient:
    """Thin, defensive REST client for a single Moonraker instance."""

    def __init__(self, cfg: MoonrakerConfig, session: Optional[requests.Session] = None) -> None:
        self._cfg = cfg
        self._session = session if session is not None else requests.Session()

    def get_print_state(self) -> PrintState:
        """Poll Moonraker for current print state. Never raises; returns an
        UNKNOWN PrintState on any failure so callers fail closed."""
        url = f"{self._cfg.base_url}/printer/objects/query"
        params = {
            "print_stats": "state,filename,print_duration",
            "virtual_sdcard": "progress",
        }
        try:
            resp = self._session.get(url, params=params, timeout=self._cfg.timeout_s)
        except requests.exceptions.RequestException as exc:
            logger.warning("moonraker: get_print_state request failed: %s", exc)
            return _unknown_print_state()

        if resp.status_code != 200:
            logger.warning("moonraker: get_print_state got HTTP %s from %s", resp.status_code, url)
            return _unknown_print_state()

        try:
            data = resp.json()
            status = data["result"]["status"]
            print_stats = status["print_stats"]
            virtual_sdcard = status["virtual_sdcard"]
            raw_state = print_stats["state"]
            elapsed_s = float(print_stats["print_duration"])
            filename = print_stats["filename"]
            progress = float(virtual_sdcard["progress"])
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("moonraker: get_print_state got malformed response: %s", exc)
            return _unknown_print_state()

        state = _STATE_MAP.get(raw_state)
        if state is None:
            logger.warning(
                "moonraker: unrecognised print_stats.state %r, treating as UNKNOWN", raw_state
            )
            state = PrinterState.UNKNOWN

        return PrintState(
            state=state,
            filename=filename,
            elapsed_s=elapsed_s,
            progress=progress,
            fetched_at=time.time(),
        )

    def pause(self) -> bool:
        """Pause the active print. Uses `pause_macro` via gcode/script if
        configured, otherwise the plain pause endpoint. Never raises."""
        if self._cfg.pause_macro:
            script = quote(self._cfg.pause_macro, safe="")
            url = f"{self._cfg.base_url}/printer/gcode/script?script={script}"
        else:
            url = f"{self._cfg.base_url}/printer/print/pause"

        ok = self._post(url)
        logger.info(
            "moonraker: PAUSE requested (macro=%s) -> %s",
            self._cfg.pause_macro or "<none>",
            "ok" if ok else "FAILED",
        )
        return ok

    def cancel(self) -> bool:
        """Cancel the active print. Never raises."""
        url = f"{self._cfg.base_url}/printer/print/cancel"
        ok = self._post(url)
        logger.info("moonraker: CANCEL requested -> %s", "ok" if ok else "FAILED")
        return ok

    def health(self) -> bool:
        """True if Moonraker responds with a 2xx to /server/info."""
        url = f"{self._cfg.base_url}/server/info"
        try:
            resp = self._session.get(url, timeout=self._cfg.timeout_s)
        except requests.exceptions.RequestException as exc:
            logger.warning("moonraker: health check failed: %s", exc)
            return False
        return 200 <= resp.status_code < 300

    def close(self) -> None:
        """Release the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> "MoonrakerClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _post(self, url: str) -> bool:
        try:
            resp = self._session.post(url, timeout=self._cfg.timeout_s)
        except requests.exceptions.RequestException as exc:
            logger.warning("moonraker: POST %s failed: %s", url, exc)
            return False
        return 200 <= resp.status_code < 300
