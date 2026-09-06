"""Moonraker (Klipper) REST API client.

Safety contract: every method fails closed and never raises. `get_print_state`
returns `PrintState(state=UNKNOWN, ...)` on any failure, so callers (the
DecisionEngine's gate) treat it as "cannot confirm printing" and take no
action; `pause`/`cancel`/`health` return False on failure instead.
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


# PrintState _unknown_print_state()
# Inputs: None
# Outputs: PrintState - fail-safe snapshot with state=PrinterState.UNKNOWN, filename=None,
#                        elapsed_s=0.0, progress=0.0, fetched_at=now
# Description: Builds the safe fallback PrintState returned whenever real state cannot be
#              confirmed, so downstream logic fails closed.
# Side Effects: Reads the current wall-clock time (time.time()) for fetched_at.
def _unknown_print_state() -> PrintState:
    return PrintState(
        state=PrinterState.UNKNOWN,
        filename=None,
        elapsed_s=0.0,
        progress=0.0,
        fetched_at=time.time(),
    )


class MoonrakerClient:
    """Thin, defensive REST client for a single Moonraker instance."""

    # None __init__(self, MoonrakerConfig cfg, Optional[requests.Session] session=None)
    # Inputs: MoonrakerConfig cfg - base_url, timeout_s, pause_macro
    #         Optional[requests.Session] session - HTTP session to reuse; defaults to None, in
    #                                               which case a fresh requests.Session() is
    #                                               created (mainly for injecting a mock in tests)
    # Outputs: None
    # Description: Stores the config and HTTP session used for every Moonraker request.
    # Side Effects: Constructs a new requests.Session() (which may open connection pools) when
    #               `session` is not provided.
    def __init__(self, cfg: MoonrakerConfig, session: Optional[requests.Session] = None) -> None:
        self._cfg = cfg
        self._session = session if session is not None else requests.Session()

    # PrintState get_print_state(self)
    # Inputs: None
    # Outputs: PrintState - current print state; UNKNOWN on any failure
    # Description: Polls Moonraker's /printer/objects/query endpoint for print_stats and
    #              virtual_sdcard, mapping the response into a PrintState. Never raises; any
    #              connection error, timeout, non-2xx status, malformed JSON, missing key, or
    #              unrecognised print_stats.state value results in the safe UNKNOWN fallback so
    #              callers fail closed.
    # Side Effects: Issues an HTTP GET to `cfg.base_url`/printer/objects/query. Logs a warning on
    #               any failure. Never raises.
    def get_print_state(self) -> PrintState:
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

    # bool pause(self)
    # Inputs: None
    # Outputs: bool - True if Moonraker acknowledged the request with a 2xx, else False
    # Description: Pauses the active print, using cfg.pause_macro via /printer/gcode/script if
    #              configured, otherwise the plain /printer/print/pause endpoint.
    # Side Effects: PAUSES THE USER'S PRINT (issues an HTTP POST to Moonraker). Logs an info line
    #               with the outcome. Never raises.
    def pause(self) -> bool:
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

    # bool cancel(self)
    # Inputs: None
    # Outputs: bool - True if Moonraker acknowledged the request with a 2xx, else False
    # Description: Cancels the active print via /printer/print/cancel.
    # Side Effects: CANCELS THE USER'S PRINT (issues an HTTP POST to Moonraker). Logs an info
    #               line with the outcome. Never raises.
    def cancel(self) -> bool:
        url = f"{self._cfg.base_url}/printer/print/cancel"
        ok = self._post(url)
        logger.info("moonraker: CANCEL requested -> %s", "ok" if ok else "FAILED")
        return ok

    # bool health(self)
    # Inputs: None
    # Outputs: bool - True if Moonraker responded with a 2xx status, else False
    # Description: Checks Moonraker connectivity/liveness via /server/info.
    # Side Effects: Issues an HTTP GET to `cfg.base_url`/server/info. Logs a warning on failure.
    #               Never raises.
    def health(self) -> bool:
        url = f"{self._cfg.base_url}/server/info"
        try:
            resp = self._session.get(url, timeout=self._cfg.timeout_s)
        except requests.exceptions.RequestException as exc:
            logger.warning("moonraker: health check failed: %s", exc)
            return False
        return 200 <= resp.status_code < 300

    # None close(self)
    # Inputs: None
    # Outputs: None
    # Description: Releases the underlying HTTP session's resources (connection pool, etc).
    # Side Effects: Closes `self._session`.
    def close(self) -> None:
        self._session.close()

    # MoonrakerClient __enter__(self)
    # Inputs: None
    # Outputs: MoonrakerClient - self, for use as a context manager
    # Description: Enables `with MoonrakerClient(...) as client:` usage.
    # Side Effects: None
    def __enter__(self) -> "MoonrakerClient":
        return self

    # None __exit__(self, object *exc_info)
    # Inputs: object *exc_info - exception type/value/traceback from the `with` block (unused)
    # Outputs: None
    # Description: Context-manager exit hook that ensures the HTTP session is released.
    # Side Effects: Closes `self._session` (via close()).
    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # bool _post(self, str url)
    # Inputs: str url - fully-formed URL to POST to
    # Outputs: bool - True if the response status was 2xx, else False
    # Description: Shared POST helper used by pause()/cancel() to issue the request and interpret
    #              the response.
    # Side Effects: Issues an HTTP POST to `url`. Logs a warning on a request exception. Never
    #               raises.
    def _post(self, url: str) -> bool:
        try:
            resp = self._session.post(url, timeout=self._cfg.timeout_s)
        except requests.exceptions.RequestException as exc:
            logger.warning("moonraker: POST %s failed: %s", url, exc)
            return False
        return 200 <= resp.status_code < 300
