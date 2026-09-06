"""Discord webhook notifications for Argus Monitor events.

A notification failure (network error, Discord outage, bad webhook URL) must
never take down the monitoring loop and must never block a pause/cancel from
happening -- notification is best-effort side reporting, not part of the
safety-critical path. Every public method here is therefore designed to
never raise.
"""

from __future__ import annotations

import datetime
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import requests

from argus.config import NotifyConfig
from argus.types import Action, Event

logger = logging.getLogger(__name__)

_COLOR_ALERT = 0xFF4444  # PAUSE / CANCEL
_COLOR_WARN = 0xFFAA00  # NOTIFY (and anything else)
_ALERT_ACTIONS = frozenset({Action.PAUSE, Action.CANCEL})


# str _redact_webhook_url(Optional[str] url)
# Inputs: Optional[str] url - the webhook URL to redact (may be None/empty)
# Outputs: str - "<none>" if `url` is falsy, else "...<last 6 chars>"
# Description: Redacts a Discord webhook URL for safe logging.
# Side Effects: None
def _redact_webhook_url(url: Optional[str]) -> str:
    if not url:
        return "<none>"
    return f"...{url[-6:]}"


class Notifier(ABC):
    """Common interface for delivering an Event somewhere a human will see it."""

    # bool send(self, Event event, Optional[str] frame_path=None)
    # Inputs: Event event - the event to deliver
    #         Optional[str] frame_path - path to a captured frame to attach; defaults to None
    #                                     (no attachment)
    # Outputs: bool - True on (best-effort-confirmed) success
    # Description: Abstract interface method: deliver `event` (and optionally the frame at
    #              `frame_path`) somewhere a human will see it. Implementations must never raise.
    # Side Effects: Implementation-defined (typically network I/O); this base declaration has
    #               none of its own.
    @abstractmethod
    def send(self, event: Event, frame_path: Optional[str] = None) -> bool:
        raise NotImplementedError

    # None close(self)
    # Inputs: None
    # Outputs: None
    # Description: Default no-op resource-release hook; subclasses override to close real
    #              connections.
    # Side Effects: None
    def close(self) -> None:
        return None

    # Notifier __enter__(self)
    # Inputs: None
    # Outputs: Notifier - self, for use as a context manager
    # Description: Enables `with build_notifier(cfg) as notifier:` usage.
    # Side Effects: None
    def __enter__(self) -> "Notifier":
        return self

    # None __exit__(self, object *exc_info)
    # Inputs: object *exc_info - exception type/value/traceback from the `with` block (unused)
    # Outputs: None
    # Description: Context-manager exit hook that ensures resources are released.
    # Side Effects: Calls self.close() (effects depend on the concrete subclass).
    def __exit__(self, *exc_info: object) -> None:
        self.close()


class NullNotifier(Notifier):
    """Used when no webhook is configured. Logs locally and reports success
    so callers don't treat "notifications disabled" as an error condition."""

    # bool send(self, Event event, Optional[str] frame_path=None)
    # Inputs: Event event - the event that would have been delivered
    #         Optional[str] frame_path - path to a captured frame; defaults to None, and is
    #                                     ignored regardless (never sent anywhere)
    # Outputs: bool - always True (so callers don't treat "notifications disabled" as an error)
    # Description: Drops the event on the floor since no webhook is configured.
    # Side Effects: Logs an info line noting the dropped event.
    def send(self, event: Event, frame_path: Optional[str] = None) -> bool:
        logger.info(
            "NullNotifier: no webhook configured, dropping event (action=%s, reason=%s)",
            event.action.value,
            event.reason,
        )
        return True


class DiscordWebhookNotifier(Notifier):
    """Posts Event alerts as Discord embeds via an incoming webhook."""

    # None __init__(self, NotifyConfig cfg, Optional[requests.Session] session=None, Callable[[], float] clock=time.monotonic)
    # Inputs: NotifyConfig cfg - webhook URL, attach_frame flag, min_interval_s rate limit
    #         Optional[requests.Session] session - HTTP session to reuse; defaults to None, in
    #                                               which case a fresh requests.Session() is
    #                                               created (mainly for injecting a mock in tests)
    #         Callable[[], float] clock - time source for rate-limiting; defaults to
    #                                      time.monotonic (overridable in tests for determinism)
    # Outputs: None
    # Description: Stores config/session/clock and initializes the rate-limit timestamp.
    # Side Effects: Constructs a new requests.Session() when `session` is not provided. Logs an
    #               info line with the (redacted) webhook config.
    def __init__(
        self,
        cfg: NotifyConfig,
        session: Optional[requests.Session] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = cfg
        self._session = session if session is not None else requests.Session()
        self._clock = clock
        self._last_send_at: Optional[float] = None
        logger.info(
            "DiscordWebhookNotifier configured (webhook %s, min_interval_s=%s, attach_frame=%s)",
            _redact_webhook_url(cfg.discord_webhook_url),
            cfg.min_interval_s,
            cfg.attach_frame,
        )

    # bool send(self, Event event, Optional[str] frame_path=None)
    # Inputs: Event event - the event to deliver as a Discord embed
    #         Optional[str] frame_path - path to a captured frame to attach; defaults to None
    #                                     (no attachment, even if cfg.attach_frame is True)
    # Outputs: bool - True if the embed POST succeeded (2xx); False if rate-limited by our own
    #                 min_interval_s, rate-limited by Discord (429), or the request failed/errored
    # Description: Enforces the configured minimum send interval, then posts `event` as a Discord
    #              embed via the webhook, and (if configured, attached, and the frame file still
    #              exists) best-effort uploads the frame image as a follow-up attachment.
    # Side Effects: Issues an HTTP POST to the Discord webhook URL (embed), and a second POST
    #               (multipart file upload) when attaching a frame. Reads the frame file from
    #               disk. Mutates self._last_send_at on success. Logs debug/info/warning lines.
    #               Never raises.
    def send(self, event: Event, frame_path: Optional[str] = None) -> bool:
        now = self._clock()
        if self._last_send_at is not None and (now - self._last_send_at) < self._cfg.min_interval_s:
            logger.debug(
                "DiscordWebhookNotifier: rate-limited (%.1fs since last send < min_interval_s=%s), skipping",
                now - self._last_send_at,
                self._cfg.min_interval_s,
            )
            return False

        payload = _build_embed_payload(event)
        try:
            resp = self._session.post(self._cfg.discord_webhook_url, json=payload, timeout=10)
        except requests.exceptions.RequestException as exc:
            logger.warning("DiscordWebhookNotifier: embed POST failed: %s", exc)
            return False

        if resp.status_code == 429:
            logger.warning(
                "DiscordWebhookNotifier: rate limited by Discord (429), retry_after=%s",
                _extract_retry_after(resp),
            )
            return False

        if not (200 <= resp.status_code < 300):
            logger.warning("DiscordWebhookNotifier: embed POST returned HTTP %s", resp.status_code)
            return False

        self._last_send_at = now
        logger.info(
            "DiscordWebhookNotifier: sent alert (action=%s, class=%s)",
            event.action.value,
            event.class_name,
        )

        if self._cfg.attach_frame and frame_path and Path(frame_path).exists():
            self._send_frame(frame_path)

        return True

    # None _send_frame(self, str frame_path)
    # Inputs: str frame_path - path to the JPEG frame file to attach
    # Outputs: None
    # Description: Uploads `frame_path` as a follow-up file post to the Discord webhook, as a
    #              best-effort addition to an already-sent text alert.
    # Side Effects: Reads `frame_path` from disk. Issues an HTTP POST (multipart file upload) to
    #               the Discord webhook URL. Logs a warning on any failure. Never raises (all
    #               failures, including missing/unreadable file, are swallowed).
    def _send_frame(self, frame_path: str) -> None:
        try:
            with open(frame_path, "rb") as fh:
                files = {"file": (Path(frame_path).name, fh, "image/jpeg")}
                resp = self._session.post(self._cfg.discord_webhook_url, files=files, timeout=15)
            if not (200 <= resp.status_code < 300):
                logger.warning(
                    "DiscordWebhookNotifier: frame upload returned HTTP %s", resp.status_code
                )
        except requests.exceptions.RequestException as exc:
            logger.warning("DiscordWebhookNotifier: frame upload failed: %s", exc)
        except OSError as exc:
            logger.warning("DiscordWebhookNotifier: could not read frame '%s': %s", frame_path, exc)

    # None close(self)
    # Inputs: None
    # Outputs: None
    # Description: Releases the underlying HTTP session's resources (connection pool, etc).
    # Side Effects: Closes `self._session`.
    def close(self) -> None:
        self._session.close()


# Optional[str] _extract_retry_after(requests.Response resp)
# Inputs: requests.Response resp - the 429 response from Discord
# Outputs: Optional[str] - the retry_after value as a string, or None if it cannot be determined
# Description: Best-effort extraction of Discord's retry_after, from the JSON body first
#              (Discord's documented 429 shape) and falling back to the Retry-After header.
# Side Effects: None (parses the already-fetched response; issues no new I/O).
def _extract_retry_after(resp: requests.Response) -> Optional[str]:
    try:
        data = resp.json()
        if isinstance(data, dict) and "retry_after" in data:
            return str(data["retry_after"])
    except ValueError:
        pass
    return resp.headers.get("Retry-After")


# str _fmt_percent(Optional[float] fraction)
# Inputs: Optional[float] fraction - a 0..1 fraction, or None
# Outputs: str - "n/a" if `fraction` is None, else the fraction formatted as a percentage
#                ("NN.N%")
# Description: Formats an optional confidence fraction for display in the Discord embed.
# Side Effects: None
def _fmt_percent(fraction: Optional[float]) -> str:
    if fraction is None:
        return "n/a"
    return f"{fraction * 100:.1f}%"


# dict _build_embed_payload(Event event)
# Inputs: Event event - the event to render
# Outputs: dict - a Discord webhook payload of the form {"embeds": [embed]}
# Description: Builds the Discord embed JSON body (title, color, timestamp, and field table) for
#              `event`, colored red for PAUSE/CANCEL and amber otherwise.
# Side Effects: None
def _build_embed_payload(event: Event) -> dict:
    d = event.to_dict()
    subject = d["class_name"] or d["action"]
    color = _COLOR_ALERT if event.action in _ALERT_ACTIONS else _COLOR_WARN

    embed = {
        "title": f"Print failure detected: {subject}",
        "color": color,
        "timestamp": datetime.datetime.fromtimestamp(
            d["timestamp"], tz=datetime.timezone.utc
        ).isoformat(),
        "fields": [
            {"name": "Confidence", "value": _fmt_percent(d["confidence"]), "inline": True},
            {"name": "Action", "value": d["action"], "inline": True},
            {"name": "Score", "value": f"{d['score']:.2f}", "inline": True},
            {"name": "Votes", "value": str(d["votes"]), "inline": True},
            {"name": "State", "value": d["state"], "inline": True},
            {"name": "Reason", "value": d["reason"] or "n/a", "inline": False},
            {"name": "Print file", "value": d["print_filename"] or "n/a", "inline": False},
        ],
    }
    return {"embeds": [embed]}


# Notifier build_notifier(NotifyConfig cfg)
# Inputs: NotifyConfig cfg - discord_webhook_url plus DiscordWebhookNotifier settings
# Outputs: Notifier - a NullNotifier if cfg.discord_webhook_url is unset, else a
#                     DiscordWebhookNotifier
# Description: Factory that selects the appropriate Notifier implementation for `cfg`.
# Side Effects: Constructing a DiscordWebhookNotifier creates a requests.Session and logs an
#               info line; otherwise none.
def build_notifier(cfg: NotifyConfig) -> Notifier:
    if not cfg.discord_webhook_url:
        return NullNotifier()
    return DiscordWebhookNotifier(cfg)
