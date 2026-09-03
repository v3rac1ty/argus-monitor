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


def _redact_webhook_url(url: Optional[str]) -> str:
    """Last-6-chars redaction -- the webhook URL is a secret and must never
    be logged in full."""
    if not url:
        return "<none>"
    return f"...{url[-6:]}"


class Notifier(ABC):
    """Common interface for delivering an Event somewhere a human will see it."""

    @abstractmethod
    def send(self, event: Event, frame_path: Optional[str] = None) -> bool:
        """Deliver `event`, optionally attaching the frame at `frame_path`.
        Returns True on (best-effort-confirmed) success. Must never raise."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any underlying resources. Default is a no-op."""
        return None

    def __enter__(self) -> "Notifier":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class NullNotifier(Notifier):
    """Used when no webhook is configured. Logs locally and reports success
    so callers don't treat "notifications disabled" as an error condition."""

    def send(self, event: Event, frame_path: Optional[str] = None) -> bool:
        logger.info(
            "NullNotifier: no webhook configured, dropping event (action=%s, reason=%s)",
            event.action.value,
            event.reason,
        )
        return True


class DiscordWebhookNotifier(Notifier):
    """Posts Event alerts as Discord embeds via an incoming webhook."""

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

    def _send_frame(self, frame_path: str) -> None:
        """Best-effort image attachment. Any failure here is logged and
        swallowed -- the text alert already succeeded and is what matters."""
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

    def close(self) -> None:
        self._session.close()


def _extract_retry_after(resp: requests.Response) -> Optional[str]:
    """Best-effort extraction of Discord's retry_after, from the JSON body
    first (Discord's documented 429 shape) and falling back to the header."""
    try:
        data = resp.json()
        if isinstance(data, dict) and "retry_after" in data:
            return str(data["retry_after"])
    except ValueError:
        pass
    return resp.headers.get("Retry-After")


def _fmt_percent(fraction: Optional[float]) -> str:
    if fraction is None:
        return "n/a"
    return f"{fraction * 100:.1f}%"


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


def build_notifier(cfg: NotifyConfig) -> Notifier:
    """Factory: NullNotifier when no webhook is configured, else a real one."""
    if not cfg.discord_webhook_url:
        return NullNotifier()
    return DiscordWebhookNotifier(cfg)
