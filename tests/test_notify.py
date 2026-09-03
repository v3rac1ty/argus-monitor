"""Tests for argus.notify: Discord embed shape, rate limiting, frame
attachment, and failure handling.

ZERO real network calls are made -- every test injects a mock `requests`
session. Time is controlled via an injected `clock` callable, never `sleep`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
import requests

from argus.config import NotifyConfig
from argus.notify import DiscordWebhookNotifier, NullNotifier, build_notifier
from argus.types import Action, DecisionState, Event

WEBHOOK_URL = "https://discord.com/api/webhooks/123456789012345678/SuperSecretTokenABCDEF"


def _event(
    action: Action = Action.NOTIFY,
    state: DecisionState = DecisionState.WARNING,
    class_name: Optional[str] = "spaghetti",
    confidence: Optional[float] = 0.87,
    score: float = 0.61,
    votes: int = 5,
    reason: str = "score exceeded warn threshold",
    print_filename: Optional[str] = "benchy.gcode",
    frame_path: Optional[str] = None,
) -> Event:
    return Event(
        timestamp=1_700_000_000.0,
        action=action,
        state=state,
        score=score,
        p_raw=0.9,
        votes=votes,
        reason=reason,
        class_name=class_name,
        confidence=confidence,
        frame_path=frame_path,
        print_filename=print_filename,
        elapsed_s=120.0,
        detections=(),
    )


def _http_response(status_code: int, json_data: Optional[Any] = None, headers: Optional[dict] = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError("no body")
    return resp


class _FakeClock:
    """Injectable monotonic clock a test can advance without sleeping."""

    def __init__(self, start: float = 1000.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _notifier(session: MagicMock, clock: Optional[_FakeClock] = None, **cfg_kwargs: Any) -> DiscordWebhookNotifier:
    cfg = NotifyConfig(discord_webhook_url=WEBHOOK_URL, **cfg_kwargs)
    return DiscordWebhookNotifier(cfg, session=session, clock=clock or _FakeClock())


# --------------------------------------------------------------------------
# Embed payload shape & colour
# --------------------------------------------------------------------------


def test_embed_payload_shape_and_fields():
    session = MagicMock()
    session.post.return_value = _http_response(200)
    notifier = _notifier(session)

    ok = notifier.send(_event(class_name="spaghetti", confidence=0.87, score=0.6123, votes=5))

    assert ok is True
    args, kwargs = session.post.call_args
    assert args[0] == WEBHOOK_URL
    payload = kwargs["json"]
    embed = payload["embeds"][0]
    assert embed["title"] == "Print failure detected: spaghetti"

    fields = {f["name"]: f for f in embed["fields"]}
    assert fields["Confidence"]["value"] == "87.0%"
    assert fields["Confidence"]["inline"] is True
    assert fields["Action"]["value"] == "notify"
    assert fields["Action"]["inline"] is True
    assert fields["Score"]["value"] == "0.61"
    assert fields["Score"]["inline"] is True
    assert fields["Votes"]["value"] == "5"
    assert fields["Votes"]["inline"] is True
    assert fields["State"]["value"] == "warning"
    assert fields["State"]["inline"] is True
    assert fields["Reason"]["inline"] is False
    assert "Print file" in fields
    assert fields["Print file"]["value"] == "benchy.gcode"
    assert fields["Print file"]["inline"] is False


def test_embed_title_falls_back_to_action_name_when_no_class():
    session = MagicMock()
    session.post.return_value = _http_response(200)
    notifier = _notifier(session)

    notifier.send(_event(action=Action.PAUSE, class_name=None))

    payload = session.post.call_args.kwargs["json"]
    assert payload["embeds"][0]["title"] == "Print failure detected: pause"


@pytest.mark.parametrize(
    "action, expected_color",
    [
        (Action.PAUSE, 0xFF4444),
        (Action.CANCEL, 0xFF4444),
        (Action.NOTIFY, 0xFFAA00),
    ],
)
def test_embed_colour_by_action(action, expected_color):
    session = MagicMock()
    session.post.return_value = _http_response(200)
    notifier = _notifier(session)

    notifier.send(_event(action=action))

    payload = session.post.call_args.kwargs["json"]
    assert payload["embeds"][0]["color"] == expected_color


# --------------------------------------------------------------------------
# Rate limiting (injected clock, never sleep)
# --------------------------------------------------------------------------


def test_rate_limiter_suppresses_second_send_within_window():
    session = MagicMock()
    session.post.return_value = _http_response(200)
    clock = _FakeClock()
    notifier = _notifier(session, clock=clock, min_interval_s=60)

    assert notifier.send(_event()) is True
    clock.advance(10)
    assert notifier.send(_event()) is False
    assert session.post.call_count == 1


def test_rate_limiter_permits_send_after_window_elapses():
    session = MagicMock()
    session.post.return_value = _http_response(200)
    clock = _FakeClock()
    notifier = _notifier(session, clock=clock, min_interval_s=60)

    assert notifier.send(_event()) is True
    clock.advance(60)
    assert notifier.send(_event()) is True
    assert session.post.call_count == 2


def test_rate_limiter_does_not_consume_window_on_failed_send():
    session = MagicMock()
    session.post.side_effect = requests.exceptions.ConnectionError("refused")
    clock = _FakeClock()
    notifier = _notifier(session, clock=clock, min_interval_s=60)

    assert notifier.send(_event()) is False
    clock.advance(1)
    # Still within window relative to nothing having succeeded yet -- the
    # failed send must not have started the rate-limit clock.
    session.post.side_effect = None
    session.post.return_value = _http_response(200)
    assert notifier.send(_event()) is True


# --------------------------------------------------------------------------
# Frame attachment
# --------------------------------------------------------------------------


def test_attach_frame_posts_second_multipart_request(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")

    session = MagicMock()
    session.post.return_value = _http_response(200)
    notifier = _notifier(session, attach_frame=True)

    ok = notifier.send(_event(frame_path=str(frame)), frame_path=str(frame))

    assert ok is True
    assert session.post.call_count == 2
    second_call = session.post.call_args_list[1]
    assert "files" in second_call.kwargs
    assert "file" in second_call.kwargs["files"]


def test_attach_frame_skipped_when_disabled(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake")

    session = MagicMock()
    session.post.return_value = _http_response(200)
    notifier = _notifier(session, attach_frame=False)

    ok = notifier.send(_event(), frame_path=str(frame))

    assert ok is True
    assert session.post.call_count == 1


def test_attach_frame_skipped_when_file_missing(tmp_path):
    missing = tmp_path / "does_not_exist.jpg"

    session = MagicMock()
    session.post.return_value = _http_response(200)
    notifier = _notifier(session, attach_frame=True)

    ok = notifier.send(_event(), frame_path=str(missing))

    assert ok is True
    assert session.post.call_count == 1


def test_image_upload_failure_still_returns_true(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake")

    session = MagicMock()
    # First call (embed) succeeds, second call (multipart upload) raises.
    session.post.side_effect = [
        _http_response(200),
        requests.exceptions.ConnectionError("upload failed"),
    ]
    notifier = _notifier(session, attach_frame=True)

    ok = notifier.send(_event(), frame_path=str(frame))

    assert ok is True
    assert session.post.call_count == 2


def test_image_upload_http_error_still_returns_true(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake")

    session = MagicMock()
    session.post.side_effect = [_http_response(200), _http_response(500)]
    notifier = _notifier(session, attach_frame=True)

    ok = notifier.send(_event(), frame_path=str(frame))

    assert ok is True


# --------------------------------------------------------------------------
# Network failures & Discord rate limiting never raise
# --------------------------------------------------------------------------


def test_network_exception_on_embed_returns_false_without_raising():
    session = MagicMock()
    session.post.side_effect = requests.exceptions.Timeout("timed out")
    notifier = _notifier(session)

    assert notifier.send(_event()) is False


def test_discord_429_is_treated_as_failure():
    session = MagicMock()
    session.post.return_value = _http_response(429, json_data={"retry_after": 1.5})
    notifier = _notifier(session)

    assert notifier.send(_event()) is False


def test_non_2xx_embed_status_returns_false():
    session = MagicMock()
    session.post.return_value = _http_response(400)
    notifier = _notifier(session)

    assert notifier.send(_event()) is False


# --------------------------------------------------------------------------
# NullNotifier / build_notifier factory
# --------------------------------------------------------------------------


def test_null_notifier_returns_true_and_does_not_touch_network():
    notifier = NullNotifier()
    assert notifier.send(_event()) is True


def test_build_notifier_returns_null_notifier_when_url_is_none():
    cfg = NotifyConfig(discord_webhook_url=None)
    assert isinstance(build_notifier(cfg), NullNotifier)


def test_build_notifier_returns_discord_notifier_when_url_set():
    cfg = NotifyConfig(discord_webhook_url=WEBHOOK_URL)
    assert isinstance(build_notifier(cfg), DiscordWebhookNotifier)


# --------------------------------------------------------------------------
# Secret hygiene: webhook URL must never be logged
# --------------------------------------------------------------------------


def test_webhook_url_never_appears_in_logs(caplog, tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake")

    session = MagicMock()
    session.post.side_effect = [
        _http_response(200),  # successful embed (also logged)
        requests.exceptions.ConnectionError("upload failed"),  # frame upload failure (logged)
    ]

    with caplog.at_level(logging.DEBUG):
        clock = _FakeClock()
        notifier = _notifier(session, clock=clock, attach_frame=True, min_interval_s=60)
        notifier.send(_event(), frame_path=str(frame))

        # Rate-limited skip (also logged).
        clock.advance(1)
        notifier.send(_event())

        # Failure path (also logged).
        session.post.side_effect = requests.exceptions.ConnectionError("refused")
        clock.advance(60)
        notifier.send(_event())

        # 429 path (also logged).
        session.post.side_effect = None
        session.post.return_value = _http_response(429, json_data={"retry_after": 2})
        clock.advance(60)
        notifier.send(_event())

    for record in caplog.records:
        assert WEBHOOK_URL not in record.getMessage()
