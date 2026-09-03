"""Tests for argus.moonraker.MoonrakerClient.

All tests inject a mock `requests.Session` -- ZERO real network calls are
made. The overriding property under test is safety: get_print_state() must
resolve to PrinterState.UNKNOWN (never raise) on any failure mode, so the
DecisionEngine's gate fails closed.
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
import requests

from argus.config import MoonrakerConfig
from argus.moonraker import MoonrakerClient
from argus.types import PrinterState, PrintState


def _query_response(
    state: str = "printing",
    filename: str = "benchy.gcode",
    print_duration: float = 123.4,
    progress: float = 0.42,
    status_code: int = 200,
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {
        "result": {
            "status": {
                "print_stats": {
                    "state": state,
                    "filename": filename,
                    "print_duration": print_duration,
                },
                "virtual_sdcard": {
                    "progress": progress,
                },
            }
        }
    }
    return resp


def _http_response(status_code: int, json_data: Optional[Any] = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


def _client(session: MagicMock, **cfg_kwargs: Any) -> MoonrakerClient:
    cfg = MoonrakerConfig(**cfg_kwargs)
    return MoonrakerClient(cfg, session=session)


# --------------------------------------------------------------------------
# get_print_state: happy path & state mapping
# --------------------------------------------------------------------------


def test_get_print_state_success_parses_all_fields():
    session = MagicMock()
    session.get.return_value = _query_response(
        state="printing", filename="benchy.gcode", print_duration=123.4, progress=0.42
    )
    client = _client(session)

    result = client.get_print_state()

    assert isinstance(result, PrintState)
    assert result.state is PrinterState.PRINTING
    assert result.filename == "benchy.gcode"
    assert result.elapsed_s == 123.4
    assert result.progress == 0.42
    assert result.fetched_at > 0

    # Hits the objects/query endpoint, requesting both objects in one call.
    args, kwargs = session.get.call_args
    assert args[0].endswith("/printer/objects/query")
    assert kwargs["params"] == {
        "print_stats": "state,filename,print_duration",
        "virtual_sdcard": "progress",
    }
    assert kwargs["timeout"] == 5.0


def test_get_print_state_uses_configured_base_url_and_timeout():
    session = MagicMock()
    session.get.return_value = _query_response()
    client = _client(session, base_url="http://printer.local:7125", timeout_s=2.5)

    client.get_print_state()

    args, kwargs = session.get.call_args
    assert args[0] == "http://printer.local:7125/printer/objects/query"
    assert kwargs["timeout"] == 2.5


@pytest.mark.parametrize(
    "raw_state, expected",
    [
        ("printing", PrinterState.PRINTING),
        ("paused", PrinterState.PAUSED),
        ("complete", PrinterState.COMPLETE),
        ("cancelled", PrinterState.CANCELLED),
        ("error", PrinterState.ERROR),
        ("standby", PrinterState.STANDBY),
    ],
)
def test_get_print_state_maps_each_moonraker_state(raw_state, expected):
    session = MagicMock()
    session.get.return_value = _query_response(state=raw_state)
    client = _client(session)

    result = client.get_print_state()

    assert result.state is expected


def test_get_print_state_unrecognised_state_string_is_unknown():
    session = MagicMock()
    session.get.return_value = _query_response(state="some_future_state")
    client = _client(session)

    result = client.get_print_state()

    assert result.state is PrinterState.UNKNOWN


# --------------------------------------------------------------------------
# get_print_state: fail-closed on every error mode
# --------------------------------------------------------------------------


def test_get_print_state_connection_error_is_unknown_and_does_not_raise():
    session = MagicMock()
    session.get.side_effect = requests.exceptions.ConnectionError("refused")
    client = _client(session)

    result = client.get_print_state()

    assert result.state is PrinterState.UNKNOWN
    assert result.filename is None


def test_get_print_state_timeout_is_unknown_and_does_not_raise():
    session = MagicMock()
    session.get.side_effect = requests.exceptions.Timeout("timed out")
    client = _client(session)

    result = client.get_print_state()

    assert result.state is PrinterState.UNKNOWN


def test_get_print_state_http_500_is_unknown():
    session = MagicMock()
    session.get.return_value = _http_response(500)
    client = _client(session)

    result = client.get_print_state()

    assert result.state is PrinterState.UNKNOWN


def test_get_print_state_malformed_json_is_unknown():
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.side_effect = ValueError("Expecting value: line 1 column 1")
    session.get.return_value = resp
    client = _client(session)

    result = client.get_print_state()

    assert result.state is PrinterState.UNKNOWN


@pytest.mark.parametrize(
    "broken_payload",
    [
        {"result": {"status": {"virtual_sdcard": {"progress": 0.5}}}},  # missing print_stats
        {"result": {"status": {"print_stats": {"filename": "x.gcode", "print_duration": 1.0}}}},  # missing state / virtual_sdcard
        {"result": {}},  # missing status entirely
        {},  # missing result entirely
        {"result": {"status": {"print_stats": {"state": "printing", "print_duration": 1.0}, "virtual_sdcard": {"progress": 0.1}}}},  # missing filename
    ],
)
def test_get_print_state_missing_keys_is_unknown(broken_payload):
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = broken_payload
    session.get.return_value = resp
    client = _client(session)

    result = client.get_print_state()

    assert result.state is PrinterState.UNKNOWN


# --------------------------------------------------------------------------
# pause / cancel
# --------------------------------------------------------------------------


def test_pause_hits_plain_endpoint_by_default():
    session = MagicMock()
    session.post.return_value = _http_response(200)
    client = _client(session)  # pause_macro is None by default

    ok = client.pause()

    assert ok is True
    args, kwargs = session.post.call_args
    assert args[0].endswith("/printer/print/pause")


def test_pause_hits_gcode_script_endpoint_when_macro_configured():
    session = MagicMock()
    session.post.return_value = _http_response(200)
    client = _client(session, pause_macro="DETECTION_PAUSE")

    ok = client.pause()

    assert ok is True
    args, kwargs = session.post.call_args
    url = args[0]
    assert "/printer/gcode/script" in url
    assert "DETECTION_PAUSE" in url


def test_pause_returns_false_on_http_error_without_raising():
    session = MagicMock()
    session.post.return_value = _http_response(500)
    client = _client(session)

    assert client.pause() is False


def test_pause_returns_false_on_exception_without_raising():
    session = MagicMock()
    session.post.side_effect = requests.exceptions.ConnectionError("refused")
    client = _client(session)

    assert client.pause() is False


def test_cancel_hits_cancel_endpoint_and_returns_true_on_2xx():
    session = MagicMock()
    session.post.return_value = _http_response(204)
    client = _client(session)

    ok = client.cancel()

    assert ok is True
    args, kwargs = session.post.call_args
    assert args[0].endswith("/printer/print/cancel")


def test_cancel_returns_false_on_error_without_raising():
    session = MagicMock()
    session.post.side_effect = requests.exceptions.Timeout("timed out")
    client = _client(session)

    assert client.cancel() is False


# --------------------------------------------------------------------------
# health / close
# --------------------------------------------------------------------------


def test_health_true_on_2xx():
    session = MagicMock()
    session.get.return_value = _http_response(200)
    client = _client(session)

    assert client.health() is True
    args, kwargs = session.get.call_args
    assert args[0].endswith("/server/info")


def test_health_false_on_error_status():
    session = MagicMock()
    session.get.return_value = _http_response(503)
    client = _client(session)

    assert client.health() is False


def test_health_false_on_exception_without_raising():
    session = MagicMock()
    session.get.side_effect = requests.exceptions.ConnectionError("refused")
    client = _client(session)

    assert client.health() is False


def test_close_closes_session():
    session = MagicMock()
    client = _client(session)

    client.close()

    session.close.assert_called_once()


def test_context_manager_closes_session():
    session = MagicMock()
    with _client(session) as client:
        assert isinstance(client, MoonrakerClient)

    session.close.assert_called_once()
