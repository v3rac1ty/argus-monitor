"""Tests for argus.decision.DecisionEngine.

These tests exist to prove the project's overriding requirement: a LOW
FALSE-POSITIVE RATE. test_nominal_stream_never_triggers is the headline
test -- 10,000 ticks of realistic noise must never produce a PAUSE or
CANCEL. Everything else covers the temporal-suppression mechanics (voting,
consecutive-tick requirements, cooldown, hysteresis, action-mode clamping,
and print-identity resets) that make that possible.
"""

from __future__ import annotations

import random

import pytest

from argus.config import DecisionConfig
from argus.decision import DecisionEngine
from argus.types import (
    Action,
    ActionMode,
    DecisionState,
    GateResult,
    PrinterState,
    PrintState,
)

T0 = 1_700_000_000.0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def make_cfg(**overrides) -> DecisionConfig:
    """A DecisionConfig with deterministic, realistic-ish defaults for tests.

    Direct construction bypasses argus.config's validation, which is fine --
    validation only runs on the config *loading* path (per the project spec).
    action_mode defaults to PAUSE (rather than the production default
    NOTIFY_ONLY) so PAUSE-firing tests actually observe a real PAUSE action
    rather than a clamped-to-NOTIFY one; tests that care about clamping
    override action_mode explicitly.
    """
    base = dict(
        tick_interval_s=1.0,
        warmup_s=60.0,
        ema_alpha=0.35,
        window=12,
        vote_threshold=0.55,
        warn_score=0.60,
        warn_votes=4,
        pause_score=0.75,
        pause_votes=6,
        pause_consecutive=3,
        cancel_score=0.92,
        cancel_votes=9,
        cancel_consecutive=2,
        cancel_enabled=False,
        clear_score=0.40,
        clear_ticks=5,
        cooldown_s=300.0,
        action_mode=ActionMode.PAUSE,
    )
    base.update(overrides)
    return DecisionConfig(**base)


def make_print_state(
    filename: str = "print_a.gcode",
    elapsed_s: float = 999.0,
    state: PrinterState = PrinterState.PRINTING,
    progress: float = 0.5,
    fetched_at: float = T0,
) -> PrintState:
    return PrintState(
        state=state,
        filename=filename,
        elapsed_s=elapsed_s,
        progress=progress,
        fetched_at=fetched_at,
    )


def _fire_pause(engine: DecisionEngine, ps: PrintState, t0: float, max_ticks: int = 200):
    """Drive `engine` with a sustained near-certain failure (p_raw=0.95)
    until it emits PAUSE. Returns (tick_index, decision) where tick_index is
    the 0-based tick at which PAUSE fired (so `now` for that tick was
    `t0 + tick_index`)."""
    for i in range(max_ticks):
        d = engine.tick(
            p_raw=0.95, detections=(), print_state=ps, quality=GateResult.ok(), now=t0 + i
        )
        if d.action is Action.PAUSE:
            return i, d
    raise AssertionError("engine never fired PAUSE within max_ticks")


# --------------------------------------------------------------------------
# 1. The headline false-positive test
# --------------------------------------------------------------------------


def test_nominal_stream_never_triggers():
    """10,000 ticks of realistic nominal noise (mean ~0.09) plus ~1%
    isolated spikes into 0.85-0.99 must never produce a PAUSE or CANCEL.
    cancel_enabled=True and action_mode=CANCEL so nothing is masked by
    clamping -- this proves the temporal suppression itself, not the ceiling.
    """
    cfg = make_cfg(cancel_enabled=True, action_mode=ActionMode.CANCEL)
    engine = DecisionEngine(cfg)
    ps = make_print_state()
    rng = random.Random(20260901)

    pause_count = 0
    cancel_count = 0
    notify_count = 0
    for i in range(10_000):
        if rng.random() < 0.01:
            p = rng.uniform(0.85, 0.99)
        else:
            p = rng.betavariate(2, 20)
        d = engine.tick(p_raw=p, detections=(), print_state=ps, quality=GateResult.ok(), now=T0 + i)
        if d.action is Action.PAUSE:
            pause_count += 1
        elif d.action is Action.CANCEL:
            cancel_count += 1
        elif d.action is Action.NOTIFY:
            notify_count += 1

    assert pause_count == 0
    assert cancel_count == 0
    # (notify_count is not asserted here -- occasional NOTIFY-level warnings
    # from clustered spikes are acceptable; PAUSE/CANCEL are what matter.)


# --------------------------------------------------------------------------
# 2 & 3. Isolated spikes must never reach PAUSE
# --------------------------------------------------------------------------


def test_single_frame_spike_ignored():
    cfg = make_cfg()
    engine = DecisionEngine(cfg)
    ps = make_print_state()

    actions = []
    for i in range(60):
        p = 0.99 if i == 30 else 0.05
        d = engine.tick(p_raw=p, detections=(), print_state=ps, quality=GateResult.ok(), now=T0 + i)
        actions.append(d.action)

    assert Action.PAUSE not in actions
    assert Action.CANCEL not in actions


def test_two_frame_spike_ignored():
    cfg = make_cfg()
    engine = DecisionEngine(cfg)
    ps = make_print_state()

    actions = []
    for i in range(60):
        p = 0.99 if i in (30, 31) else 0.05
        d = engine.tick(p_raw=p, detections=(), print_state=ps, quality=GateResult.ok(), now=T0 + i)
        actions.append(d.action)

    assert Action.PAUSE not in actions
    assert Action.CANCEL not in actions


# --------------------------------------------------------------------------
# 4. Bounded detection latency for a genuine failure
# --------------------------------------------------------------------------


def test_sustained_failure_triggers_within_bound():
    cfg = make_cfg()
    engine = DecisionEngine(cfg)
    ps = make_print_state()

    fire_index, decision = _fire_pause(engine, ps, T0, max_ticks=50)

    assert decision.action is Action.PAUSE
    assert fire_index <= 25


# --------------------------------------------------------------------------
# 5. Cooldown prevents repeated actions
# --------------------------------------------------------------------------


def test_cooldown_prevents_repeat_actions():
    cfg = make_cfg(cooldown_s=300.0)
    engine = DecisionEngine(cfg)
    ps = make_print_state()

    pause_count = 0
    for i in range(400):
        d = engine.tick(p_raw=0.95, detections=(), print_state=ps, quality=GateResult.ok(), now=T0 + i)
        if d.action is Action.PAUSE:
            pause_count += 1

    # Score never drops (p_raw stays 0.95 forever) so cooldown's clear
    # condition is never satisfied -- exactly one PAUSE across the run.
    assert pause_count == 1


# --------------------------------------------------------------------------
# 6. Hysteresis: BOTH cooldown time AND a sustained clear are required
# --------------------------------------------------------------------------


def test_hysteresis_requires_clear_before_rearm():
    cfg = make_cfg(cooldown_s=300.0, clear_score=0.40, clear_ticks=5)
    ps = make_print_state()

    # Case A: cooldown time elapses, but score never drops below
    # clear_score -> must stay in COOLDOWN.
    engine_a = DecisionEngine(cfg)
    fire_index, _ = _fire_pause(engine_a, ps, T0)
    d = engine_a.tick(
        p_raw=0.95, detections=(), print_state=ps, quality=GateResult.ok(),
        now=T0 + fire_index + 1000,
    )
    assert engine_a.state is DecisionState.COOLDOWN
    assert d.action is Action.NONE

    # Case B: score clears immediately, but cooldown time has not elapsed
    # -> must stay in COOLDOWN.
    engine_b = DecisionEngine(cfg)
    fire_index, _ = _fire_pause(engine_b, ps, T0)
    d = None
    for k in range(1, 11):
        d = engine_b.tick(
            p_raw=0.0, detections=(), print_state=ps, quality=GateResult.ok(),
            now=T0 + fire_index + k,
        )
    assert engine_b.state is DecisionState.COOLDOWN
    assert d.action is Action.NONE

    # Case C: keep feeding low scores on engine_b until real time also
    # passes cooldown_s -> both conditions hold and it rearms to ARMED.
    for k in range(11, 400):
        d = engine_b.tick(
            p_raw=0.0, detections=(), print_state=ps, quality=GateResult.ok(),
            now=T0 + fire_index + k,
        )
    assert engine_b.state is DecisionState.ARMED
    assert d.action is Action.NONE


# --------------------------------------------------------------------------
# 7. Gated ticks must never mutate EMA/vote state
# --------------------------------------------------------------------------


def test_gated_ticks_do_not_update_state():
    """warmup, quality, and not_printing gates must all leave EMA/votes
    byte-identical.

    Historical note: this test used to drive the `not_printing` case with a
    PAUSED print_state and assert a reset to zero -- that encoded the OLD,
    buggy `_apply_identity_reset` contract, under which leaving PRINTING for
    *any* reason (including a transient/paused state) wiped every
    accumulator. Under the corrected contract (see
    `DecisionEngine._apply_identity_reset`), a reset requires POSITIVE
    confirmation of a new print identity -- a differing non-None filename,
    or a transition from PRINTING to a confirmed terminal state (COMPLETE /
    CANCELLED / ERROR / STANDBY). PAUSED is neither, so it must PRESERVE
    accumulators just like the warmup and quality gates do, even though the
    scoring gate still reports "not_printing" and refuses to act on this
    tick.
    """
    cfg = make_cfg()
    ps = make_print_state(filename="warmup_quality.gcode")

    # --- warmup gate ---
    engine = DecisionEngine(cfg)
    last = None
    for i in range(6):
        last = engine.tick(p_raw=0.4, detections=(), print_state=ps, quality=GateResult.ok(), now=T0 + i)
    assert last.score != 0.0
    score_before, votes_before = last.score, last.votes

    warmup_ps = make_print_state(filename=ps.filename, elapsed_s=cfg.warmup_s - 1.0)
    gated = engine.tick(
        p_raw=0.99, detections=(), print_state=warmup_ps, quality=GateResult.ok(), now=T0 + 6
    )
    assert gated.action is Action.NONE
    assert gated.reason == "warmup"
    assert gated.score == score_before
    assert gated.votes == votes_before

    gated2 = engine.tick(
        p_raw=0.99, detections=(), print_state=warmup_ps, quality=GateResult.ok(), now=T0 + 7
    )
    assert gated2.score == score_before
    assert gated2.votes == votes_before

    # --- quality gate ---
    engine2 = DecisionEngine(cfg)
    last = None
    for i in range(6):
        last = engine2.tick(p_raw=0.4, detections=(), print_state=ps, quality=GateResult.ok(), now=T0 + i)
    score_before2, votes_before2 = last.score, last.votes

    bad_quality = GateResult.blocked("blur")
    gated3 = engine2.tick(
        p_raw=0.99, detections=(), print_state=ps, quality=bad_quality, now=T0 + 6
    )
    assert gated3.action is Action.NONE
    assert gated3.reason == "quality:blur"
    assert gated3.score == score_before2
    assert gated3.votes == votes_before2

    # --- not_printing gate (own contract; see note above) ---
    engine3 = DecisionEngine(cfg)
    for i in range(6):
        engine3.tick(p_raw=0.4, detections=(), print_state=ps, quality=GateResult.ok(), now=T0 + i)
    not_printing_ps = make_print_state(
        filename=ps.filename, state=PrinterState.PAUSED, elapsed_s=ps.elapsed_s
    )
    gated4 = engine3.tick(
        p_raw=0.99, detections=(), print_state=not_printing_ps, quality=GateResult.ok(), now=T0 + 6
    )
    assert gated4.action is Action.NONE
    assert gated4.reason == "not_printing"
    assert gated4.score == score_before2
    assert gated4.votes == votes_before2


# --------------------------------------------------------------------------
# 8. NOTIFY_ONLY mode is a hard ceiling
# --------------------------------------------------------------------------


def test_notify_only_mode_never_pauses():
    cfg = make_cfg(action_mode=ActionMode.NOTIFY_ONLY, cancel_enabled=True)
    engine = DecisionEngine(cfg)
    ps = make_print_state()

    actions = []
    for i in range(60):
        d = engine.tick(p_raw=1.0, detections=(), print_state=ps, quality=GateResult.ok(), now=T0 + i)
        actions.append(d.action)

    assert Action.PAUSE not in actions
    assert Action.CANCEL not in actions
    assert Action.NOTIFY in actions


# --------------------------------------------------------------------------
# 9. cancel_enabled=False blocks CANCEL even with mode=CANCEL
# --------------------------------------------------------------------------


def test_cancel_disabled_by_default():
    cfg = make_cfg(cancel_enabled=False, action_mode=ActionMode.CANCEL)
    engine = DecisionEngine(cfg)
    ps = make_print_state()

    actions = []
    for i in range(1000):
        d = engine.tick(p_raw=1.0, detections=(), print_state=ps, quality=GateResult.ok(), now=T0 + i)
        actions.append(d.action)

    assert Action.CANCEL not in actions


# --------------------------------------------------------------------------
# 10. A new print resets accumulators
# --------------------------------------------------------------------------


def test_new_print_resets_state():
    cfg = make_cfg()
    engine = DecisionEngine(cfg)
    ps = make_print_state(filename="print_a.gcode")

    d = None
    for i in range(6):
        d = engine.tick(p_raw=0.6, detections=(), print_state=ps, quality=GateResult.ok(), now=T0 + i)
    assert d.score > 0.0
    assert d.votes > 0

    ps2 = make_print_state(filename="print_b.gcode", elapsed_s=ps.elapsed_s)
    d2 = engine.tick(p_raw=0.0, detections=(), print_state=ps2, quality=GateResult.ok(), now=T0 + 6)

    assert d2.score == 0.0
    assert d2.votes == 0
    assert engine.state is DecisionState.ARMED


# --------------------------------------------------------------------------
# 11. Determinism
# --------------------------------------------------------------------------


def test_determinism():
    cfg = make_cfg(cancel_enabled=True, action_mode=ActionMode.CANCEL)
    ps = make_print_state()

    def run():
        engine = DecisionEngine(cfg)
        rng = random.Random(777)
        results = []
        for i in range(2000):
            if rng.random() < 0.01:
                p = rng.uniform(0.85, 0.99)
            else:
                p = rng.betavariate(2, 20)
            d = engine.tick(p_raw=p, detections=(), print_state=ps, quality=GateResult.ok(), now=T0 + i)
            results.append((d.action, d.state, d.score, d.p_raw, d.votes, d.consecutive, d.reason))
        return results

    r1 = run()
    r2 = run()
    assert r1 == r2


# --------------------------------------------------------------------------
# 12. Print-identity reset: confirmed change vs. merely indeterminate
# --------------------------------------------------------------------------
#
# Regression coverage for a missed-detection bug: MoonrakerClient.get_print_
# state() is fail-closed -- any transient failure (connection error,
# timeout, HTTP error, malformed JSON) comes back as PrintState(state=
# UNKNOWN, filename=None, ...) rather than raising. The OLD
# `_apply_identity_reset` treated "left PRINTING" and "filename changed" as
# reset triggers, and UNKNOWN satisfies both at once (its filename=None
# differs from any real prior filename). So a single flaky Moonraker poll
# mid-print wiped every accumulated EMA/vote/consecutive-count, and a real
# sustained failure whose evidence accumulates slower than flaky polls
# arrive could never earn enough votes to fire. The corrected engine only
# resets on a POSITIVE confirmation of a new print identity (a differing
# non-None filename, or PRINTING -> a confirmed terminal state) and
# preserves accumulators otherwise -- this is safe because the scoring gate
# already blocks any action from firing whenever print_state is None,
# UNKNOWN, or PAUSED (reason "not_printing"), so preserving evidence there
# can never cause a false action.


def _build_up(engine: DecisionEngine, ps, t0: float, p_raw: float = 0.6, n: int = 6):
    """Feed `n` scored PRINTING ticks so EMA/votes are nonzero, and return
    the last Decision."""
    last = None
    for i in range(n):
        last = engine.tick(p_raw=p_raw, detections=(), print_state=ps, quality=GateResult.ok(), now=t0 + i)
    assert last.score > 0.0
    assert last.votes > 0
    return last


def test_transient_unknown_preserves_accumulators():
    """A single UNKNOWN print_state (exactly what a flaky Moonraker poll
    returns) must leave the EMA and vote window byte-identical -- only the
    (already-safe) not_printing gate applies."""
    cfg = make_cfg()
    engine = DecisionEngine(cfg)
    ps = make_print_state(filename="print_a.gcode")

    last = _build_up(engine, ps, T0)
    score_before, votes_before = last.score, last.votes

    unknown_ps = make_print_state(filename=None, state=PrinterState.UNKNOWN, elapsed_s=0.0, progress=0.0)
    d = engine.tick(p_raw=0.99, detections=(), print_state=unknown_ps, quality=GateResult.ok(), now=T0 + 6)

    assert d.action is Action.NONE
    assert d.reason == "not_printing"
    assert d.score == score_before
    assert d.votes == votes_before


def test_flaky_moonraker_still_detects():
    """Headline regression test: a genuine sustained failure (p_raw~=0.95)
    must still fire PAUSE within a bounded number of ticks even when EVERY
    OTHER poll comes back UNKNOWN, simulating a flapping Moonraker
    connection. This is the direct scenario the bug fix targets -- see the
    section docstring above and the report for confirmation this fails on
    the pre-fix engine and passes after the fix."""
    cfg = make_cfg()
    engine = DecisionEngine(cfg)
    ps = make_print_state(filename="print_a.gcode")
    unknown_ps = make_print_state(filename=None, state=PrinterState.UNKNOWN, elapsed_s=0.0, progress=0.0)

    fired_at = None
    for i in range(200):
        flaky = i % 2 == 1
        d = engine.tick(
            p_raw=0.95,
            detections=(),
            print_state=unknown_ps if flaky else ps,
            quality=GateResult.ok(),
            now=T0 + i,
        )
        if d.action is Action.PAUSE:
            fired_at = i
            break

    assert fired_at is not None, "PAUSE never fired despite a sustained genuine failure"
    assert fired_at <= 60


@pytest.mark.parametrize(
    "terminal_state",
    [PrinterState.COMPLETE, PrinterState.CANCELLED, PrinterState.ERROR, PrinterState.STANDBY],
)
def test_confirmed_terminal_state_resets(terminal_state):
    """Leaving PRINTING for a confirmed terminal state resets everything."""
    cfg = make_cfg()
    engine = DecisionEngine(cfg)
    ps = make_print_state(filename="print_a.gcode")

    _build_up(engine, ps, T0)

    terminal_ps = make_print_state(filename=ps.filename, state=terminal_state, elapsed_s=ps.elapsed_s)
    d = engine.tick(p_raw=0.0, detections=(), print_state=terminal_ps, quality=GateResult.ok(), now=T0 + 6)

    assert d.action is Action.NONE
    assert d.reason == "not_printing"
    assert d.score == 0.0
    assert d.votes == 0
    assert engine.state is DecisionState.IDLE


def test_paused_preserves_accumulators():
    """PAUSED is not a new job -- if the user resumes, prior evidence is
    still relevant, so accumulators must be preserved, not reset."""
    cfg = make_cfg()
    engine = DecisionEngine(cfg)
    ps = make_print_state(filename="print_a.gcode")

    last = _build_up(engine, ps, T0)
    score_before, votes_before = last.score, last.votes

    paused_ps = make_print_state(filename=ps.filename, state=PrinterState.PAUSED, elapsed_s=ps.elapsed_s)
    d = engine.tick(p_raw=0.99, detections=(), print_state=paused_ps, quality=GateResult.ok(), now=T0 + 6)

    assert d.action is Action.NONE
    assert d.reason == "not_printing"
    assert d.score == score_before
    assert d.votes == votes_before

    # Resuming the same job continues to accumulate from where it left off,
    # rather than starting over.
    resumed = engine.tick(p_raw=0.6, detections=(), print_state=ps, quality=GateResult.ok(), now=T0 + 7)
    assert resumed.votes == votes_before + 1


def test_none_print_state_preserves_accumulators():
    """print_state=None means 'could not determine', not 'print ended' --
    accumulators must be preserved, and the gate still returns NONE."""
    cfg = make_cfg()
    engine = DecisionEngine(cfg)
    ps = make_print_state(filename="print_a.gcode")

    last = _build_up(engine, ps, T0)
    score_before, votes_before = last.score, last.votes

    d = engine.tick(p_raw=0.99, detections=(), print_state=None, quality=GateResult.ok(), now=T0 + 6)

    assert d.action is Action.NONE
    assert d.reason == "not_printing"
    assert d.score == score_before
    assert d.votes == votes_before


def test_filename_change_resets():
    """Two different non-None filenames are a genuinely different job and
    must still reset -- this behaviour must survive the fix."""
    cfg = make_cfg()
    engine = DecisionEngine(cfg)
    ps = make_print_state(filename="print_a.gcode")

    _build_up(engine, ps, T0)

    ps2 = make_print_state(filename="print_b.gcode", elapsed_s=ps.elapsed_s)
    d = engine.tick(p_raw=0.0, detections=(), print_state=ps2, quality=GateResult.ok(), now=T0 + 6)

    assert d.score == 0.0
    assert d.votes == 0
    assert engine.state is DecisionState.ARMED


def test_filename_none_does_not_reset():
    """A real filename followed by filename=None (exactly what UNKNOWN
    carries) must NOT be treated as a filename change -- both filenames
    must be non-None to compare. This tick is kept at state=PRINTING (so it
    is not gated) to isolate the filename comparison in
    `_apply_identity_reset` from the not_printing gate."""
    cfg = make_cfg()
    engine = DecisionEngine(cfg)
    ps = make_print_state(filename="print_a.gcode")

    last = _build_up(engine, ps, T0)
    score_before, votes_before = last.score, last.votes

    none_filename_ps = make_print_state(filename=None, elapsed_s=ps.elapsed_s)
    d = engine.tick(p_raw=0.6, detections=(), print_state=none_filename_ps, quality=GateResult.ok(), now=T0 + 6)

    expected_score = cfg.ema_alpha * 0.6 + (1.0 - cfg.ema_alpha) * score_before
    assert d.score == pytest.approx(expected_score)
    assert d.votes == votes_before + 1
