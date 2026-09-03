"""Temporal DecisionEngine: converts a noisy per-frame failure score into a
small number of trustworthy actions (NOTIFY / PAUSE / CANCEL).

A single raw per-frame model score (`p_raw`) is far too noisy to act on
directly -- a single blurry frame or a stray shadow can spike it to 0.99.
This engine smooths that signal with an EMA, requires sustained agreement
across a rolling vote window, requires several consecutive qualifying ticks
before firing, and enforces a cooldown + hysteresis cycle after any action
fires so a single failure can never produce a burst of repeated actions.

The overriding design goal is a LOW FALSE-POSITIVE RATE: a false PAUSE/CANCEL
kills a print that may have been running for many hours. Every knob here
biases toward "when in doubt, do nothing" -- see config.example.yaml's
`decision` section (mirrored by `argus.config.DecisionConfig`) for the
tunable thresholds.

`tick()` never calls `time.time()`; the caller passes an explicit `now` unix
timestamp so tests (and the low-FPR test in particular) can drive time
deterministically.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from argus.config import DecisionConfig
from argus.types import (
    Action,
    ActionMode,
    Decision,
    DecisionState,
    Detection,
    GateResult,
    PrinterState,
    PrintState,
)


class DecisionEngine:
    """Stateful temporal decision engine driven one tick at a time.

    One instance is meant to live for the lifetime of the monitoring
    process and track exactly one print at a time; `tick()` detects a
    positively-confirmed change of print identity (a new filename, or
    leaving PRINTING for a confirmed terminal state) and resets its
    internal accumulators automatically -- see `_apply_identity_reset`.

    Crucially, a change of identity must be *confirmed*, not merely
    *undetermined*. `MoonrakerClient.get_print_state` is fail-closed: any
    transient failure (connection error, timeout, HTTP error, malformed
    response) comes back as `PrintState(state=UNKNOWN, filename=None, ...)`
    rather than raising. Treating that the same as "the print ended" would
    let a single flaky Moonraker poll mid-print wipe out every accumulated
    vote and EMA sample, which on a real, sustained failure can prevent
    enough evidence from ever accumulating to fire -- a missed detection,
    which is just as much a failure of this system as a false positive.
    So `_apply_identity_reset` only resets on positive confirmation and
    preserves accumulators whenever the state is merely unknown or paused.
    """

    def __init__(self, cfg: DecisionConfig) -> None:
        self._cfg = cfg
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset every temporal accumulator and the state machine to fresh.

        Called on construction, on an explicit external reset, and
        internally whenever `tick()` detects a new print (Step 1).
        """
        self._state: DecisionState = DecisionState.IDLE
        self._score: float = 0.0
        self._vote_window: "deque[bool]" = deque(maxlen=max(1, self._cfg.window))
        self._votes: int = 0

        self._warn_consec: int = 0
        self._pause_consec: int = 0
        self._cancel_consec: int = 0

        self._warn_latch: bool = False

        self._cooldown_until: float = 0.0
        self._clear_consec: int = 0

        self._prev_print_state: Optional[PrintState] = None

    @property
    def state(self) -> DecisionState:
        """Current state-machine state."""
        return self._state

    def tick(
        self,
        *,
        p_raw: float,
        detections: tuple[Detection, ...],
        print_state: Optional[PrintState],
        quality: GateResult,
        now: float,
    ) -> Decision:
        """Process one frame's worth of signal and return the Decision for it."""
        cfg = self._cfg

        # Step 1 -- print-identity reset.
        self._apply_identity_reset(print_state)

        # Step 2 -- scoring gates. A gated tick must not touch EMA/votes.
        gate_reason = self._gate_reason(print_state, quality)
        if gate_reason is not None:
            return Decision(
                action=Action.NONE,
                state=self._state,
                score=self._score,
                p_raw=p_raw,
                votes=self._votes,
                window=cfg.window,
                consecutive=0,
                reason=gate_reason,
                detections=detections,
                gate=quality,
            )

        # Step 3 -- score: EMA + vote window.
        self._score = cfg.ema_alpha * p_raw + (1.0 - cfg.ema_alpha) * self._score
        self._vote_window.append(p_raw > cfg.vote_threshold)
        self._votes = sum(self._vote_window)

        if self._state is DecisionState.IDLE:
            self._state = DecisionState.ARMED

        # Step 4 -- level conditions + consecutive counters.
        warn_ok = self._score > cfg.warn_score and self._votes >= cfg.warn_votes
        pause_ok = self._score > cfg.pause_score and self._votes >= cfg.pause_votes
        cancel_ok = self._score > cfg.cancel_score and self._votes >= cfg.cancel_votes

        self._warn_consec = self._warn_consec + 1 if warn_ok else 0
        self._pause_consec = self._pause_consec + 1 if pause_ok else 0
        self._cancel_consec = self._cancel_consec + 1 if cancel_ok else 0

        was_warn_latched = self._warn_latch
        self._warn_latch = warn_ok

        # Step 5 -- cooldown suppression / clearance.
        if self._state is DecisionState.COOLDOWN:
            if self._score < cfg.clear_score:
                self._clear_consec += 1
            else:
                self._clear_consec = 0

            cleared = now >= self._cooldown_until and self._clear_consec >= cfg.clear_ticks
            if not cleared:
                remaining = max(0.0, self._cooldown_until - now)
                reason = (
                    f"cooldown: score={self._score:.3f} votes={self._votes}/{cfg.window} "
                    f"remaining={remaining:.1f}s clear={self._clear_consec}/{cfg.clear_ticks}"
                )
                return Decision(
                    action=Action.NONE,
                    state=self._state,
                    score=self._score,
                    p_raw=p_raw,
                    votes=self._votes,
                    window=cfg.window,
                    consecutive=self._pause_consec,
                    reason=reason,
                    detections=detections,
                    gate=quality,
                )
            # Hysteresis satisfied: rearm and fall through to normal
            # evaluation for this same tick.
            self._state = DecisionState.ARMED

        # Step 6 -- candidate action, highest level wins.
        level: Optional[str]
        if cfg.cancel_enabled and cancel_ok and self._cancel_consec >= cfg.cancel_consecutive:
            pre_action, level, consecutive = Action.CANCEL, "cancel", self._cancel_consec
        elif pause_ok and self._pause_consec >= cfg.pause_consecutive:
            pre_action, level, consecutive = Action.PAUSE, "pause", self._pause_consec
        else:
            pre_action, level, consecutive = Action.NONE, None, self._warn_consec

        if pre_action is not Action.NONE:
            # Step 7 -- clamp by action_mode (safety ceiling). The state
            # machine still transitions/cools down as though the pre-clamp
            # action fired, so NOTIFY_ONLY faithfully previews what the
            # armed system would have done.
            final_action = self._clamp(pre_action, cfg)
            reason = (
                f"{level}: score={self._score:.3f} votes={self._votes}/{cfg.window} "
                f"consec={consecutive}"
            )
            if final_action is not pre_action:
                reason += f" (downgraded to {final_action.value} by action_mode={cfg.action_mode.value})"

            # Step 8 -- TRIGGERED this tick, COOLDOWN starting next tick.
            self._cooldown_until = now + cfg.cooldown_s
            self._clear_consec = 0
            self._state = DecisionState.COOLDOWN

            return Decision(
                action=final_action,
                state=DecisionState.TRIGGERED,
                score=self._score,
                p_raw=p_raw,
                votes=self._votes,
                window=cfg.window,
                consecutive=consecutive,
                reason=reason,
                detections=detections,
                gate=quality,
            )

        if warn_ok:
            self._state = DecisionState.WARNING
            if not was_warn_latched:
                # Edge-triggered: only fire on the transition into WARNING.
                action = Action.NOTIFY
                reason = (
                    f"notify: score={self._score:.3f} votes={self._votes}/{cfg.window} "
                    f"consec={self._warn_consec}"
                )
            else:
                action = Action.NONE
                reason = (
                    f"warning: score={self._score:.3f} votes={self._votes}/{cfg.window} "
                    f"consec={self._warn_consec}"
                )
        else:
            self._state = DecisionState.ARMED
            action = Action.NONE
            reason = f"armed: score={self._score:.3f} votes={self._votes}/{cfg.window}"

        return Decision(
            action=action,
            state=self._state,
            score=self._score,
            p_raw=p_raw,
            votes=self._votes,
            window=cfg.window,
            consecutive=self._warn_consec,
            reason=reason,
            detections=detections,
            gate=quality,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_identity_reset(self, print_state: Optional[PrintState]) -> None:
        """Step 1: reset all temporal accumulators, but ONLY on a
        positively-confirmed change of print identity -- never merely on an
        inability to determine the current state.

        `MoonrakerClient.get_print_state` is deliberately fail-closed: on
        any transient failure (connection error, timeout, HTTP error,
        malformed JSON) it returns `PrintState(state=UNKNOWN,
        filename=None, ...)` rather than raising. On a Pi under print load,
        such hiccups are not rare. If UNKNOWN were treated as "the print
        ended" -- which its `filename=None` invites, since it also looks
        like "the filename changed" -- a single flaky poll mid-print would
        wipe every temporal accumulator (EMA, vote window, consecutive
        counters). A genuine sustained failure whose evidence accumulates
        more slowly than flaky polls arrive could then never earn enough
        votes to fire: a missed detection, which is just as much a failure
        of this system as a false positive.

        So a reset now requires POSITIVE confirmation of a new identity:

          * `prev.filename` and `print_state.filename` are both non-None
            and differ (a genuinely different job), or
          * `prev.state is PRINTING` and the new state is a confirmed
            terminal state: COMPLETE, CANCELLED, ERROR, or STANDBY.

        Everything else -- `print_state is None`, `state is UNKNOWN`, or
        `state is PAUSED` -- means "we could not determine the state" or
        "the same job is merely paused", not "the print ended", so
        accumulators are PRESERVED. This is safe: `_gate_reason` already
        returns `Action.NONE` with reason "not_printing" whenever
        `print_state` is None or not PRINTING, so no action can ever fire
        while the state is unconfirmed or paused -- preserving accumulators
        here can only avoid discarding evidence, it can never cause a false
        action.
        """
        prev = self._prev_print_state
        should_reset = False

        if print_state is not None and prev is not None:
            filename_changed = (
                prev.filename is not None
                and print_state.filename is not None
                and prev.filename != print_state.filename
            )
            confirmed_terminal = prev.state is PrinterState.PRINTING and print_state.state in (
                PrinterState.COMPLETE,
                PrinterState.CANCELLED,
                PrinterState.ERROR,
                PrinterState.STANDBY,
            )
            should_reset = filename_changed or confirmed_terminal

        if should_reset:
            self.reset()

        self._prev_print_state = print_state

    def _gate_reason(self, print_state: Optional[PrintState], quality: GateResult) -> Optional[str]:
        """Step 2: return a gate reason if this tick must not be scored, else None."""
        cfg = self._cfg
        if print_state is None or not print_state.is_printing:
            return "not_printing"
        if print_state.elapsed_s < cfg.warmup_s:
            return "warmup"
        if not quality.passed:
            return f"quality:{quality.reason}"
        return None

    @staticmethod
    def _clamp(action: Action, cfg: DecisionConfig) -> Action:
        """Step 7: clamp a candidate action down to the configured action_mode ceiling."""
        if cfg.action_mode is ActionMode.NOTIFY_ONLY:
            if action in (Action.PAUSE, Action.CANCEL):
                return Action.NOTIFY
            return action
        if cfg.action_mode is ActionMode.PAUSE:
            if action is Action.CANCEL:
                return Action.PAUSE
            return action
        # ActionMode.CANCEL: allowed as-is (cfg.cancel_enabled already gated
        # whether a CANCEL-level pre_action could ever be produced).
        return action
