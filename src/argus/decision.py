"""Temporal DecisionEngine: converts a noisy per-frame failure score into a
small number of trustworthy actions (NOTIFY / PAUSE / CANCEL) via an EMA,
a rolling vote window, consecutive-tick requirements, and a cooldown +
hysteresis cycle after any action fires.

Design goal is a LOW FALSE-POSITIVE RATE (a false PAUSE/CANCEL kills a
print that may have run for hours); tunables live in
`argus.config.DecisionConfig`. `tick()` takes an explicit `now` rather than
calling `time.time()` so tests can drive it deterministically.
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
    """Stateful temporal decision engine driven one tick at a time; tracks one
    print and resets on a *confirmed* identity change (new filename, or a
    confirmed terminal state). UNKNOWN/PAUSED states preserve accumulators
    instead of resetting, since Moonraker fails closed to UNKNOWN on transient
    errors -- see `_apply_identity_reset`.
    """

    # None __init__(self, DecisionConfig cfg)
    # Inputs: DecisionConfig cfg - smoothing/voting/threshold/cooldown tuning for this engine
    #                               instance
    # Outputs: None
    # Description: Stores the config and initializes all temporal state via `reset()`.
    # Side Effects: Delegates to reset() (see its Side Effects).
    def __init__(self, cfg: DecisionConfig) -> None:
        self._cfg = cfg
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # None reset(self)
    # Inputs: None
    # Outputs: None
    # Description: Resets every temporal accumulator (EMA score, vote window, consecutive
    #              counters, warn latch, cooldown deadline, clear counter, previous print state)
    #              and the state machine back to IDLE. Called on construction, on an explicit
    #              external reset, and internally whenever `tick()` detects a new print.
    # Side Effects: Mutates every instance attribute holding temporal state (self._state,
    #               self._score, self._vote_window, self._votes, self._warn_consec,
    #               self._pause_consec, self._cancel_consec, self._warn_latch,
    #               self._cooldown_until, self._clear_consec, self._prev_print_state).
    def reset(self) -> None:
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

    # DecisionState state(self)
    # Inputs: None
    # Outputs: DecisionState - the engine's current state-machine state
    # Description: Read-only accessor for the current state-machine state.
    # Side Effects: None
    @property
    def state(self) -> DecisionState:
        return self._state

    # Decision tick(self, float p_raw, tuple[Detection, ...] detections, Optional[PrintState] print_state, GateResult quality, float now)
    # Inputs: float p_raw - raw (unsmoothed) per-frame failure probability for this tick
    #         tuple[Detection, ...] detections - detections observed on this tick, carried
    #                                             through unchanged into the returned Decision
    #         Optional[PrintState] print_state - current Moonraker print snapshot, or None if
    #                                             unavailable
    #         GateResult quality - pre-inference quality gate outcome for the frame behind this
    #                               tick
    #         float now - caller-supplied unix timestamp for this tick (never read internally via
    #                      time.time(), so tests can drive time deterministically)
    # Outputs: Decision - the action (NONE/NOTIFY/PAUSE/CANCEL), resulting state, and full
    #                     scoring detail for this tick
    # Description: Advances the engine by one tick: applies a print-identity reset if positively
    #              confirmed, gates the tick (not-printing/warmup/quality) without touching
    #              temporal state if gated, otherwise updates the EMA score and vote window,
    #              evaluates warn/pause/cancel level conditions and consecutive-tick counters,
    #              enforces cooldown/hysteresis, clamps any candidate action to the configured
    #              action_mode ceiling, and returns the resulting Decision.
    # Side Effects: Mutates nearly all of the engine's temporal state (self._state, self._score,
    #               self._vote_window, self._votes, self._warn_consec, self._pause_consec,
    #               self._cancel_consec, self._warn_latch, self._cooldown_until,
    #               self._clear_consec, self._prev_print_state) -- except on a gated tick (Step
    #               2), which deliberately leaves EMA/vote/consecutive state untouched. May call
    #               self.reset() via _apply_identity_reset. Does not itself call Moonraker,
    #               notify, or storage -- those are the caller's (ArgusService's) responsibility
    #               based on the returned Decision.action.
    def tick(
        self,
        *,
        p_raw: float,
        detections: tuple[Detection, ...],
        print_state: Optional[PrintState],
        quality: GateResult,
        now: float,
    ) -> Decision:
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

    # None _apply_identity_reset(self, Optional[PrintState] print_state)
    # Inputs: Optional[PrintState] print_state - the latest Moonraker print snapshot for this
    #                                             tick, or None if unavailable
    # Outputs: None
    # Description: Step 1 of tick(). Resets all temporal accumulators (via self.reset()) only on
    #              a POSITIVELY CONFIRMED change of print identity -- a genuinely different
    #              non-None filename, or a confirmed transition out of PRINTING into a terminal
    #              state (COMPLETE/CANCELLED/ERROR/STANDBY). Deliberately does NOT reset on
    #              `print_state is None`, `state is UNKNOWN`, or `state is PAUSED`, since those
    #              mean "state could not be determined" or "same job merely paused", not "the
    #              print ended" -- see the full safety rationale in this method's docstring.
    # Side Effects: May call self.reset(), wiping every temporal accumulator. Always mutates
    #               self._prev_print_state to `print_state`.
    def _apply_identity_reset(self, print_state: Optional[PrintState]) -> None:
        # Resetting on UNKNOWN (Moonraker's fail-closed response to any transient
        # error) would let a flaky poll wipe accumulators mid-print and block
        # detection; only a positively-confirmed identity change resets.
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

    # Optional[str] _gate_reason(self, Optional[PrintState] print_state, GateResult quality)
    # Inputs: Optional[PrintState] print_state - current Moonraker print snapshot, or None
    #         GateResult quality - pre-inference quality gate outcome for the frame behind this
    #                               tick
    # Outputs: Optional[str] - a short machine-readable gate reason ("not_printing", "warmup", or
    #                          "quality:<reason>") if this tick must not be scored, else None
    # Description: Step 2 of tick(). Determines whether the current tick is even eligible to be
    #              scored: the printer must be confirmed printing, past the configured warmup
    #              period, and the frame must have passed the quality gate.
    # Side Effects: None
    def _gate_reason(self, print_state: Optional[PrintState], quality: GateResult) -> Optional[str]:
        cfg = self._cfg
        if print_state is None or not print_state.is_printing:
            return "not_printing"
        if print_state.elapsed_s < cfg.warmup_s:
            return "warmup"
        if not quality.passed:
            return f"quality:{quality.reason}"
        return None

    # Action _clamp(Action action, DecisionConfig cfg)
    # Inputs: Action action - the candidate pre-clamp action (PAUSE or CANCEL)
    #         DecisionConfig cfg - carries the action_mode safety ceiling
    # Outputs: Action - `action` downgraded to at most what `cfg.action_mode` permits (NOTIFY
    #                   under NOTIFY_ONLY, PAUSE under PAUSE, unchanged under CANCEL)
    # Description: Step 7 of tick(). Enforces the configured action_mode as a hard ceiling on
    #              automated action, independent of the state machine's own transitions.
    # Side Effects: None
    @staticmethod
    def _clamp(action: Action, cfg: DecisionConfig) -> Action:
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
