# budget.py
from __future__ import annotations

import logging
import threading

import config

log = logging.getLogger("budget")


def is_worth_buying(expected_points_gain: float, cost: float, penalty_factor: float) -> bool:
    """Never buy when expected_points_gain < cost * penalty_factor."""
    try:
        gain = float(expected_points_gain)
        c = float(cost)
        pf = float(penalty_factor)
    except (TypeError, ValueError):
        return False
    if gain != gain or c != c or pf != pf:  # NaN guard
        return False
    if c < 0 or pf < 0:
        return False
    return not (gain < c * pf)


class Governor:
    """Tracks credits and enforces ROI + closing reserve."""

    def __init__(self, credits: float | None = None, penalty_factor: float | None = None):
        self._lock = threading.Lock()
        self.credits: float = float(config.STARTING_CREDITS if credits is None else credits)
        self.penalty_factor: float = float(
            config.PENALTY_FACTOR if penalty_factor is None else penalty_factor
        )

    def sync(self, credits: float | None) -> None:
        if credits is None:
            return
        try:
            with self._lock:
                self.credits = float(credits)
        except (TypeError, ValueError):
            pass

    def spent(self, cost: float) -> None:
        with self._lock:
            self.credits = max(0.0, self.credits - float(cost))

    def floor(self, phase: str) -> float:
        """Credits that must remain after a purchase in this phase."""
        return 0.0 if phase == "closing" else float(config.CLOSING_RESERVE)

    def offer_cost(self, phase: str) -> int:
        return config.COST_OFFER_CLOSING if phase == "closing" else config.COST_OFFER

    def can_afford(self, cost: float, phase: str) -> bool:
        with self._lock:
            return (self.credits - float(cost)) >= self.floor(phase)

    def approve(self, expected_gain: float, cost: float, phase: str) -> bool:
        """ROI check AND reserve check. Both must pass before any paid call."""
        if not is_worth_buying(expected_gain, cost, self.penalty_factor):
            return False
        return self.can_afford(cost, phase)