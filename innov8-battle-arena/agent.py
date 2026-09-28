# agent.py
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime

import config
from arena_client import Arena, Exhausted, WrongPhase
from budget import Governor
from dedupe import OfferGuard
from market import run_market_step
from recon import run_recon_step
from state import State

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("agent")


def _hhmm(s: str) -> tuple[int, int]:
    h, m = s.split(":")
    return int(h), int(m)


def _clock_phase() -> str:
    now = datetime.now()
    cur = (now.hour, now.minute)
    if cur >= _hhmm(config.CLOSING_START):
        return "closing"
    if cur >= _hhmm(config.MARKET_OPEN):
        return "market"
    return "recon"


_ORDER = {"recon": 0, "market": 1, "closing": 2}


def _next_phase(current: str) -> str:
    """Phases only move forward; wall clock decides."""
    target = _clock_phase()
    return target if _ORDER[target] >= _ORDER.get(current, 0) else current


def main() -> None:
    try:
        arena = Arena(config.ARENA_KEY)
    except TypeError:
        arena = Arena(api_key=config.ARENA_KEY)

    guard = OfferGuard()
    state = State(guard)
    gov = Governor(credits=state.credits if state.credits is not None else config.STARTING_CREDITS)
    log.info("agent start phase=%s credits=%.0f", state.phase, gov.credits)

    while True:
        try:
            phase = _next_phase(state.phase)
            if phase != state.phase:
                state.set_phase(phase)

            if state.phase == "recon":
                more = run_recon_step(arena, state, gov, guard)
                if not more:
                    time.sleep(config.LOOP_SLEEP_SECONDS)
            else:
                did = run_market_step(arena, state, gov, guard, state.phase)
                if not did:
                    time.sleep(config.LOOP_SLEEP_SECONDS)

            state.credits = gov.credits
            state.save()

        except Exhausted:
            log.warning("credits exhausted; backing off")
            gov.sync(0)
            state.credits = 0
            state.save(force=True)
            time.sleep(config.EXHAUSTED_BACKOFF_SECONDS)
        except WrongPhase:
            log.warning("wrong phase for call; advancing per clock")
            nxt = _next_phase(state.phase)
            if nxt == state.phase and state.phase != "closing":
                nxt = "market" if state.phase == "recon" else "closing"
            state.set_phase(nxt)
            time.sleep(config.WRONG_PHASE_BACKOFF_SECONDS)
        except KeyboardInterrupt:
            state.save(force=True)
            raise
        except Exception:
            log.exception("unexpected error; continuing")
            state.save(force=True)
            time.sleep(config.LOOP_SLEEP_SECONDS)


if __name__ == "__main__":
    main()  