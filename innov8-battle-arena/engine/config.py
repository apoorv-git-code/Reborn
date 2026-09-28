# config.py
from __future__ import annotations

import os
import sys


def _req(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        sys.stderr.write(f"FATAL: environment variable {name} is required\n")
        raise SystemExit(2)
    return val


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


ARENA_KEY: str = _req("ARENA_KEY")
PENALTY_FACTOR: float = _float("PENALTY_FACTOR", 1.0)

# ---- budget thresholds -------------------------------------------------
STARTING_CREDITS: int = int(_float("STARTING_CREDITS", 10_000))
CLOSING_RESERVE: int = int(_float("CLOSING_RESERVE", 2000))

# ---- credit costs ------------------------------------------------------
COST_REQUISITIONS: int = 0
COST_SEARCH: int = 1
COST_BATCH: int = 60
BATCH_SIZE: int = 50
COST_OFFER: int = 10
COST_OFFER_CLOSING: int = 20
COST_MARKET: int = 2

# ---- ROI model (points per unit) --------------------------------------
DEFAULT_REQ_VALUE: float = _float("DEFAULT_REQ_VALUE", 100.0)
OFFER_ACCEPT_PROB_FLOOR: float = _float("OFFER_ACCEPT_PROB_FLOOR", 0.05)
BATCH_YIELD: float = _float("BATCH_YIELD", 0.35)    # share of prescore mass that converts
SEARCH_YIELD: float = _float("SEARCH_YIELD", 0.02)  # share of prescore mass that converts
MIN_FIT_TO_QUEUE: float = _float("MIN_FIT_TO_QUEUE", 0.5)

# ---- search / recon shape ---------------------------------------------
MAX_SEARCH_PAGES_PER_REQ: int = int(_float("MAX_SEARCH_PAGES_PER_REQ", 6))
TOP_PER_REQ: int = int(_float("TOP_PER_REQ", 50))

# ---- timing (local wall clock, HH:MM 24h; server phase wins if given) --
MARKET_OPEN: str = os.environ.get("MARKET_OPEN", "11:30")
CLOSING_START: str = os.environ.get("CLOSING_START", "16:00")
MARKET_POLL_SECONDS: float = _float("MARKET_POLL_SECONDS", 20.0)
LOOP_SLEEP_SECONDS: float = _float("LOOP_SLEEP_SECONDS", 1.0)
EXHAUSTED_BACKOFF_SECONDS: float = _float("EXHAUSTED_BACKOFF_SECONDS", 5.0)
WRONG_PHASE_BACKOFF_SECONDS: float = _float("WRONG_PHASE_BACKOFF_SECONDS", 2.0)
FILLING_THRESHOLD: float = _float("FILLING_THRESHOLD", 0.7)  # filled/capacity

# ---- persistence -------------------------------------------------------
STATE_PATH: str = os.environ.get("STATE_PATH", "state.json")
STATE_SAVE_INTERVAL: float = _float("STATE_SAVE_INTERVAL", 5.0)