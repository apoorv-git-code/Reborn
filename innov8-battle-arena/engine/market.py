# market.py
from __future__ import annotations

import logging
import time
from typing import Any

import config
from budget import Governor
from dedupe import OfferGuard
from recon import (
    _get, _items, _num, _open_reqs, _post, _req_value, _sync_credits,
    fetch_profiles, search_req,
)
from state import State

log = logging.getLogger("market")

_SUCCESS_STATES = {"signed", "accepted", "ok", "success", "hired"}


def _guard_mark(guard: OfferGuard, cid: str, rid: str) -> None:
    for name in ("record", "mark", "add"):
        fn = getattr(guard, name, None)
        if callable(fn):
            try:
                fn(cid, rid)
                return
            except TypeError:
                try:
                    fn(f"{cid}|{rid}")
                    return
                except Exception:
                    pass
            except Exception:
                pass


def _parse_offer(resp: Any) -> tuple[bool, str]:
    if not isinstance(resp, dict):
        return False, "unknown"
    reason = str(resp.get("reason") or resp.get("error") or "").lower()
    status = str(resp.get("status") or "").lower()
    ok = bool(resp.get("ok") or resp.get("accepted") or resp.get("signed")) or status in _SUCCESS_STATES
    if reason:
        ok = False
    return ok, reason or status or ("signed" if ok else "unknown")


def fire_offer(arena: Any, state: State, gov: Governor, guard: OfferGuard, item: dict, phase: str) -> str:
    """Returns one of: signed, blocked, rejected, skipped."""
    cid, rid, person = item["cid"], item["rid"], item.get("person", "")
    key = f"{cid}|{rid}"
    if key in state.offered or person in state.signed_persons or rid in state.filled_reqs:
        return "skipped"
    cost = gov.offer_cost(phase)
    req = state.requisitions.get(rid, {})
    gain = _req_value(req) * max(config.OFFER_ACCEPT_PROB_FLOOR, min(1.0, _num(item.get("score"))))
    if not gov.approve(gain, cost, phase):
        return "blocked"
    state.offered.add(key)
    _guard_mark(guard, cid, rid)
    resp = _post(arena, "/offer", {"candidate_id": cid, "requisition_id": rid})
    gov.spent(cost)
    _sync_credits(arena, resp, gov, state)
    ok, reason = _parse_offer(resp)
    if ok:
        state.signed_persons.add(person)
        log.info("SIGNED %s -> %s (credits=%.0f)", cid, rid, gov.credits)
        state.save(force=True)
        return "signed"
    if reason == "requisition_full":
        state.filled_reqs.add(rid)
    elif reason in ("already_signed", "same_person_already_signed"):
        state.signed_persons.add(person)
    else:
        log.warning("offer rejected cid=%s rid=%s reason=%s", cid, rid, reason)
    state.save(force=True)
    return "rejected"


def poll_market(arena: Any, state: State, gov: Governor) -> dict:
    resp = _get(arena, "/market")
    gov.spent(config.COST_MARKET)
    _sync_credits(arena, resp, gov, state)
    state.last_market = time.time()
    focus: list[tuple[float, str]] = []
    rows = _items(resp, "requisitions", "market", "items")
    for row in rows:
        rid = str(row.get("id", row.get("requisition_id", "")))
        if not rid:
            continue
        cap = _num(row.get("capacity", row.get("slots", 0)))
        filled = _num(row.get("filled", row.get("signed", 0)))
        if cap > 0 and filled >= cap:
            state.filled_reqs.add(rid)
            continue
        frac = (filled / cap) if cap > 0 else 0.0
        if frac >= config.FILLING_THRESHOLD:
            focus.append((frac, rid))  # racing: fill it first
    focus.sort(reverse=True)
    state.focus = [rid for _, rid in focus]
    state.save()
    return resp if isinstance(resp, dict) else {"items": rows}


def run_market_step(arena: Any, state: State, gov: Governor, guard: OfferGuard, phase: str) -> bool:
    """One unit of market work. Returns True if anything was done."""
    did = False
    if (time.time() - state.last_market) >= config.MARKET_POLL_SECONDS and gov.can_afford(config.COST_MARKET, phase):
        poll_market(arena, state, gov)
        did = True

    # Fire offers, racing requisitions that are filling up first.
    focus = set(state.focus) if state.focus else None
    fired = 0
    while fired < 10:
        item = state.pop_ready(allow_reqs=focus) or (state.pop_ready() if focus else None)
        if item is None:
            break
        result = fire_offer(arena, state, gov, guard, item, phase)
        fired += 1
        did = True
        if result == "blocked":
            state.push_ready(item)
            break

    # Refill the queue for requisitions with thin ready queues (dynamic focus shift).
    if not state.ready and phase != "closing":
        targets = [r for r in state.focus if r not in state.filled_reqs] or _open_reqs(state)
        targets.sort(key=lambda r: -_req_value(state.requisitions[r]))
        for rid in targets[:2]:
            if len(state.shortlist.get(rid, [])) < config.BATCH_SIZE and rid not in state.search_done:
                if search_req(arena, state, gov, rid, phase) > 0:
                    did = True
            if fetch_profiles(arena, state, gov, guard, rid, phase) > 0:
                did = True
    state.save()
    return did