# recon.py
from __future__ import annotations

import logging
from typing import Any

import config
from budget import Governor
from dedupe import OfferGuard, person_key
from scoring import fit_score, prescore_summary
from state import State

log = logging.getLogger("recon")


# ---- adapter helpers (single place to adjust to the real client) -------
def _get(arena: Any, path: str, params: dict | None = None) -> Any:
    return arena.get(path, params=params) if params else arena.get(path)


def _post(arena: Any, path: str, body: dict | None = None) -> Any:
    return arena.post(path, json=body) if body is not None else arena.post(path)


def _items(resp: Any, *keys: str) -> list:
    if isinstance(resp, list):
        return resp
    if isinstance(resp, dict):
        for k in keys:
            v = resp.get(k)
            if isinstance(v, list):
                return v
    return []


def _num(x: Any, default: float = 0.0) -> float:
    if isinstance(x, dict):
        for k in ("score", "fit", "value"):
            if k in x:
                return _num(x[k], default)
        return default
    if isinstance(x, tuple) and x:
        return _num(x[0], default)
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _eligible(x: Any, score: float) -> bool:
    if isinstance(x, dict) and "eligible" in x:
        return bool(x["eligible"])
    if isinstance(x, tuple) and len(x) > 1 and isinstance(x[1], bool):
        return x[1]
    return score >= config.MIN_FIT_TO_QUEUE


def _sync_credits(arena: Any, resp: Any, gov: Governor, state: State) -> None:
    val = None
    if isinstance(resp, dict):
        val = resp.get("credits", resp.get("credits_remaining"))
    if val is None:
        val = getattr(arena, "credits", None)
    if val is not None:
        gov.sync(val)
        state.credits = gov.credits


def _cid(obj: dict) -> str | None:
    v = obj.get("id", obj.get("candidate_id"))
    return None if v is None else str(v)


def _rid(obj: dict) -> str | None:
    v = obj.get("id", obj.get("requisition_id"))
    return None if v is None else str(v)


def _req_value(req: dict) -> float:
    for k in ("points", "value", "reward"):
        if k in req:
            return _num(req[k], config.DEFAULT_REQ_VALUE)
    return config.DEFAULT_REQ_VALUE


def load_requisitions(arena: Any, state: State) -> None:
    resp = _get(arena, "/requisitions")
    for r in _items(resp, "requisitions", "items", "data"):
        rid = _rid(r)
        if rid:
            state.requisitions[rid] = r
    state.save(force=True)


def _open_reqs(state: State) -> list[str]:
    return [r for r in state.requisitions if r not in state.filled_reqs]


def search_req(arena: Any, state: State, gov: Governor, rid: str, phase: str = "recon") -> int:
    """Page /search for one requisition. Returns number of summaries added to shortlist."""
    req = state.requisitions[rid]
    added = 0
    while state.search_pages.get(rid, 0) < config.MAX_SEARCH_PAGES_PER_REQ and rid not in state.search_done:
        page = state.search_pages.get(rid, 0)
        # Expected gain of a page: modest, scaled by requisition value and unfilled need.
        gain = _req_value(req) * config.SEARCH_YIELD * 50
        if not gov.approve(gain, config.COST_SEARCH, phase):
            break
        resp = _get(arena, "/search", {"requisition_id": rid, "page": page})
        gov.spent(config.COST_SEARCH)
        _sync_credits(arena, resp, gov, state)
        summaries = _items(resp, "results", "candidates", "items")
        state.search_pages[rid] = page + 1
        if not summaries:
            state.search_done.add(rid)
            break
        scored: list[tuple[float, str]] = []
        for s in summaries:
            cid = _cid(s)
            if not cid or cid in state.fetched:
                continue
            try:
                sc = _num(prescore_summary(s, req))
            except Exception:
                log.exception("prescore failed for %s", cid)
                continue
            scored.append((sc, cid))
        scored.sort(reverse=True)
        bucket = state.shortlist.setdefault(rid, [])
        existing = set(bucket)
        for _, cid in scored:
            if cid not in existing:
                bucket.append(cid)
                existing.add(cid)
                added += 1
        # keep best TOP_PER_REQ by insertion (scored order preserved per page)
        del bucket[config.TOP_PER_REQ * 2:]
        state.save()
    return added


def fetch_profiles(arena: Any, state: State, gov: Governor, guard: OfferGuard, rid: str, phase: str = "recon") -> int:
    """Batch-fetch up to BATCH_SIZE shortlisted profiles for rid and score them."""
    bucket = state.shortlist.get(rid, [])
    ids = [c for c in bucket if c not in state.fetched][: config.BATCH_SIZE]
    if not ids:
        return 0
    req = state.requisitions[rid]
    gain = _req_value(req) * config.BATCH_YIELD * len(ids) * 0.5
    if not gov.approve(gain, config.COST_BATCH, phase):
        return 0
    resp = _post(arena, "/candidates/batch", {"ids": ids})
    gov.spent(config.COST_BATCH)
    _sync_credits(arena, resp, gov, state)
    profiles = _items(resp, "candidates", "profiles", "results", "items")
    queued = 0
    for p in profiles:
        cid = _cid(p)
        if not cid:
            continue
        state.fetched.add(cid)
        try:
            fs = fit_score(p, req)
            score = _num(fs)
            if not _eligible(fs, score):
                continue
            pk = str(person_key(p))
        except Exception:
            log.exception("fit_score failed for %s", cid)
            continue
        if pk in state.signed_persons:
            continue
        state.push_ready({"cid": cid, "rid": rid, "score": score, "person": pk})
        queued += 1
    state.shortlist[rid] = [c for c in bucket if c not in set(ids)]
    state.save()
    log.info("recon rid=%s fetched=%d queued=%d credits=%.0f", rid, len(ids), queued, gov.credits)
    return queued


def run_recon_step(arena: Any, state: State, gov: Governor, guard: OfferGuard) -> bool:
    """One unit of recon work. Returns True if more work remains."""
    if not state.requisitions:
        load_requisitions(arena, state)
    reqs = _open_reqs(state)
    if not reqs:
        state.recon_done = True
        return False
    # Highest-value requisitions with the thinnest ready queue first.
    ready_count = {r: sum(1 for x in state.ready if x["rid"] == r) for r in reqs}
    reqs.sort(key=lambda r: (ready_count[r], -_req_value(state.requisitions[r])))
    progressed = False
    for rid in reqs:
        if len(state.shortlist.get(rid, [])) < config.BATCH_SIZE and rid not in state.search_done:
            if search_req(arena, state, gov, rid) > 0:
                progressed = True
        if state.shortlist.get(rid):
            if fetch_profiles(arena, state, gov, guard, rid) > 0:
                progressed = True
    if not progressed:
        state.recon_done = True
    state.save()
    return progressed