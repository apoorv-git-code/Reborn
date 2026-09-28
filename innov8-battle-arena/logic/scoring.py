"""scoring.py - zero-credit local ranking + strict eligibility checks."""
from __future__ import annotations

from typing import Any, Optional, Sequence

import pandas as pd

from fraud import FRAUD_THRESHOLD, fraud_risk
from normalize import normalize_city, normalize_profile, normalize_requisition

STRICT_UNKNOWN = True          # unverifiable budget / bar / min-exp => ineligible (protects 10-credit offers)
MIN_MUST_COVERAGE = 0.6        # default share of must-have skills required (req["min_skill_coverage"] overrides)
EST_EXPECTED_HIKE = 1.3        # expected CTC estimate = current CTC * 1.3 when expected is missing
W_SKILL, W_EXP, W_ASSESS, W_CITY, W_NOTICE, W_BUDGET = 35.0, 20.0, 25.0, 10.0, 5.0, 5.0  # sums to 100


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def _coverage(have: set, want: Sequence[str]) -> Optional[float]:
    if not want:
        return None
    return sum(1 for w in want if w in have) / len(want)


def _skill_score(have: set, r: dict) -> Optional[float]:
    m, n = _coverage(have, r["must_skills"]), _coverage(have, r["nice_skills"])
    if m is not None and n is not None:
        return 0.75 * m + 0.25 * n
    return m if m is not None else n


def _exp_fit(exp: Optional[float], lo: Optional[float], hi: Optional[float]) -> Optional[float]:
    if lo is None and hi is None:
        return 1.0
    if exp is None:
        return None
    if lo is not None and exp < lo:
        return _clip((exp / lo) ** 2) if lo > 0 else 1.0
    if hi is not None and exp > hi:
        return _clip(1.0 - 0.12 * (exp - hi), 0.3, 1.0)
    return 1.0


def _city_fit(city: Optional[str], relocate: Optional[bool], r: dict) -> float:
    if r["remote_ok"] or not r["cities"]:
        return 1.0
    if city is None:
        return 0.4
    if city == "remote":
        return 0.8
    if city in r["cities"]:
        return 1.0
    return 0.6 if relocate else 0.0


def prescore_summary(summary: dict, req: dict) -> float:
    """Fast local rank on listing-level data (skills, exp, city) -> [0, 1]. Zero credits."""
    try:
        p, r = normalize_profile(summary), normalize_requisition(req)
        sk = _skill_score(set(p["skills"]), r)
        sk = 0.5 if sk is None else (sk if p["skills"] else min(sk, 0.3))
        ex = _exp_fit(p["exp_years"], r["min_exp"], r["max_exp"])
        ex = 0.35 if ex is None else ex
        ci = _city_fit(p["city"], p["relocate"], r)
        base = 0.5 * sk + 0.3 * ex + 0.2 * ci
        return round(_clip(base * (1.0 - 0.85 * fraud_risk(summary))), 4)
    except Exception:
        return 0.0


def rank_summaries(summaries: Sequence[dict], req: dict, top_k: Optional[int] = None) -> list[tuple[int, float]]:
    """Vectorised ranking: [(index, prescore)] best first."""
    r = normalize_requisition(req)
    s = pd.Series([prescore_summary(x, r) for x in (summaries or [])], dtype="float64")
    s = s.sort_values(ascending=False, kind="mergesort")
    if top_k is not None:
        s = s.head(int(top_k))
    return [(int(i), float(v)) for i, v in s.items()]


def _evaluate(profile: dict, req: dict, fraud_threshold: float) -> dict:
    p, r = normalize_profile(profile), normalize_requisition(req)
    risk = fraud_risk(profile)
    have = set(p["skills"])
    reasons: list[str] = []
    eligible = True

    if risk >= fraud_threshold:
        eligible = False
        reasons.append("fraud")

    # must-have skills
    m = _coverage(have, r["must_skills"])
    min_cov = r["min_skill_coverage"] if r["min_skill_coverage"] is not None else MIN_MUST_COVERAGE
    if m is not None and m + 1e-9 < min_cov:
        eligible = False
        reasons.append("must_skills")

    # minimum experience
    exp = p["exp_years"]
    if r["min_exp"] is not None:
        if exp is None:
            if STRICT_UNKNOWN:
                eligible = False
                reasons.append("exp_unverified")
        elif exp + 1e-9 < r["min_exp"]:
            eligible = False
            reasons.append("exp_below_min")

    # budget (expected CTC must not exceed it)
    ctc, ctc_estimated = p["expected_ctc"], False
    if ctc is None and p["current_ctc"] is not None:
        ctc, ctc_estimated = int(p["current_ctc"] * EST_EXPECTED_HIKE), True
    if r["budget"] is not None:
        if ctc is None:
            if STRICT_UNKNOWN:
                eligible = False
                reasons.append("budget_unverified")
        elif ctc > r["budget"]:
            eligible = False
            reasons.append("over_budget")

    # assessment bar
    score = p["score"]
    if r["bar"] is not None:
        if score is None:
            if STRICT_UNKNOWN:
                eligible = False
                reasons.append("bar_unverified")
        elif score + 1e-9 < r["bar"]:
            eligible = False
            reasons.append("below_bar")

    # notice
    nd = p["notice_days"]
    if r["max_notice"] is not None and nd is not None and nd > r["max_notice"]:
        eligible = False
        reasons.append("notice_too_long")

    # hard location filter (opt-in)
    if r["city_strict"] and r["cities"] and not r["remote_ok"]:
        if p["city"] not in r["cities"] and p["city"] != "remote" and not p["relocate"]:
            eligible = False
            reasons.append("city_mismatch")

    # ---- points (0-100) ----
    sk = _skill_score(have, r)
    sk = 0.0 if sk is None and r["must_skills"] else (0.5 if sk is None else sk)
    ex = _exp_fit(exp, r["min_exp"], r["max_exp"])
    ex = 0.4 if ex is None else ex
    asm = 0.3 if score is None else _clip(score / 100.0)
    ci = _city_fit(p["city"], p["relocate"], r)
    no = 0.5 if nd is None else _clip(1.0 - nd / 90.0)
    if r["budget"] is None:
        bu = 0.5
    elif ctc is None:
        bu = 0.3
    else:
        bu = _clip(1.0 - ctc / r["budget"]) if r["budget"] > 0 else 0.0
    raw = W_SKILL * sk + W_EXP * ex + W_ASSESS * asm + W_CITY * ci + W_NOTICE * no + W_BUDGET * bu
    points = 0.0 if risk >= fraud_threshold else raw * (1.0 - 0.8 * risk)

    # ---- confidence: share of decision-relevant fields actually present ----
    presence = (
        0.25 * bool(p["skills"]) + 0.20 * (exp is not None) + 0.20 * (score is not None)
        + 0.15 * (p["expected_ctc"] is not None or p["current_ctc"] is not None)
        + 0.10 * (p["city"] is not None) + 0.10 * (nd is not None)
    )
    if ctc_estimated:
        presence -= 0.05
    confidence = _clip(presence) * (1.0 - risk)

    return {
        "points": round(_clip(points, 0.0, 100.0), 3),
        "confidence": round(_clip(confidence), 4),
        "eligible": bool(eligible),
        "reasons": reasons,
        "fraud_risk": risk,
    }


def fit_score_detail(profile: dict, req: dict, fraud_threshold: float = FRAUD_THRESHOLD) -> dict:
    """Same as fit_score plus 'reasons' (why ineligible) and 'fraud_risk' for logging."""
    try:
        return _evaluate(profile, req, fraud_threshold)
    except Exception:
        return {"points": 0.0, "confidence": 0.0, "eligible": False, "reasons": ["error"], "fraud_risk": 1.0}


def fit_score(profile: dict, req: dict) -> dict:
    """{"points": 0-100, "confidence": 0-1, "eligible": bool}; eligible only if fraud-free,
    meets min experience + must-have skills, expected CTC <= budget, and assessment >= bar."""
    d = fit_score_detail(profile, req)
    return {"points": d["points"], "confidence": d["confidence"], "eligible": d["eligible"]}