"""fraud.py - heuristic fabricated-profile detector. fraud_risk() -> [0.0, 1.0], never raises."""
from __future__ import annotations

import re
from typing import Any

from normalize import CURRENT_YEAR, keymap, normalize_profile, parse_bool

FRAUD_THRESHOLD = 0.6  # >= this => treat as fabricated (score 0 points)

_PLACEHOLDER = frozenset({
    "test", "dummy", "fake", "asdf", "qwerty", "xxx", "sample", "abc", "foo", "bar", "lorem", "ipsum",
    "candidate", "user", "na", "unknown", "name", "firstname", "lastname", "xyz", "john", "jane", "doe",
})
_SENIOR_RE = re.compile(r"\b(senior|sr|lead|principal|staff|architect|head|director|vp|manager|chief|cto|ceo)\b")
_JUNIOR_RE = re.compile(r"\b(intern|trainee|fresher|junior|jr|apprentice)\b")
_UNEMPLOYED_RE = re.compile(r"unemploy|not\s*working|between\s*jobs|fresher|^false$|^no$")
_DISPOSABLE = frozenset({
    "mailinator.com", "tempmail.com", "guerrillamail.com", "10minutemail.com", "yopmail.com",
    "trashmail.com", "example.com", "test.com", "fakeinbox.com", "throwawaymail.com",
})
_EMAIL_RE = re.compile(r"^[a-z0-9._%+\-]+@[a-z0-9\-]+(?:\.[a-z0-9\-]+)+$")
_EXPLICIT_FLAGS = ("is_fake", "fake", "fabricated", "is_fabricated", "flagged", "fraud", "is_fraud", "suspicious")


def _flags(profile: Any) -> list[tuple[str, float]]:
    p = normalize_profile(profile)
    km = keymap(profile)
    f: list[tuple[str, float]] = []
    exp, age, gy = p["exp_years"], p["age"], p["grad_year"]
    cur, expc = p["current_ctc"], p["expected_ctc"]

    # explicit markers
    if any(parse_bool(km.get(k)) is True for k in _EXPLICIT_FLAGS):
        f.append(("explicit_flag", 0.95))

    # age vs experience
    if age is not None:
        if age < 18:
            f.append(("age_under_18", 0.6))
        elif age > 75:
            f.append(("age_over_75", 0.5))
        if exp is not None:
            if exp > age - 16:
                f.append(("exp_exceeds_working_age", 0.95))
            elif exp > age - 18.5:
                f.append(("exp_starts_before_18", 0.45))
    if exp is not None:
        if exp > 45:
            f.append(("exp_absurd", 0.9))
        elif exp > 40:
            f.append(("exp_implausible", 0.5))

    # graduation year contradictions
    if gy is not None:
        if exp is not None:
            since = CURRENT_YEAR - gy
            if exp > since + 2.0:
                f.append(("exp_exceeds_years_since_graduation", 0.8))
            elif exp > since + 1.0:
                f.append(("exp_slightly_exceeds_graduation", 0.35))
            if gy > CURRENT_YEAR + 1 and exp >= 1:
                f.append(("future_graduation_with_experience", 0.7))
        if age is not None and age - (CURRENT_YEAR - gy) < 17:
            f.append(("graduated_before_17", 0.6))

    # assessment patterns
    vals = list(p["scores"].values())
    n = len(vals)
    if n >= 2 and min(vals) >= 99.5:
        f.append(("perfect_scores_everywhere", 0.8))
    elif n >= 3 and min(vals) >= 97:
        f.append(("near_perfect_scores", 0.6))
    elif n == 1 and vals[0] >= 99.5:
        f.append(("single_perfect_score", 0.15))
    if n >= 4 and (max(vals) - min(vals)) < 1e-9:
        f.append(("identical_scores", 0.4))

    # compensation contradictions
    if cur is not None:
        if cur > 100_000_000:
            f.append(("ctc_absurd", 0.9))
        if exp is not None:
            if exp < 1 and cur > 3_000_000:
                f.append(("fresher_high_ctc", 0.6))
            if exp >= 1 and cur / exp > 15_000_000:
                f.append(("ctc_per_year_exp_absurd", 0.5))
    if expc is not None:
        if expc > 100_000_000:
            f.append(("expected_ctc_absurd", 0.8))
        if cur:
            if expc > 4 * cur:
                f.append(("expected_ctc_over_4x", 0.3))
            elif expc < 0.5 * cur:
                f.append(("expected_ctc_below_half", 0.2))

    # notice / employment contradictions
    nd = p["notice_days"]
    if nd is not None:
        if nd > 180:
            f.append(("notice_over_180", 0.3))
        if nd > 30 and p["employment_status"] and _UNEMPLOYED_RE.search(p["employment_status"]):
            f.append(("unemployed_with_notice", 0.3))

    # title vs experience
    title = p["title"]
    if title and exp is not None:
        if _SENIOR_RE.search(title) and exp < 2:
            f.append(("senior_title_low_exp", 0.5))
        if _JUNIOR_RE.search(title) and exp >= 10:
            f.append(("junior_title_high_exp", 0.4))

    # skill stuffing
    ns = len(p["skills"])
    if ns > 60:
        f.append(("skill_stuffing", 0.6))
    elif ns > 35:
        f.append(("many_skills", 0.3))

    # identity sanity
    name = p["name"]
    if not name:
        f.append(("missing_name", 0.15))
    else:
        toks = re.findall(r"[a-z]+", name.lower())
        if toks and all(t in _PLACEHOLDER for t in toks):
            f.append(("placeholder_name", 0.9))
        elif re.search(r"\d", name):
            f.append(("digits_in_name", 0.5))
        elif len("".join(toks)) <= 1:
            f.append(("single_char_name", 0.6))

    email = p["email"]
    if email:
        if not _EMAIL_RE.match(email):
            f.append(("invalid_email", 0.4))
        elif email.rsplit("@", 1)[-1] in _DISPOSABLE:
            f.append(("disposable_email", 0.6))
    digits = re.sub(r"\D", "", p["phone"])
    if digits:
        d10 = digits[-10:] if len(digits) >= 10 else digits
        if len(d10) != 10 or len(digits) > 13:
            f.append(("invalid_phone_length", 0.4))
        elif len(set(d10)) <= 2 or d10 in ("1234567890", "0123456789", "9876543210"):
            f.append(("fake_phone_pattern", 0.7))
        elif d10[0] in "012345":
            f.append(("non_mobile_prefix", 0.3))

    # mostly-empty profile
    missing = sum([not name, p["city"] is None, not p["skills"], exp is None])
    if missing >= 3:
        f.append(("mostly_empty", 0.4))
    return f


def fraud_report(profile: Any) -> dict:
    try:
        flags = _flags(profile)
    except Exception:
        return {"risk": 0.3, "flags": [("internal_error", 0.3)]}
    keep = 1.0
    for _, w in flags:
        keep *= 1.0 - w
    return {"risk": round(min(max(1.0 - keep, 0.0), 1.0), 4), "flags": flags}


def fraud_risk(profile: dict) -> float:
    """Noisy-OR combination of independent red flags, in [0.0, 1.0]."""
    return float(fraud_report(profile)["risk"])


def is_fabricated(profile: dict, threshold: float = FRAUD_THRESHOLD) -> bool:
    return fraud_risk(profile) >= threshold