"""dedupe.py - detect the same human under different candidate IDs."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Callable, Optional, Sequence

import pandas as pd

from normalize import is_missing, name_tokens, normalize_profile


def _exp_bucket(exp: Optional[float]) -> str:
    return "x" if exp is None else str(int(math.floor(exp + 0.5)))


def person_key(profile: dict) -> str:
    """Deterministic 16-hex key from normalized name tokens + city + rounded experience.

    Name tokens are sorted and initials dropped, so 'Sharma, Rahul K.' == 'rahul sharma'.
    Profiles without a usable name get an ID-based key so unrelated anonymous rows never merge.
    """
    try:
        n = normalize_profile(profile)
        tokens = name_tokens(n["name"])
        if not tokens:
            base = f"id:{n['id']}" if n["id"] else "anon:" + json.dumps(profile, sort_keys=True, default=str)
            return hashlib.sha1(base.encode("utf-8", "ignore")).hexdigest()[:16]
        raw = "|".join((" ".join(tokens), n["city"] or "unk", _exp_bucket(n["exp_years"])))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return hashlib.sha1(repr(profile).encode("utf-8", "ignore")).hexdigest()[:16]


def _completeness(profile: Any) -> int:
    if not isinstance(profile, dict):
        return 0
    return sum(0 if is_missing(v) else 1 for v in profile.values())


def group_duplicates(profiles: Sequence[dict], exp_tolerance: float = 1.0) -> list[list[int]]:
    """Cluster indices of likely-same people: same name tokens + city, experience within tolerance.

    Catches boundary cases the rounded hash misses (4.4 vs 4.6 years).
    """
    rows = []
    for i, p in enumerate(profiles or []):
        n = normalize_profile(p)
        toks = name_tokens(n["name"])
        if not toks:
            continue
        exp = n["exp_years"]
        rows.append({"i": i, "name": " ".join(toks), "city": n["city"] or "", "exp": -999.0 if exp is None else float(exp)})
    if not rows:
        return []
    df = pd.DataFrame(rows).sort_values(["name", "city", "exp"], kind="mergesort")
    df["cluster"] = df.groupby(["name", "city"], sort=False)["exp"].transform(
        lambda s: s.diff().gt(exp_tolerance).cumsum()
    )
    out: list[list[int]] = []
    for _, g in df.groupby(["name", "city", "cluster"], sort=False):
        if len(g) > 1:
            out.append(sorted(int(x) for x in g["i"]))
    return out


def dedupe_profiles(
    profiles: Sequence[dict],
    exp_tolerance: float = 1.0,
    quality: Optional[Callable[[dict], float]] = None,
) -> tuple[list[dict], dict[str, list[str]]]:
    """Return (unique_profiles, {kept_id: [dropped_ids]}). Keeps the most complete record."""
    profiles = list(profiles or [])
    score = quality or _completeness
    drop: set[int] = set()
    dropped_map: dict[str, list[str]] = {}
    for cluster in group_duplicates(profiles, exp_tolerance):
        best = max(cluster, key=lambda i: (score(profiles[i]), -i))
        kept_id = normalize_profile(profiles[best])["id"] or str(best)
        for i in cluster:
            if i != best:
                drop.add(i)
                dropped_map.setdefault(kept_id, []).append(normalize_profile(profiles[i])["id"] or str(i))
    return [p for i, p in enumerate(profiles) if i not in drop], dropped_map


class OfferGuard:
    """Live-market guard: never pay a 10-credit offer fee twice for the same person."""

    def __init__(self, exp_tolerance: float = 1.0) -> None:
        self.tol = exp_tolerance
        self._keys: set[str] = set()
        self._seen: dict[tuple[str, str], list[float]] = defaultdict(list)

    def _sig(self, profile: Any) -> tuple[str, str, Optional[float]]:
        n = normalize_profile(profile)
        return " ".join(name_tokens(n["name"])), n["city"] or "", n["exp_years"]

    def already_offered(self, profile: dict) -> bool:
        if person_key(profile) in self._keys:
            return True
        name, city, exp = self._sig(profile)
        if not name:
            return False
        prior = self._seen.get((name, city), [])
        if exp is None:
            return False
        return any(abs(exp - e) <= self.tol for e in prior)

    def mark_offered(self, profile: dict) -> None:
        self._keys.add(person_key(profile))
        name, city, exp = self._sig(profile)
        if name and exp is not None:
            self._seen[(name, city)].append(exp)