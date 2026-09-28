"""normalize.py - defensive normalisation of messy candidate / requisition fields.

Every public function accepts None / "" / NaN / wrong types and never raises.
Assumptions (documented, tweak the alias tuples if the arena schema differs):
  * Money is INR. Bare numbers < 1000 are LAKHS (12 -> 12 LPA), otherwise rupees.
  * Bare notice numbers: <= 6 -> months, else days (unless the key says "days").
  * Bare scores: <=1 -> fraction, <=10 -> /10 scale, <=100 -> already 0-100.
"""
from __future__ import annotations

import functools
import math
import numbers
import re
import unicodedata
from datetime import datetime
from typing import Any, Iterable, Mapping, Optional

import pandas as pd

CURRENT_YEAR: int = datetime.now().year

_MISSING = frozenset({
    "", "none", "null", "nan", "n/a", "na", "-", "--", "?", "unknown", "undefined",
    "tbd", "not available", "not provided", "not disclosed", "not specified",
})

# One number: Indian (12,00,000) / western (1,200,000) grouping, decimals, or plain.
_NUM = r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+"
_RANGE_SEP = frozenset({"-", "–", "—", "to", "~", "and", "upto", "up to"})


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def is_missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in _MISSING
    if isinstance(value, (list, tuple, set, frozenset, dict)):
        return len(value) == 0
    return False


def _key(k: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(k).lower()).strip("_")


def keymap(obj: Any) -> dict:
    """Lower-snake-case key -> value view of a dict / Series (first key wins)."""
    if isinstance(obj, pd.Series):
        obj = obj.to_dict()
    out: dict = {}
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            out.setdefault(_key(k), v)
    return out


def pick(km: Mapping, names: Iterable[str]) -> Any:
    """First non-missing value among alias names, else None."""
    for n in names:
        v = km.get(_key(n))
        if not is_missing(v):
            return v
    return None


def pick_with_key(km: Mapping, names: Iterable[str]) -> tuple[Optional[str], Any]:
    for n in names:
        k = _key(n)
        v = km.get(k)
        if not is_missing(v):
            return k, v
    return None, None


def get_field(profile: Any, *names: str, default: Any = None) -> Any:
    v = pick(keymap(profile), names)
    return default if v is None else v


def _num(text: str) -> float:
    return float(text.replace(",", ""))


def _is_real(v: Any) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(float(v))


def parse_bool(value: Any) -> Optional[bool]:
    if is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if _is_real(value):
        return float(value) != 0.0
    s = str(value).strip().lower()
    if s in {"yes", "y", "true", "t", "1", "open", "willing", "ok", "okay"}:
        return True
    if s in {"no", "n", "false", "f", "0", "not willing", "unwilling"}:
        return False
    return None


def _as_text(value: Any, limit: int = 2000) -> str:
    if is_missing(value):
        return ""
    if isinstance(value, Mapping):
        value = " ".join(str(v) for v in value.values() if not is_missing(v))
    elif isinstance(value, (list, tuple, set)):
        value = " | ".join(str(v) for v in value if not is_missing(v))
    return re.sub(r"\s+", " ", str(value)).strip()[:limit]


# --------------------------------------------------------------------------- #
# Skills
# --------------------------------------------------------------------------- #
_SKILL_GROUPS: dict[str, list[str]] = {
    "javascript": ["js", "java script", "ecmascript", "es6", "es2015", "vanilla js", "vanilla javascript"],
    "typescript": ["ts"],
    "python": ["py", "python3", "python 3", "python2", "cpython"],
    "java": ["core java", "java8", "java 8", "java11", "java 11", "j2ee", "jdk"],
    "kubernetes": ["k8s", "kube"],
    "docker": ["dockers", "docker containers", "containerization"],
    "postgresql": ["postgres", "psql", "pgsql", "postgre sql"],
    "mysql": ["my sql"],
    "mongodb": ["mongo", "mongo db"],
    "sql server": ["mssql", "ms sql", "microsoft sql server", "t-sql", "tsql"],
    "aws": ["amazon web services", "amazon aws"],
    "gcp": ["google cloud", "google cloud platform"],
    "azure": ["microsoft azure", "ms azure"],
    "machine learning": ["ml", "machinelearning"],
    "deep learning": ["dl"],
    "natural language processing": ["nlp"],
    "computer vision": ["cv", "opencv"],
    "artificial intelligence": ["ai"],
    "large language models": ["llm", "llms", "large language model"],
    "generative ai": ["genai", "gen ai", "generative artificial intelligence"],
    "react": ["reactjs", "react.js", "react js"],
    "react native": ["react-native", "reactnative"],
    "nodejs": ["node", "node.js", "node js"],
    "vue": ["vuejs", "vue.js", "vue js"],
    "angular": ["angularjs", "angular.js", "angular js"],
    "nextjs": ["next.js", "next js"],
    "express": ["expressjs", "express.js"],
    "c++": ["cpp", "c plus plus", "cplusplus"],
    "c#": ["csharp", "c sharp", "c-sharp"],
    "golang": ["go", "go lang", "go-lang"],
    "dotnet": [".net", "net", "asp.net", "asp net", ".net core", "dotnet core", "net core"],
    "sql": ["structured query language"],
    "nosql": ["no sql", "no-sql"],
    "power bi": ["powerbi", "power-bi", "microsoft power bi"],
    "excel": ["ms excel", "microsoft excel", "advanced excel"],
    "scikit-learn": ["sklearn", "scikit learn", "scikitlearn"],
    "tensorflow": ["tf", "tensor flow"],
    "pytorch": ["torch", "py torch"],
    "spark": ["apache spark", "pyspark", "py spark"],
    "kafka": ["apache kafka"],
    "airflow": ["apache airflow"],
    "hadoop": ["apache hadoop"],
    "elasticsearch": ["elastic search", "elk", "elastic"],
    "ci/cd": ["cicd", "ci cd", "ci-cd", "continuous integration", "continuous delivery"],
    "rest api": ["rest", "restful", "restful api", "rest apis", "restful apis", "rest-api"],
    "graphql": ["graph ql"],
    "spring boot": ["springboot", "spring-boot", "spring"],
    "html": ["html5"],
    "css": ["css3"],
    "linux": ["unix", "gnu/linux", "ubuntu"],
    "git": ["github", "gitlab", "version control"],
    "terraform": ["tf infra", "iac"],
    "ruby on rails": ["rails", "ror", "ruby rails"],
    "devops": ["dev ops", "dev-ops"],
    "r": ["r language", "r programming", "rlang"],
    "postman": ["postman api"],
    "tableau": ["tableau desktop"],
}


def _compact(t: str) -> str:
    return re.sub(r"[\s._\-]+", "", t)


_SKILL_LOOKUP: dict[str, str] = {}
for _canon, _aliases in _SKILL_GROUPS.items():
    _SKILL_LOOKUP[_compact(_canon)] = _canon
    for _a in _aliases:
        _SKILL_LOOKUP.setdefault(_compact(_a), _canon)


@functools.lru_cache(maxsize=16384)
def normalize_skill(token: str) -> str:
    """'K8s' -> 'kubernetes', 'JS (Advanced)' -> 'javascript'; unknown skills lower-cased."""
    t = str(token or "").lower().strip()
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", t)
    t = re.sub(r"\b(beginner|intermediate|advanced|expert|basic|proficient|strong|hands[- ]on)\b", " ", t)
    t = re.sub(r"\s*[:\-–]\s*\d+(?:\.\d+)?\s*(?:years?|yrs?|y)?\b.*$", "", t)
    t = re.sub(r"[^\w+#./&\s-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" .-/")
    if not t:
        return ""
    return _SKILL_LOOKUP.get(_compact(t), t)


def _iter_skill_tokens(value: Any, depth: int = 0) -> Iterable[str]:
    if depth > 3 or is_missing(value):
        return
    if isinstance(value, str):
        for part in re.split(r"[,;|\n\r\t•·●▪]+|\s+/\s+|\s+&\s+|\s+and\s+", value):
            if part.strip():
                yield part
    elif isinstance(value, Mapping):
        named = pick(keymap(value), ("name", "skill", "title", "label"))
        if named is not None:
            yield from _iter_skill_tokens(named, depth + 1)
        else:
            for k in value.keys():
                yield from _iter_skill_tokens(str(k), depth + 1)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for v in value:
            yield from _iter_skill_tokens(v, depth + 1)
    elif _is_real(value) or isinstance(value, bool):
        return
    else:
        yield str(value)


def normalize_skills(value: Any) -> list[str]:
    """Any messy skills payload -> ordered, de-duplicated list of canonical skills."""
    seen: set[str] = set()
    out: list[str] = []
    for tok in _iter_skill_tokens(value):
        t = normalize_skill(tok)
        if not t:
            continue
        parts = [t]
        if "/" in t and _compact(t) not in _SKILL_LOOKUP:
            parts = [normalize_skill(p) for p in t.split("/") if p.strip()]
        for p in parts:
            if p and len(p) <= 40 and p not in seen:
                seen.add(p)
                out.append(p)
        if len(out) >= 100:
            break
    return out


# --------------------------------------------------------------------------- #
# City / name
# --------------------------------------------------------------------------- #
_CITY_ALIASES = {
    "bangalore": "bengaluru", "bengaluru urban": "bengaluru", "blr": "bengaluru", "bangaluru": "bengaluru",
    "bombay": "mumbai", "madras": "chennai", "calcutta": "kolkata",
    "gurgaon": "gurugram", "new delhi": "delhi", "delhi ncr": "delhi", "ncr": "delhi", "nct": "delhi",
    "secunderabad": "hyderabad", "hyd": "hyderabad", "poona": "pune",
    "trivandrum": "thiruvananthapuram", "cochin": "kochi", "mysore": "mysuru",
    "vizag": "visakhapatnam", "baroda": "vadodara", "pondicherry": "puducherry",
    "allahabad": "prayagraj", "bhubaneshwar": "bhubaneswar", "wfh": "remote",
    "work from home": "remote", "anywhere": "remote", "pan india": "remote",
}


@functools.lru_cache(maxsize=4096)
def _city_str(s: str) -> Optional[str]:
    t = s.lower().strip()
    t = re.sub(r"\(.*?\)", " ", t)
    t = t.split(",")[0]
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+city$", "", t)
    if not t:
        return None
    return _CITY_ALIASES.get(t, t)


def normalize_city(value: Any) -> Optional[str]:
    if isinstance(value, (list, tuple, set)):
        for v in value:
            c = normalize_city(v)
            if c:
                return c
        return None
    if is_missing(value) or not isinstance(value, str):
        return None
    return _city_str(value)


_TITLES = frozenset({"mr", "mrs", "ms", "miss", "dr", "shri", "smt", "sri", "prof", "er", "mx"})


def name_tokens(name: Any, drop_initials: bool = True) -> tuple[str, ...]:
    """Order-insensitive canonical tokens: 'Sharma, Rahul K.' == 'rahul sharma'."""
    if is_missing(name):
        return ()
    s = unicodedata.normalize("NFKD", str(name)).lower()
    if re.search(r"[a-z]", s):
        s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[\d_]+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    tokens = [t for t in s.split() if t not in _TITLES]
    if drop_initials:
        longer = [t for t in tokens if len(t) > 1]
        if len(longer) >= 2:
            tokens = longer
    return tuple(sorted(tokens))


def normalize_name(value: Any) -> str:
    return " ".join(name_tokens(value))


# --------------------------------------------------------------------------- #
# Numeric field parsers
# --------------------------------------------------------------------------- #
_SAL_RE = re.compile(
    rf"(?P<n>(?:{_NUM}))(?:\s*(?P<u>crores?|cr|lakhs?|lacs?|lpa|l|thousand|k|million|mn|m)(?![a-z]))?"
)
_MONTHLY_RE = re.compile(r"per\s*month|/\s*mo(?:nth)?\b|monthly|\bpm\b|p\.m\b|a\s+month|per\s*mo\b")
_SAL_MIN, _SAL_MAX = 10_000, 10_000_000_000


def _salary_amount(n: float, unit: Optional[str]) -> float:
    u = (unit or "").lower()
    if u.startswith("cr"):
        return n * 1e7
    if u in ("lpa", "l") or u.startswith(("lakh", "lac")):
        return n * 1e5
    if u in ("k", "thousand"):
        return n * 1e3
    if u in ("m", "mn", "million"):
        return n * 1e6
    return n * 1e5 if n < 1000 else n  # bare number: lakhs if small, rupees otherwise


def parse_salary(value: Any) -> Optional[int]:
    """'12 LPA' / '₹12,00,000' / '1.2 Cr' / '10-12 LPA' / '50k per month' -> annual INR int."""
    try:
        if is_missing(value) or isinstance(value, bool):
            return None
        if isinstance(value, Mapping):
            km = keymap(value)
            lo, hi = parse_salary(km.get("min")), parse_salary(km.get("max"))
            if lo and hi:
                return int(round((lo + hi) / 2))
            return lo or hi or parse_salary(pick(km, ("amount", "value", "ctc", "salary")))
        if isinstance(value, (list, tuple)):
            vals = [v for v in (parse_salary(x) for x in value) if v]
            return int(round(sum(vals) / len(vals))) if vals else None
        if _is_real(value):
            v = float(value)
            amt = v * 1e5 if 0 < v < 1000 else v
            return int(round(amt)) if _SAL_MIN <= amt <= _SAL_MAX else None
        s = str(value).lower()
        s = re.sub(r"₹|\brs\.?|\binr\b|\brupees?\b|/-", " ", s)
        matches = list(_SAL_RE.finditer(s))
        if not matches:
            return None
        is_range = False
        if len(matches) >= 2:
            seg = s[matches[0].end():matches[1].start()].strip(" ,")
            is_range = seg in _RANGE_SEP
        use = matches[:2] if is_range else matches[:1]
        trailing = next((m.group("u") for m in reversed(use) if m.group("u")), None)
        amounts = []
        for m in use:
            unit = m.group("u") or (trailing if is_range else None)
            amounts.append(_salary_amount(_num(m.group("n")), unit))
        total = sum(amounts) / len(amounts)
        if _MONTHLY_RE.search(s):
            total *= 12
        return int(round(total)) if _SAL_MIN <= total <= _SAL_MAX else None
    except Exception:
        return None


_DUR_RE = re.compile(
    rf"(?P<n>(?:{_NUM}))(?:\s*(?P<u>days?|d|weeks?|wks?|w|months?|mos?|mths?|m|years?|yrs?|y)(?![a-z]))?"
)
_IMMEDIATE_RE = re.compile(
    r"\b(immediate(?:ly)?|asap|right\s*away|no\s*notice|nil|zero|available\s*now|ready\s*to\s*join|joined|served)\b"
)


def _unit_days(u: str) -> int:
    return {"d": 1, "w": 7, "m": 30, "y": 365}[u[0].lower()]


def _bare_notice(n: float, bare_unit: str) -> float:
    if bare_unit == "days":
        return n
    if bare_unit == "months":
        return n * 30
    return n * 30 if n <= 6 else n


def parse_notice_days(value: Any, bare_unit: str = "auto") -> Optional[int]:
    """'Immediate' -> 0, '2 months' -> 60, '1-2 months' -> 60 (max), '45 days' -> 45."""
    try:
        if is_missing(value) or isinstance(value, bool):
            return None
        if _is_real(value):
            d = _bare_notice(float(value), bare_unit)
            return int(round(d)) if 0 <= d <= 365 else (365 if d > 365 else None)
        s = str(value).lower()
        ms = list(_DUR_RE.finditer(s))
        if not ms:
            if _IMMEDIATE_RE.search(s):
                return 0
            if "serving" in s:
                return 30
            return None

        def days(m: "re.Match[str]", fallback_unit: Optional[str] = None) -> float:
            n, u = _num(m.group("n")), m.group("u") or fallback_unit
            return n * _unit_days(u) if u else _bare_notice(n, bare_unit)

        val = days(ms[0])
        if len(ms) >= 2 and s[ms[0].end():ms[1].start()].strip(" ,") in _RANGE_SEP:
            val = max(days(ms[0], ms[1].group("u")), days(ms[1]))
        if val < 0:
            return None
        return int(round(min(val, 365)))
    except Exception:
        return None


_FRAC_RE = re.compile(rf"(?P<a>(?:{_NUM}))\s*(?:/|out\s*of)\s*(?P<b>(?:{_NUM}))")
_PCT_RE = re.compile(rf"(?P<a>(?:{_NUM}))\s*(?:%|percent|pct)")
_ANYNUM_RE = re.compile(rf"(?:{_NUM})")


def _score_from_number(v: float) -> Optional[float]:
    if not math.isfinite(v) or v < 0:
        return None
    if v <= 1.0:
        return v * 100.0
    if v <= 10.0:
        return v * 10.0
    if v <= 100.0:
        return v
    return None


def parse_score(value: Any) -> Optional[float]:
    """'71/100', '7.1/10', '4.2/5', '71%', 0.71, 7.1, 71 -> float on a 0-100 scale."""
    try:
        if is_missing(value) or isinstance(value, bool):
            return None
        if _is_real(value):
            r = _score_from_number(float(value))
            return None if r is None else round(min(r, 100.0), 2)
        s = str(value).lower().strip()
        m = _FRAC_RE.search(s)
        if m:
            a, b = _num(m.group("a")), _num(m.group("b"))
            if b <= 0 or a < 0:
                return None
            return round(min(a / b * 100.0, 100.0), 2)
        m = _PCT_RE.search(s)
        if m:
            a = _num(m.group("a"))
            return round(min(a, 100.0), 2) if a >= 0 else None
        m = _ANYNUM_RE.search(s)
        if m:
            r = _score_from_number(_num(m.group(0)))
            return None if r is None else round(min(r, 100.0), 2)
        return None
    except Exception:
        return None


_SCORE_NAME_KEYS = ("name", "skill", "test", "subject", "assessment", "type", "title")
_SCORE_VAL_KEYS = ("score", "marks", "value", "result", "percentage", "percent")


def parse_scores(value: Any) -> dict[str, float]:
    """Dict / list / scalar of assessment results -> {name: 0-100 score}."""
    out: dict[str, float] = {}
    try:
        if is_missing(value):
            return out
        if isinstance(value, Mapping):
            km = keymap(value)
            direct = pick(km, _SCORE_VAL_KEYS)
            if direct is not None and not isinstance(direct, (Mapping, list, tuple)):
                s = parse_score(direct)
                if s is not None:
                    out[_key(pick(km, _SCORE_NAME_KEYS) or "overall")] = s
                return out
            for k, v in value.items():
                if isinstance(v, (Mapping, list, tuple)):
                    sub = parse_scores(v)
                    if sub:
                        out[_key(k)] = sum(sub.values()) / len(sub)
                    continue
                s = parse_score(v)
                if s is not None:
                    out[_key(k)] = s
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, Mapping):
                    km = keymap(item)
                    s = parse_score(pick(km, _SCORE_VAL_KEYS))
                    if s is not None:
                        out.setdefault(_key(pick(km, _SCORE_NAME_KEYS) or f"s{i}"), s)
                else:
                    s = parse_score(item)
                    if s is not None:
                        out[f"s{i}"] = s
        else:
            s = parse_score(value)
            if s is not None:
                out["overall"] = s
    except Exception:
        return out
    return out


_EXP_RE = re.compile(
    rf"(?P<n>(?:{_NUM}))\s*\+?\s*(?:(?P<u>years?|yrs?|y|months?|mos?|mths?|m)(?![a-z]))?"
)
_FRESHER_RE = re.compile(r"\b(fresher|fresh\s*graduate|no\s*experience|entry[\s-]*level|intern|trainee)\b")


def parse_experience_years(value: Any) -> Optional[float]:
    """'5 years', '5+ yrs', '3-5 years', '18 months', '5y 6m', 4.5, 'fresher' -> float years."""
    try:
        if is_missing(value) or isinstance(value, bool):
            return None
        if _is_real(value):
            return round(float(value), 2) if 0 <= float(value) <= 100 else None
        if not isinstance(value, str):
            return None
        s = value.lower()
        ms = list(_EXP_RE.finditer(s))
        if not ms:
            return 0.0 if _FRESHER_RE.search(s) else None

        def yrs(m: "re.Match[str]", fallback_unit: Optional[str] = None) -> float:
            n, u = _num(m.group("n")), (m.group("u") or fallback_unit or "y")
            return n / 12.0 if u.startswith("m") else n

        if len(ms) >= 2 and s[ms[0].end():ms[1].start()].strip(" ,") in _RANGE_SEP:
            total = (yrs(ms[0], ms[1].group("u")) + yrs(ms[1])) / 2.0
        else:
            total = yrs(ms[0])
            first_unit = (ms[0].group("u") or "y")
            for m in ms[1:]:
                if first_unit.startswith("y") and (m.group("u") or "").startswith("m"):
                    total += yrs(m)
                else:
                    break
        return round(total, 2) if 0 <= total <= 100 else None
    except Exception:
        return None


def parse_exp_range(value: Any) -> tuple[Optional[float], Optional[float]]:
    """Requisition experience: '3-5 years' -> (3,5); '5+' / 5 -> (5,None)."""
    try:
        if is_missing(value):
            return None, None
        if isinstance(value, str):
            ms = list(_EXP_RE.finditer(value.lower()))
            if len(ms) >= 2 and value.lower()[ms[0].end():ms[1].start()].strip(" ,") in _RANGE_SEP:
                lo = parse_experience_years(ms[0].group(0) + (ms[1].group("u") or ""))
                hi = parse_experience_years(ms[1].group(0))
                return lo, hi
        return parse_experience_years(value), None
    except Exception:
        return None, None


def _parse_year(value: Any, lo: int = 1950, hi: Optional[int] = None) -> Optional[int]:
    hi = hi if hi is not None else CURRENT_YEAR + 8
    if is_missing(value) or isinstance(value, bool):
        return None
    m = re.search(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", str(value))
    if not m:
        return None
    y = int(m.group(1))
    return y if lo <= y <= hi else None


def _parse_age(age_v: Any, dob_v: Any) -> Optional[int]:
    if not is_missing(age_v):
        m = re.search(r"\d+(?:\.\d+)?", str(age_v))
        if m:
            a = float(m.group(0))
            if 0 < a < 120:
                return int(a)
    y = _parse_year(dob_v, 1900, CURRENT_YEAR)
    if y is not None:
        return CURRENT_YEAR - y
    return None


def _id_str(v: Any) -> Optional[str]:
    if is_missing(v):
        return None
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    s = str(v).strip()
    return s or None


# --------------------------------------------------------------------------- #
# Profile / requisition normalisation
# --------------------------------------------------------------------------- #
_ID_KEYS = ("candidate_id", "id", "profile_id", "cand_id", "applicant_id", "uid")
_NAME_KEYS = ("name", "full_name", "candidate_name", "fullname")
_CITY_KEYS = ("city", "current_city", "location", "current_location", "base_location")
_SKILL_KEYS = ("skills", "skill_set", "skillset", "tech_stack", "technologies", "key_skills", "primary_skills", "top_skills", "skills_list")
_EXP_KEYS = ("experience_years", "years_of_experience", "total_experience", "total_experience_years", "experience", "exp", "exp_years", "yoe", "years_experience", "work_experience_years")
_CUR_CTC_KEYS = ("current_ctc", "ctc", "current_salary", "current_compensation", "present_ctc", "salary", "cctc")
_EXP_CTC_KEYS = ("expected_ctc", "expected_salary", "ectc", "expected_compensation", "salary_expectation", "expected_ctc_lpa")
_NOTICE_KEYS = ("notice_period_days", "notice_days", "notice_period", "notice", "notice_period_in_days", "serving_notice")
_SCORE_KEYS = ("assessment_scores", "assessments", "assessment_score", "assessment", "test_scores", "test_score", "scores", "score")
_SCORE_EXCLUDE = ("credit", "cibil", "match", "fit", "relevance", "similarity", "rating", "prescore", "rank")
_TITLE_KEYS = ("title", "current_title", "designation", "current_designation", "job_title", "role", "current_role", "headline")
_NOTE_KEYS = ("recruiter_notes", "recruiter_note", "notes", "note", "remarks", "comments", "summary", "feedback", "recruiter_feedback")
_EMAIL_KEYS = ("email", "email_id", "email_address", "mail")
_PHONE_KEYS = ("phone", "mobile", "phone_number", "contact", "contact_number", "mobile_number", "phone_no")
_RELOC_KEYS = ("willing_to_relocate", "relocate", "open_to_relocation", "relocation")
_EMPL_KEYS = ("employment_status", "currently_employed", "employed")
_AGE_KEYS = ("age", "candidate_age", "age_years")
_DOB_KEYS = ("dob", "date_of_birth", "birth_date", "birth_year", "year_of_birth")
_GRAD_KEYS = ("graduation_year", "grad_year", "passout_year", "passing_year", "year_of_passing", "year_of_graduation", "batch", "pass_out_year")


def normalize_profile(profile: Any) -> dict:
    """Raw profile/summary -> canonical dict. Missing values are None / '' / [] / {}."""
    km = keymap(profile)

    skills: list[str] = []
    seen: set[str] = set()
    for k in _SKILL_KEYS:
        for s in normalize_skills(km.get(_key(k))):
            if s not in seen:
                seen.add(s)
                skills.append(s)

    nk, nv = pick_with_key(km, _NOTICE_KEYS)
    bare_unit = "days" if nk and (nk.endswith("days") or "in_days" in nk) else "auto"

    scores: dict[str, float] = {}
    handled = set()
    for k in _SCORE_KEYS:
        handled.add(_key(k))
        for name, val in parse_scores(km.get(_key(k))).items():
            scores.setdefault(name, val)
    for k, v in km.items():
        if k in handled or not k.endswith(("_score", "_marks", "_assessment")):
            continue
        if any(x in k for x in _SCORE_EXCLUDE):
            continue
        s = parse_score(v)
        if s is not None:
            scores.setdefault(k, s)
    if "overall" in scores:
        overall: Optional[float] = scores["overall"]
    elif scores:
        overall = round(sum(scores.values()) / len(scores), 2)
    else:
        overall = None

    return {
        "id": _id_str(pick(km, _ID_KEYS)),
        "name": _as_text(pick(km, _NAME_KEYS), 120),
        "city": normalize_city(pick(km, _CITY_KEYS)),
        "skills": skills,
        "exp_years": parse_experience_years(pick(km, _EXP_KEYS)),
        "current_ctc": parse_salary(pick(km, _CUR_CTC_KEYS)),
        "expected_ctc": parse_salary(pick(km, _EXP_CTC_KEYS)),
        "notice_days": parse_notice_days(nv, bare_unit),
        "scores": scores,
        "score": overall,
        "age": _parse_age(pick(km, _AGE_KEYS), pick(km, _DOB_KEYS)),
        "grad_year": _parse_year(pick(km, _GRAD_KEYS)),
        "title": _as_text(pick(km, _TITLE_KEYS), 120).lower(),
        "email": _as_text(pick(km, _EMAIL_KEYS), 200).lower(),
        "phone": _as_text(pick(km, _PHONE_KEYS), 50),
        "relocate": parse_bool(pick(km, _RELOC_KEYS)),
        "employment_status": _as_text(pick(km, _EMPL_KEYS), 60).lower(),
        "notes": _as_text(pick(km, _NOTE_KEYS)),
    }


_R_MUST = ("must_have_skills", "required_skills", "must_have", "mandatory_skills", "skills_required", "skills", "primary_skills", "key_skills", "tech_stack")
_R_NICE = ("nice_to_have_skills", "nice_to_have", "preferred_skills", "good_to_have", "optional_skills", "secondary_skills", "bonus_skills")
_R_MIN_EXP = ("min_experience", "min_exp", "experience_min", "min_years", "minimum_experience", "min_experience_years")
_R_RANGE_EXP = ("experience_required", "experience_range", "experience", "exp", "exp_range", "years_of_experience")
_R_MAX_EXP = ("max_experience", "max_exp", "experience_max", "max_years", "maximum_experience")
_R_CITY = ("city", "cities", "location", "locations", "job_location", "work_location")
_R_BUDGET = ("budget", "max_budget", "budget_max", "ctc_budget", "max_ctc", "budget_ctc", "salary_budget", "max_salary", "ctc_max", "ctc_cap", "budget_per_head")
_R_BAR = ("bar", "min_score", "assessment_bar", "cutoff", "min_assessment_score", "score_bar", "pass_score", "min_assessment", "threshold", "assessment_cutoff")
_R_NOTICE = ("max_notice_days", "notice_max", "max_notice", "notice_period_max", "max_notice_period", "notice_days_max", "notice_period", "notice")
_R_TITLE = ("title", "role", "job_title", "position", "requisition_title", "name")
_REMOTE_TOKENS = frozenset({"remote", "anywhere", "any", "pan india", "work from home", "wfh", "all"})


def normalize_requisition(req: Any) -> dict:
    """Raw requisition -> canonical dict (idempotent)."""
    if isinstance(req, Mapping) and req.get("__normalized__") is True:
        return dict(req)
    km = keymap(req)

    must = normalize_skills(pick(km, _R_MUST))
    nice = [s for s in normalize_skills(pick(km, _R_NICE)) if s not in must]

    min_exp = parse_experience_years(pick(km, _R_MIN_EXP))
    max_exp = parse_experience_years(pick(km, _R_MAX_EXP))
    if min_exp is None or max_exp is None:
        lo, hi = parse_exp_range(pick(km, _R_RANGE_EXP))
        min_exp = lo if min_exp is None else min_exp
        max_exp = hi if max_exp is None else max_exp

    cities: list[str] = []
    remote_ok = bool(parse_bool(pick(km, ("remote", "remote_ok", "is_remote", "work_from_home"))))
    raw_city = pick(km, _R_CITY)
    parts = (raw_city if isinstance(raw_city, (list, tuple, set)) else re.split(r"[,;|/]+", str(raw_city))) if raw_city is not None else []
    for p in parts:
        if is_missing(p):
            continue
        low = str(p).strip().lower()
        if low in _REMOTE_TOKENS:
            remote_ok = True
            continue
        c = normalize_city(p)
        if c and c != "remote" and c not in cities:
            cities.append(c)
        elif c == "remote":
            remote_ok = True

    nk, nv = pick_with_key(km, _R_NOTICE)
    bare_unit = "days" if nk and nk.endswith("days") else "auto"

    return {
        "__normalized__": True,
        "title": _as_text(pick(km, _R_TITLE), 100),
        "must_skills": must,
        "nice_skills": nice,
        "min_exp": min_exp,
        "max_exp": max_exp,
        "cities": cities,
        "remote_ok": remote_ok,
        "city_strict": bool(parse_bool(pick(km, ("city_strict", "strict_location", "location_strict")))),
        "budget": parse_salary(pick(km, _R_BUDGET)),
        "bar": parse_score(pick(km, _R_BAR)),
        "max_notice": parse_notice_days(nv, bare_unit),
        "min_skill_coverage": (lambda x: None if x is None else min(max(float(x), 0.0), 1.0))(
            _safe_float(pick(km, ("min_skill_coverage", "min_skill_match")))
        ),
    }


def _safe_float(v: Any) -> Optional[float]:
    try:
        if is_missing(v) or isinstance(v, bool):
            return None
        f = float(str(v).replace("%", "").strip())
        if not math.isfinite(f):
            return None
        return f / 100.0 if f > 1.0 else f
    except Exception:
        return None


def normalize_frame(records: Any) -> pd.DataFrame:
    """List[dict] / DataFrame -> DataFrame of normalized profiles (index preserved)."""
    if isinstance(records, pd.DataFrame):
        idx = records.index
        rows = records.to_dict("records")
    else:
        rows = list(records or [])
        idx = pd.RangeIndex(len(rows))
    return pd.DataFrame([normalize_profile(r) for r in rows], index=idx)