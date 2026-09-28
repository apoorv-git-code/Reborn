# state.py
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from typing import Any

import config

log = logging.getLogger("state")


def _guard_dump(guard: Any) -> list[str]:
    try:
        if hasattr(guard, "to_dict"):
            d = guard.to_dict()
            return sorted(str(x) for x in (d.get("seen", []) if isinstance(d, dict) else d))
        seen = getattr(guard, "seen", None)
        if seen is not None:
            return sorted(str(x) for x in seen)
    except Exception:
        log.exception("guard dump failed")
    return []


def _guard_load(guard: Any, seen: list[str]) -> None:
    try:
        if hasattr(guard, "load_dict"):
            guard.load_dict({"seen": list(seen)})
            return
        cur = getattr(guard, "seen", None)
        if isinstance(cur, set):
            cur.update(seen)
        elif isinstance(cur, dict):
            for s in seen:
                cur[s] = True
        elif isinstance(cur, list):
            cur.extend(x for x in seen if x not in cur)
    except Exception:
        log.exception("guard load failed")


class State:
    def __init__(self, guard: Any, path: str | None = None):
        self.guard = guard
        self.path = path or config.STATE_PATH
        self._lock = threading.RLock()
        self._last_save = 0.0

        self.phase: str = "recon"
        self.credits: float | None = None
        self.requisitions: dict[str, dict] = {}
        self.shortlist: dict[str, list[str]] = {}      # req_id -> candidate ids awaiting profile
        self.ready: list[dict] = []                    # {cid, rid, score, person}
        self.fetched: set[str] = set()                 # candidate ids with profiles fetched
        self.search_pages: dict[str, int] = {}         # req_id -> next page
        self.search_done: set[str] = set()
        self.offered: set[str] = set()                 # "cid|rid"
        self.signed_persons: set[str] = set()
        self.filled_reqs: set[str] = set()
        self.recon_done: bool = False
        self.last_market: float = 0.0
        self.focus: list[str] = []

        self.load()

    # ---- persistence ----------------------------------------------------
    def to_dict(self) -> dict:
        with self._lock:
            return {
                "phase": self.phase,
                "credits": self.credits,
                "requisitions": self.requisitions,
                "shortlist": self.shortlist,
                "ready": self.ready,
                "fetched": sorted(self.fetched),
                "search_pages": self.search_pages,
                "search_done": sorted(self.search_done),
                "offered": sorted(self.offered),
                "signed_persons": sorted(self.signed_persons),
                "filled_reqs": sorted(self.filled_reqs),
                "recon_done": self.recon_done,
                "last_market": self.last_market,
                "focus": self.focus,
                "guard_seen": _guard_dump(self.guard),
            }

    def load(self) -> bool:
        if not os.path.exists(self.path):
            return False
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if not isinstance(d, dict):
                return False
        except (OSError, json.JSONDecodeError):
            log.exception("state load failed; starting fresh")
            return False
        with self._lock:
            self.phase = d.get("phase", self.phase)
            self.credits = d.get("credits")
            self.requisitions = d.get("requisitions", {}) or {}
            self.shortlist = {k: list(v) for k, v in (d.get("shortlist") or {}).items()}
            self.ready = list(d.get("ready") or [])
            self.fetched = set(d.get("fetched") or [])
            self.search_pages = {k: int(v) for k, v in (d.get("search_pages") or {}).items()}
            self.search_done = set(d.get("search_done") or [])
            self.offered = set(d.get("offered") or [])
            self.signed_persons = set(d.get("signed_persons") or [])
            self.filled_reqs = set(d.get("filled_reqs") or [])
            self.recon_done = bool(d.get("recon_done", False))
            self.last_market = float(d.get("last_market") or 0.0)
            self.focus = list(d.get("focus") or [])
            _guard_load(self.guard, d.get("guard_seen") or [])
        log.info("state restored: phase=%s ready=%d offered=%d", self.phase, len(self.ready), len(self.offered))
        return True

    def save(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_save) < config.STATE_SAVE_INTERVAL:
            return
        payload = self.to_dict()
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(prefix=".state-", suffix=".tmp", dir=directory)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.path)
            self._last_save = now
        except OSError:
            log.exception("state save failed")
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # ---- helpers --------------------------------------------------------
    def set_phase(self, phase: str) -> None:
        with self._lock:
            if phase != self.phase:
                log.info("phase %s -> %s", self.phase, phase)
                self.phase = phase
        self.save(force=True)

    def push_ready(self, item: dict) -> None:
        with self._lock:
            key = f'{item["cid"]}|{item["rid"]}'
            if key in self.offered:
                return
            if any(r["cid"] == item["cid"] and r["rid"] == item["rid"] for r in self.ready):
                return
            self.ready.append(item)
            self.ready.sort(key=lambda r: r.get("score", 0.0), reverse=True)

    def pop_ready(self, allow_reqs: set[str] | None = None) -> dict | None:
        with self._lock:
            for i, item in enumerate(self.ready):
                if item["rid"] in self.filled_reqs:
                    continue
                if item.get("person") in self.signed_persons:
                    continue
                if allow_reqs is not None and item["rid"] not in allow_reqs:
                    continue
                return self.ready.pop(i)
            return None