from __future__ import annotations
"""
Module: attestation_registry.py — Minimal anti-replay registry (JSONL-backed).

Responsibilities:
- Deduplicate on (asset_id, p1_sha256).
- Fast O(1) membership checks via in-memory cache synchronized with file mtime.
- Append-only JSONL log for auditability.

Storage:
- Default path: <OUTPUTS_DIR>/attest_registry.jsonl
  where OUTPUTS_DIR = os.environ.get("OUTPUTS_DIR", "notebooks/outputs")

SECURITY:
- Records must not contain PII; only asset_id, p1_sha256, txid, network, issuer, ts.
- Do not log raw attestation payloads.

WARN:
- Thread-safe within a single process via threading.Lock. Multi-process safety is best-effort:
  append on POSIX is atomic, but cross-process cache coherence depends on mtime refresh.
"""

import json
import os
import threading
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


class AttestationRegistry:
    """Append-only JSONL registry with an in-memory dedup cache."""

    def __init__(self, path: Optional[str] = None):
        base = os.getenv("OUTPUTS_DIR", "notebooks/outputs")
        self.path = Path(path or (Path(base) / "attest_registry.jsonl"))
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        # Cache for O(1) membership: {(asset_id, p1_sha256)}
        self._cache: set[Tuple[str, str]] = set()
        self._mtime: float = 0.0

        self._load_cache_if_needed()

    # ------------------------------
    # Public API
    # ------------------------------
    def seen(self, p1_sha256: str, asset_id: str) -> bool:
        """Return True if (asset_id, p1_sha256) is already recorded."""
        key = (str(asset_id or "").strip(), str(p1_sha256 or "").strip())
        if not key[0] or not key[1]:
            return False
        self._load_cache_if_needed()
        return key in self._cache

    def record(
        self,
        p1_sha256: str,
        asset_id: str,
        txid: str,
        network: str,
        issuer: str,
        ts: int,
    ) -> None:
        """
        Append a new JSONL row and update the in-memory cache.

        NOTE:
        - This is append-only; no updates/deletes.
        - `issuer` should be a public address (no secrets).
        """
        rec = {
            "asset_id": str(asset_id or ""),
            "p1_sha256": str(p1_sha256 or ""),
            "txid": str(txid or ""),
            "network": str(network or ""),
            "issuer": str(issuer or ""),
            "ts": int(ts or 0),
        }
        if not rec["asset_id"] or not rec["p1_sha256"]:
            return

        line = json.dumps(rec, separators=(",", ":"), ensure_ascii=False)

        # PERF/CONSISTENCY: Per-process mutex; POSIX append is atomic.
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._cache.add((rec["asset_id"], rec["p1_sha256"]))
            try:
                self._mtime = self.path.stat().st_mtime
            except Exception:
                # Non-fatal: cache will be refreshed on next _load_cache_if_needed()
                pass

    # ------------------------------
    # Helpers
    # ------------------------------
    def _load_cache_if_needed(self) -> None:
        """Reload in-memory cache if the backing file's mtime changed."""
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            return
        except Exception:
            return

        if mtime == self._mtime and self._cache:
            return

        try:
            cache: set[Tuple[str, str]] = set()
            for rec in self._iter_jsonl():
                a = str(rec.get("asset_id", "")).strip()
                h = str(rec.get("p1_sha256", "")).strip()
                if a and h:
                    cache.add((a, h))
            with self._lock:
                self._cache = cache
                self._mtime = mtime
        except Exception:
            # Non-blocking: best-effort refresh; keep previous cache if parsing fails
            pass

    def _iter_jsonl(self) -> Iterable[Dict]:
        """Yield parsed JSON objects from the registry file, if present."""
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    yield json.loads(s)
                except Exception:
                    continue
