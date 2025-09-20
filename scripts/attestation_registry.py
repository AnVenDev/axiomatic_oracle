from __future__ import annotations
"""
Registro minimale anti-replay basato su JSONL.
Chiave di dedup: (asset_id, p1_sha256).
- seen(p1_sha256, asset_id) -> bool
- record(p1_sha256, asset_id, txid, network, issuer, ts) -> None

Percorso di default:  <OUTPUTS_DIR>/attest_registry.jsonl
dove OUTPUTS_DIR = os.environ.get("OUTPUTS_DIR", "notebooks/outputs")
"""

import json
import os
import threading
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

class AttestationRegistry:
    def __init__(self, path: Optional[str] = None):
        base = os.getenv("OUTPUTS_DIR", "notebooks/outputs")
        self.path = Path(path or (Path(base) / "attest_registry.jsonl"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Cache in-memory per check O(1)
        self._cache: set[Tuple[str, str]] = set()  # {(asset_id, p1_sha256)}
        self._mtime: float = 0.0
        self._load_cache_if_needed()
    
    # ------------------------------
    # Public API
    # ------------------------------
    def seen(self, p1_sha256: str, asset_id: str) -> bool:
        """Ritorna True se una coppia (asset_id, p1_sha256) è già presente nel registro."""
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
        """Appende una nuova riga JSONL e aggiorna la cache in-memory."""
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
        with self._lock:
            # append atomico su POSIX; sufficiente per questo use-case
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._cache.add((rec["asset_id"], rec["p1_sha256"]))
            try:
                self._mtime = self.path.stat().st_mtime
            except Exception:
                pass
            
    # ------------------------------
    # Helpers
    # ------------------------------
    def _load_cache_if_needed(self) -> None:
        """Ricarica la cache se il file è cambiato su disco (mtime diverso)."""
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
            # in caso di errore non blocchiamo il flusso
            pass
        
    def _iter_jsonl(self) -> Iterable[Dict]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
                