# scripts/canon.py
from __future__ import annotations

import json
import math
import hashlib
from typing import Any, Dict, Iterable, List, Optional, Tuple

"""
Canon utilities for PoVal (ACJ-1 canonical JSON bytes)
- canonicalize_jcs(obj) -> bytes                 # JSON canonicizzato (chiavi ordinate, separatori minimi, UTF-8)
- sha256_hex(b) -> str                           # SHA-256 esadecimale
- size_bytes_acj1(obj) -> int                    # dimensione in byte della forma canonica
- build_canonical_input(rec, allowed_keys)       # sottoinsieme canonico delle sole raw features attese
- compute_input_hash(rec, *, allowed_keys) -> str# SHA-256 (hex) dei canonical bytes dell'input canonico
- build_p1(...) / build_p1_from_response(...)    # costruttori della PoVal p1
- canonical_note_bytes_p1(p1) -> (bytes, sha, n) # bytes+sha256+size da pubblicare on-chain
"""

# -----------------------------------------------------------------------------
# Canonical JSON (ACJ-1)
# -----------------------------------------------------------------------------
def _ensure_json_safe(value: Any) -> Any:
    # numeri finiti, niente NaN/Inf
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite float not allowed in canonical JSON (NaN/Inf).")
        return float(value)
    # bool, int, str, None pass-through
    if isinstance(value, (int, bool, str)) or value is None:
        return value
    # liste e dict ricorsivi
    if isinstance(value, list):
        return [_ensure_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_ensure_json_safe(v) for v in value]
    if isinstance(value, dict):
        # chiavi tutte string e ordinate lessicograficamente
        out: Dict[str, Any] = {}
        for k in sorted(map(str, value.keys())):
            out[k] = _ensure_json_safe(value[k])
        return out
    # tipi numpy o simili
    try:
        import numpy as np  # type: ignore
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            fv = float(value)
            if not math.isfinite(fv):
                raise ValueError("Non-finite float not allowed in canonical JSON (NaN/Inf).")
            return fv
        if isinstance(value, (np.ndarray,)):
            return [_ensure_json_safe(v) for v in value.tolist()]
    except Exception:
        pass
    # fallback: stringa
    return str(value)

def canonicalize_jcs(obj: Any) -> bytes:
    """
    JSON canonico:
      - chiavi ordinate (sort_keys=True)
      - separatori minimi (',' e ':')
      - nessuno spazio
      - ensure_ascii=False (UTF-8)
    """
    safe_obj = _ensure_json_safe(obj)
    return json.dumps(safe_obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def size_bytes_acj1(obj: Any) -> int:
    return len(canonicalize_jcs(obj))

# -----------------------------------------------------------------------------
# Canonical input hashing
# -----------------------------------------------------------------------------
def build_canonical_input(rec: Dict[str, Any], *, allowed_keys: Iterable[str], strip_none: bool = True) -> Dict[str, Any]:
    """
    Restituisce il sotto-insieme (key->value) dell'input limitato alle sole **raw features attese**
    (ordine delle chiavi non rilevante perché la canonicalizzazione le ordina).
    """
    allowed = set(map(str, allowed_keys))
    out: Dict[str, Any] = {}
    for k, v in rec.items():
        k = str(k)
        if k not in allowed:
            continue
        if strip_none and (v is None):
            continue
        out[k] = v
    # niente derived qui; solo features raw
    return out

def compute_input_hash(rec: Dict[str, Any], *, allowed_keys: Iterable[str]) -> str:
    """
    Hash dell'input canonico (SHA-256 esadecimale).
    """
    cin = build_canonical_input(rec, allowed_keys=allowed_keys)
    return sha256_hex(canonicalize_jcs(cin))

# -----------------------------------------------------------------------------
# PoVal p1 builders
# -----------------------------------------------------------------------------
def build_p1(
    *,
    asset_tag: str,            # es. "re:EUR"
    model_version: str,        # es. "v2"
    model_hash_hex: str,       # 64 hex, opzionale/può essere ""
    input_hash_hex: str,       # 64 hex
    value_eur: float,          # valore in EUR
    uncertainty_low_eur: float,
    uncertainty_high_eur: float,
    timestamp_epoch: int       # secondi
) -> Dict[str, Any]:
    if uncertainty_low_eur > uncertainty_high_eur:
        raise ValueError("uncertainty_low_eur > uncertainty_high_eur")
    p1 = {
        "s": "p1",
        "a": str(asset_tag),
        "mv": str(model_version),
        "mh": str(model_hash_hex or ""),
        "ih": str(input_hash_hex),
        "v": float(value_eur),
        "u": [float(uncertainty_low_eur), float(uncertainty_high_eur)],
        "ts": int(timestamp_epoch),
    }
    # Validazione minima (nessun NaN/Inf, limiti u coerenti)
    _ = canonicalize_jcs(p1)  # solleva se non serializzabile/finito (NaN/Inf)
    return p1

def build_p1_from_response(response: Dict[str, Any], *, allowed_input_keys: Iterable[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Costruisce p1 a partire dalla response schema v2.
    Ritorna (p1, debug_meta) dove debug_meta include cin (input canonico) e hash calcolati.
    """
    metrics = response.get("metrics") or {}
    meta = response.get("model_meta") or {}
    v_k = float(metrics["valuation_k"])
    lo_k = float(metrics["confidence_low_k"])
    hi_k = float(metrics["confidence_high_k"])
    v_eur = v_k * 1000.0
    lo_eur = lo_k * 1000.0
    hi_eur = hi_k * 1000.0

    mv = str(meta.get("value_model_version") or meta.get("model_version") or "v?").strip()[:16]
    mh = (meta.get("model_hash") or "").strip().lower()
    if mh and len(mh) != 64:
        raise ValueError(f"model_hash must be 64 hex chars, got {len(mh)}")

    cin = build_canonical_input(response.get("validation", {}), allowed_keys=allowed_input_keys)  # Fallback se non si passa l’input originale
    # Idealmente passa il record raw 'rec' da predict(); qui gestiamo fallback:
    if not cin:
        # prova a leggere dal campo 'canonical_input' o 'canonical_input_subset' se presente in response
        cin = dict(response.get("canonical_input") or response.get("canonical_input_subset") or {})

    ih = sha256_hex(canonicalize_jcs(cin))

    ts = response.get("timestamp")
    try:
        # timestamp ISO -> epoch se serve
        import datetime, dateutil.parser  # type: ignore
        if isinstance(ts, str):
            dt = dateutil.parser.isoparse(ts)
            timestamp_epoch = int(dt.timestamp())
        else:
            raise Exception()
    except Exception:
        import time
        timestamp_epoch = int(time.time())

    p1 = build_p1(
        asset_tag="re:EUR",
        model_version=mv,
        model_hash_hex=mh,
        input_hash_hex=ih,
        value_eur=v_eur,
        uncertainty_low_eur=lo_eur,
        uncertainty_high_eur=hi_eur,
        timestamp_epoch=timestamp_epoch,
    )
    debug_meta = {"canonical_input": cin, "ih": ih}
    return p1, debug_meta

def canonical_note_bytes_p1(p1: Dict[str, Any]) -> bytes:
    """
    Restituisce la terna (bytes, sha256_hex, size) da pubblicare come nota su Algorand.
    """
    if not (isinstance(p1, dict) and p1.get("s") == "p1"):
        raise ValueError("Invalid p1 object")
    b = canonicalize_jcs(p1)
    return b, sha256_hex(b), len(b)