# scripts/canon.py
from __future__ import annotations
"""
Module: canon.py — Canon utilities for PoVal (ACJ-1 canonical JSON bytes)

Responsibilities:
- Canonical JSON (ACJ-1):
    - canonicalize_jcs(obj) -> bytes
    - sha256_hex(bytes) -> str
    - size_bytes_acj1(obj) -> int
- Canonical input hashing:
    - build_canonical_input(rec, allowed_keys) -> dict
    - compute_input_hash(rec, *, allowed_keys) -> str
- PoVal p1 builders:
    - build_p1(...)
    - build_p1_from_response(response, *, allowed_input_keys) -> (p1, debug_meta)
    - canonical_note_bytes_p1(p1) -> (bytes, sha256_hex, size_bytes)

NOTE:
- ACJ-1 here means: UTF-8 JSON with sorted keys, minimal separators, no spaces, and JSON-safe values
  (finite numbers only; no NaN/Inf; keys coerced to strings).
- We keep this module free of network/FS concerns to allow deterministic behavior and easy testing.

SECURITY:
- Do not include PII in canonical input or p1 payloads.
"""

# =========================
# Standard library imports
# =========================
import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Tuple

__all__ = [
    "canonicalize_jcs",
    "sha256_hex",
    "size_bytes_acj1",
    "build_canonical_input",
    "compute_input_hash",
    "build_p1",
    "build_p1_from_response",
    "canonical_note_bytes_p1",
]

# =============================================================================
# ACJ-1: Canonical JSON helpers
# =============================================================================
def _ensure_json_safe(value: Any) -> Any:
    """
    Coerce a Python value into a JSON-safe, deterministic representation:
    - Floats must be finite (no NaN/Inf).
    - Numbers stay numbers; bool/None/str pass through.
    - Tuples -> lists; numpy types handled if available.
    - Dict keys coerced to strings and sorted lexicographically.
    """
    # Finite floats only
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite float not allowed in canonical JSON (NaN/Inf).")
        return float(value)

    # Pass-through simple types
    if isinstance(value, (int, bool, str)) or value is None:
        return value

    # Sequences
    if isinstance(value, (list, tuple)):
        return [_ensure_json_safe(v) for v in list(value)]

    # Mappings
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k in sorted(map(str, value.keys())):
            out[k] = _ensure_json_safe(value[k])
        return out

    # Numpy family (optional)
    try:
        import numpy as np  # type: ignore
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            fv = float(value)
            if not math.isfinite(fv):
                raise ValueError("Non-finite float not allowed in canonical JSON (NaN/Inf).")
            return fv
        if isinstance(value, np.ndarray):
            return [_ensure_json_safe(v) for v in value.tolist()]
    except Exception:
        pass

    # Fallback: stringify
    return str(value)


def canonicalize_jcs(obj: Any) -> bytes:
    """
    Produce ACJ-1 bytes:
      - UTF-8 JSON
      - sorted keys
      - minimal separators (',' and ':')
      - no spaces
      - ensure_ascii=False (allow UTF-8)
    """
    safe_obj = _ensure_json_safe(obj)
    return json.dumps(
        safe_obj,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Return hex-encoded SHA-256 of bytes."""
    return hashlib.sha256(data).hexdigest()


def size_bytes_acj1(obj: Any) -> int:
    """Return length in bytes of ACJ-1 canonical form for obj."""
    return len(canonicalize_jcs(obj))


# =============================================================================
# Canonical input hashing
# =============================================================================
def build_canonical_input(
    rec: Dict[str, Any],
    *,
    allowed_keys: Iterable[str],
    strip_none: bool = True,
) -> Dict[str, Any]:
    """
    Return the subset (key->value) of `rec` limited to raw expected features (allowed_keys).
    Derived fields must NOT be added here.

    NOTE:
    - Key order is irrelevant (ACJ-1 sorts keys).
    - If `strip_none=True`, fields with None are omitted.
    """
    allowed = set(map(str, allowed_keys))
    out: Dict[str, Any] = {}
    for k, v in rec.items():
        k = str(k)
        if k not in allowed:
            continue
        if strip_none and v is None:
            continue
        out[k] = v
    return out


def compute_input_hash(rec: Dict[str, Any], *, allowed_keys: Iterable[str]) -> str:
    """
    Compute SHA-256 (hex) over the ACJ-1 bytes of the canonical input subset.
    """
    cin = build_canonical_input(rec, allowed_keys=allowed_keys)
    return sha256_hex(canonicalize_jcs(cin))


# =============================================================================
# PoVal p1 builders
# =============================================================================
def build_p1(
    *,
    asset_tag: str,             # e.g., "re:EUR"
    model_version: str,         # e.g., "v2"
    model_hash_hex: str,        # 64 hex, optional/empty string allowed
    input_hash_hex: str,        # 64 hex
    value_eur: float,           # point estimate in EUR
    uncertainty_low_eur: float,
    uncertainty_high_eur: float,
    timestamp_epoch: int,       # seconds (UTC)
) -> Dict[str, Any]:
    """
    Build PoVal p1 object.

    VALIDATION:
    - uncertainty_low_eur <= uncertainty_high_eur
    - numbers must be finite (validated by ACJ-1 serialization)
    """
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

    # Minimal validation via canonicalization (raises on NaN/Inf / non-serializable)
    _ = canonicalize_jcs(p1)
    return p1


def _iso_to_epoch_seconds(ts: Any) -> int:
    """
    Convert ISO 8601 (or epoch-like) into integer Unix seconds (UTC).
    Accepts:
    - ISO strings with optional trailing 'Z'
    - int/float epoch seconds (will be truncated to seconds)

    Fallback: current UTC time if parsing fails.
    """
    # numeric epoch
    if isinstance(ts, (int, float)):
        return int(float(ts))

    # string: try ISO
    if isinstance(ts, str):
        s = ts.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except Exception:
            pass

    # fallback: now
    return int(datetime.now(tz=timezone.utc).timestamp())


def build_p1_from_response(
    response: Dict[str, Any],
    *,
    allowed_input_keys: Iterable[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build p1 from a schema v2 prediction response.
    Returns (p1, debug_meta) where debug_meta includes:
      - canonical_input: the exact subset used for input hash
      - ih: SHA-256 (hex) over ACJ-1 bytes of canonical_input

    PRIORITY for canonical input:
      1) response["canonical_input"] if present (preferred; set by inference_api)
      2) build from original input if provided by caller (not available here)
      3) (best-effort) fallback to response["validation"] filtered by allowed_input_keys
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
    if mh and (len(mh) != 64 or any(c not in "0123456789abcdef" for c in mh)):
        raise ValueError(f"model_hash must be 64 hex chars, got: {mh!r}")

    # Preferred: exact canonical_input attached by inference_api
    cin = dict(response.get("canonical_input") or {})
    if not cin:
        # Best-effort fallback: filter validation block (may contain raw-like fields)
        cin = build_canonical_input(response.get("validation", {}), allowed_keys=allowed_input_keys)

    ih = sha256_hex(canonicalize_jcs(cin))

    ts = response.get("timestamp")
    timestamp_epoch = _iso_to_epoch_seconds(ts)

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


def canonical_note_bytes_p1(p1: Dict[str, Any]) -> Tuple[bytes, str, int]:
    """
    Return (bytes, sha256_hex, size_bytes) for the p1 note to publish on-chain.
    """
    if not (isinstance(p1, dict) and p1.get("s") == "p1"):
        raise ValueError("Invalid p1 object")
    b = canonicalize_jcs(p1)
    return b, sha256_hex(b), len(b)
