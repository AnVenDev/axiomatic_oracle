from __future__ import annotations
from typing import Any, Dict, Iterable, Tuple
import time

from .jcs import to_jcs_bytes, sha256_hex   # type: ignore

NOTE_MAX_BYTES = 1024
DEFAULT_ASSET_TAG = "re:EUR"

def _assert_finite(x: float, name: str) -> None:
    if not isinstance(x, (int, float)) or not (float("-inf") < float(x) < float("inf")):
        raise ValueError(f"{name} must be a finite number")

def build_p1(
    *,
    asset_tag: str = DEFAULT_ASSET_TAG,
    model_version: str,
    model_hash_hex: str = "",
    input_hash_hex: str,
    value_eur: float,
    uncertainty_low_eur: float,
    uncertainty_high_eur: float,
    timestamp_epoch: int | None = None,
) -> Dict[str, Any]:
    if uncertainty_low_eur > uncertainty_high_eur:
        raise ValueError("uncertainty_low_eur > uncertainty_high_eur")
    _assert_finite(value_eur, "value_eur")
    _assert_finite(uncertainty_low_eur, "uncertainty_low_eur")
    _assert_finite(uncertainty_high_eur, "uncertainty_high_eur")
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("model_version is required")
    if not (isinstance(input_hash_hex, str) and len(input_hash_hex) == 64):
        raise ValueError("input_hash_hex must be 64 hex chars")
    if model_hash_hex and len(model_hash_hex) != 64:
        raise ValueError("model_hash_hex must be 64 hex chars (or empty)")

    p1: Dict[str, Any] = {
        "s": "p1",
        "a": str(asset_tag),
        "mv": str(model_version),
        "mh": str(model_hash_hex or ""),
        "ih": str(input_hash_hex),
        "v": float(value_eur),
        "u": [float(uncertainty_low_eur), float(uncertainty_high_eur)],
        "ts": int(timestamp_epoch if timestamp_epoch is not None else time.time()),
    }

    _ = to_jcs_bytes(p1)
    return p1

def canonical_note_bytes_p1(p1: Dict[str, Any]) -> Tuple[bytes, str, int]:
    if not (isinstance(p1, dict) and p1.get("s") == "p1"):
        raise ValueError("Invalid p1 object")
    b = to_jcs_bytes(p1)
    return b, sha256_hex(b), len(b)

def assert_note_size_ok(p1: Dict[str, Any], max_bytes: int = NOTE_MAX_BYTES) -> None:
    _, _, size = canonical_note_bytes_p1(p1)
    if size > max_bytes:
        raise ValueError(f"p1 note too large: {size} bytes (max {max_bytes})")

def build_canonical_input(
    rec: Dict[str, Any],
    *,
    allowed_keys: Iterable[str],
    strip_none: bool = True,
) -> Dict[str, Any]:
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
    cin = build_canonical_input(rec, allowed_keys=allowed_keys)
    return sha256_hex(to_jcs_bytes(cin))
