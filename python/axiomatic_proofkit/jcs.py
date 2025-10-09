# python/axiomatic_proofkit/jcs.py
import json, hashlib, math
from typing import Any, Dict

def _ensure_json_safe(value: Any) -> Any:
    # numeri finiti
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite float not allowed in canonical JSON (NaN/Inf).")
        return float(value)
    if isinstance(value, (int, bool, str)) or value is None:
        return value
    # sequenze
    if isinstance(value, (list, tuple)):
        return [_ensure_json_safe(v) for v in list(value)]
    # mapping con chiavi ordinate
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k in sorted(map(str, value.keys())):
            out[k] = _ensure_json_safe(value[k])
        return out
    # fallback: string
    return str(value)

def to_jcs_bytes(obj: Any) -> bytes:
    safe = _ensure_json_safe(obj)
    s = json.dumps(safe, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    return s.encode("utf-8")

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()
