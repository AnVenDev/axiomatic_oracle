import json, hashlib, math
from typing import Any, Dict, List, Tuple

def _ensure_json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite float not allowed in canonical JSON (NaN/Inf).")
        return float(value)

    if isinstance(value, (int, bool, str)) or value is None:
        return value

    if isinstance(value, (list, tuple)):
        return [_ensure_json_safe(v) for v in list(value)]

    if isinstance(value, dict):
        items: List[Tuple[str, Any]] = []
        for k0, v0 in value.items():
            k_str = str(k0)
            items.append((k_str, _ensure_json_safe(v0)))
        items.sort(key=lambda kv: kv[0])
        out: Dict[str, Any] = {}
        for k_str, v in items:
            out[k_str] = v
        return out

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

    return str(value)

def canonicalize_jcs(obj: Any) -> bytes:
    safe = _ensure_json_safe(obj)
    s = json.dumps(safe, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    return s.encode("utf-8")

def to_jcs_bytes(obj: Any) -> bytes:
    return canonicalize_jcs(obj)

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def jcs_sha256(obj: Any) -> str:
    return sha256_hex(canonicalize_jcs(obj))
