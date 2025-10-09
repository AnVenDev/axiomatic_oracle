import json, hashlib

def to_jcs_bytes(obj: dict) -> bytes:
    s = json.dumps(obj, separators=(',', ':'), sort_keys=True, ensure_ascii=False)
    return s.encode('utf-8')

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()