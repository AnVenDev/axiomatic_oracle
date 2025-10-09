from __future__ import annotations
"""
axiomatic_verifier.verify — verifica on-chain della nota p1.

- Indexer (algoexplorer) → nota base64 → JSON → bytes ACJ-1 → SHA-256
- confronto con eventuale hash presente
- time-window su ts (epoch seconds)
"""

from typing import Any, Dict, Optional
import base64, json, time, os

# JCS
from jcs import to_jcs_bytes, sha256_hex  # type: ignore


DEFAULT_SKEW_PAST_SEC = int(os.getenv("P1_TS_SKEW_PAST", "600"))   # 10 min
DEFAULT_SKEW_FUTURE_SEC = int(os.getenv("P1_TS_SKEW_FUTURE", "120"))  # 2 min

def _indexer_base(network: str) -> str:
    net = (network or "testnet").lower()
    return f"https://algoindexer.{'' if net=='mainnet' else 'testnet.'}algoexplorerapi.io"

def _explorer_url(txid: str, network: str) -> str:
    net = (network or "testnet").lower()
    return f"https://{'' if net=='mainnet' else 'testnet.'}algoexplorer.io/tx/{txid}"

def _fetch_tx(txid: str, network: str, indexer_url: Optional[str] = None) -> Dict[str, Any]:
    import requests  # lazy
    base = indexer_url or _indexer_base(network)
    r = requests.get(f"{base}/v2/transactions/{txid}", timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"tx_not_found: {r.status_code}")
    return r.json() or {}

def _decode_note(note_b64: Optional[str]) -> tuple[Optional[bytes], Optional[dict]]:
    if not note_b64:
        return None, None
    try:
        note_bytes = base64.b64decode(note_b64)
        obj = json.loads(note_bytes.decode("utf-8"))
        return note_bytes, obj if isinstance(obj, dict) else None
    except Exception:
        return None, None

def verify_tx(
    txid: str,
    *,
    network: str = "testnet",
    indexer_url: Optional[str] = None,
    max_skew_past_sec: int = DEFAULT_SKEW_PAST_SEC,
    max_skew_future_sec: int = DEFAULT_SKEW_FUTURE_SEC,
) -> Dict[str, Any]:
    """
    Verifica p1:
      - schema "light": richiede s='p1' + campi chiave
      - JCS SHA-256 sui bytes canonici della nota on-chain
      - time-window su ts se presente (epoch seconds)
    """
    try:
        data = _fetch_tx(txid, network, indexer_url)
    except Exception as e:
        return {"verified": False, "reason": f"tx_not_found:{e}", "mode": "unknown", "txid": txid}

    tx = data.get("transaction") or data
    note_b64 = tx.get("note")
    note_bytes, note_json = _decode_note(note_b64)
    explorer = _explorer_url(txid, network)
    if note_json is None:
        return {
            "txid": txid,
            "verified": False,
            "mode": "unknown",
            "reason": "note_not_json" if note_b64 else "note_missing",
            "explorer_url": explorer,
        }

    # Modalità
    mode = "p1" if note_json.get("s") == "p1" else ("legacy" if ("ref" in note_json or "schema_version" in note_json) else "unknown")
    if mode != "p1":
        return {
            "txid": txid,
            "verified": mode == "legacy",
            "mode": mode,
            "reason": None if mode == "legacy" else "unsupported_or_empty_note",
            "explorer_url": explorer,
        }

    # Hash parity dai bytes JCS (nota on-chain = fonte di verità)
    rebuilt_sha = sha256_hex(to_jcs_bytes(note_json))
    onchain_sha = note_json.get("note_sha256") or note_json.get("sha256")
    if onchain_sha and onchain_sha != rebuilt_sha:
        return {
            "txid": txid,
            "verified": False,
            "mode": "p1",
            "reason": "onchain_hash_mismatch",
            "note_sha256": onchain_sha,
            "rebuilt_sha256": rebuilt_sha,
            "explorer_url": explorer,
        }

    # Time-window (se ts presente, epoch seconds)
    ts = note_json.get("ts")
    now = int(time.time())
    if isinstance(ts, (int, float)):
        ts = int(ts)
        if ts < now - max_skew_past_sec or ts > now + max_skew_future_sec:
            return {
                "txid": txid,
                "verified": False,
                "mode": "p1",
                "reason": "ts_out_of_window",
                "note_sha256": onchain_sha or rebuilt_sha,
                "rebuilt_sha256": rebuilt_sha,
                "explorer_url": explorer,
            }

    return {
        "txid": txid,
        "verified": True,
        "mode": "p1",
        "reason": None,
        "note_sha256": onchain_sha or rebuilt_sha,
        "rebuilt_sha256": rebuilt_sha,
        "confirmed_round": tx.get("confirmed-round"),
        "explorer_url": explorer,
        "note": note_json,  # utile in debug; rimuovi se non lo vuoi esporre
    }
