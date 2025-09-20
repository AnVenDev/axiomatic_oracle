"""
Module: algorand_utils.py — Utilities for publishing and verifying PoVal™ on Algorand.

Responsibilities:
- Create clients (Algod / Indexer) from env config.
- Publish PoVal p1 using ACJ-1 canonical bytes (strict size guard).
- Legacy JSON note publish with compact fallback (ref/hash/url).
- Create 1:1 ASA (NFT-like) when needed.
- Resolve tx → decode note (Indexer first, Algod fallback) for Verify flow.

NOTE:
- p1 is compact ACJ-1 JSON; we do NOT compact/fallback it. If it exceeds NOTE_MAX_BYTES, fail.
- For historical transactions, always use the Indexer. Algod is only reliable for pending/immediate.
- Indexer ingestion may lag for a few seconds — short retry/backoff is applied.

SECURITY:
- Never log raw private keys or raw on-chain note payloads; expose only sizes/hashes.
"""

from __future__ import annotations

# =========================
# Standard library imports
# =========================
import base64
import hashlib
import json
import os
import time
from typing import Any, Dict, Optional, Tuple, TypedDict, Union

# ===========
# Third-party
# ===========
from algosdk import transaction  # type: ignore
from algosdk.v2client import algod, indexer  # type: ignore

# ========
# Local os
# ========
from scripts.canon import canonicalize_jcs  # ACJ-1 bytes for p1
from scripts.secrets_manager import (
    get_account,
    get_algod_config,
    get_indexer_config,
    get_network,
)

__all__ = [
    "AlgorandError",
    "create_algod_client",
    "wait_for_confirmation",
    "publish_p1_attestation",
    "publish_to_algorand",
    "create_token_for_asset",
    "get_tx_note_info",
    "explorer_url",
]

# =============================================================================
# Constants / Env
# =============================================================================
_DEFAULT_NOTE_MAX = int(os.getenv("NOTE_MAX_BYTES", "1024"))  # p1 must be < 1KB (typical)

# =============================================================================
# Error type
# =============================================================================
class AlgorandError(Exception):
    """Custom exception for Algorand operations."""
    pass


# =============================================================================
# Typed return shapes
# =============================================================================
class PublishResult(TypedDict, total=False):
    txid: str
    note_size: int
    note_sha256: str
    is_compacted: bool
    fallback_url_used: bool
    confirmed_round: Optional[int]


class TxNoteInfo(TypedDict, total=False):
    confirmed_round: Optional[int]
    note_size: Optional[int]
    note_sha256: Optional[str]
    note_json: Optional[dict]
    explorer_url: Optional[str]
    raw: Dict[str, Any]


# =============================================================================
# Client & account helpers
# =============================================================================
def create_algod_client() -> algod.AlgodClient:
    """
    Create an Algod client using env config (Algonode/Sandbox/Custom).
    NOTE: Token passed via headers for Algonode compatibility.
    """
    cfg = get_algod_config()
    token = ""  # token via headers
    headers = cfg.headers or {}
    return algod.AlgodClient(token, cfg.algod_url, headers)


def _create_indexer_client() -> Optional[indexer.IndexerClient]:
    """Instantiate Indexer client if configured; otherwise None."""
    try:
        cfg = get_indexer_config()
        token = ""  # Algonode uses header; compatible with cfg.headers
        headers = cfg.headers or {}
        return indexer.IndexerClient(token, cfg.indexer_url, headers)
    except Exception:
        return None


def _require_signing_account() -> Tuple[str, bytes]:
    """Return (address, private_key). Raise if secrets are missing."""
    acc = get_account(require_signing=True)
    if not acc.address or not acc.private_key:
        raise AlgorandError("Signing material not available (address/private_key). Check env secrets.")
    return acc.address, acc.private_key


# =============================================================================
# JSON helpers
# =============================================================================
def _json_min(obj: Dict[str, Any]) -> bytes:
    """Serialize deterministic UTF-8 minified JSON (compact separators)."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _build_compact_note(full_payload: Dict[str, Any], *, fallback_url: Optional[str]) -> Dict[str, Any]:
    """LEGACY: compact note for large payloads (aioracle:v2)."""
    raw = _json_min(full_payload)
    return {
        "ref": "aioracle:v2",
        "schema": full_payload.get("schema_version"),
        "asset_id": full_payload.get("asset_id"),
        "hash": hashlib.sha256(raw).hexdigest(),
        **({"url": fallback_url} if fallback_url else {}),
    }


def _ensure_note_bytes(
    payload: Dict[str, Any],
    *,
    fallback_url: Optional[str],
    max_bytes: int,
) -> Tuple[bytes, bool, bool]:
    """
    LEGACY: If payload exceeds max_bytes, replace with ref/hash (+url if present).
    Returns: (note_bytes, is_compacted, fallback_url_used)
    """
    raw = _json_min(payload)
    if len(raw) <= max_bytes:
        return raw, False, False

    compact = _build_compact_note(payload, fallback_url=fallback_url)
    cbytes = _json_min(compact)
    fallback_used = "url" in compact

    if len(cbytes) > max_bytes and "url" in compact:
        # Try removing url to fit the budget.
        del compact["url"]
        cbytes = _json_min(compact)
        fallback_used = False

    if len(cbytes) > max_bytes:
        # Minimal ref/hash only.
        minimal = {"ref": "aioracle:v2", "hash": compact["hash"]}
        cbytes = _json_min(minimal)

    return cbytes, True, fallback_used


def _coerce_pending_info(raw: Union[Dict[str, Any], bytes]) -> Dict[str, Any]:
    """Coerce algod pending_transaction_info outputs to dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise AlgorandError(f"Invalid JSON bytes from pending_transaction_info: {e}")
    raise AlgorandError("Unexpected type from pending_transaction_info")


def _decode_note_from_base64(note_b64: Optional[str]) -> Tuple[bytes, Optional[dict]]:
    """Decode base64 note to (bytes, json|None)."""
    note_bytes = b""
    if note_b64:
        try:
            note_bytes = base64.b64decode(note_b64)
        except Exception:
            note_bytes = b""
    try:
        note_json = json.loads(note_bytes.decode("utf-8")) if note_bytes else None
    except Exception:
        note_json = None
    return note_bytes, note_json


# =============================================================================
# Explorer helper
# =============================================================================
def explorer_url(txid: str, network: Optional[str] = None) -> Optional[str]:
    """
    Build explorer URL consistent with network.
    Returns None for sandbox/custom networks.
    """
    if not txid:
        return None
    net = (network or get_network() or "").lower()
    if net == "mainnet":
        return f"https://explorer.perawallet.app/tx/{txid}"
    if net == "testnet":
        return f"https://testnet.explorer.perawallet.app/tx/{txid}"
    return None


# =============================================================================
# Core ops
# =============================================================================
def wait_for_confirmation(client: algod.AlgodClient, txid: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Wait for transaction confirmation (or timeout in N rounds).
    Returns the final pending info dict upon confirmation.

    WARN:
    - Use this only for the immediate/pending path. For historical lookups, use Indexer.
    """
    try:
        status = client.status()
        last_round = int(status.get("last-round") if isinstance(status, dict) else 0)
    except Exception as e:
        raise AlgorandError(f"Algod status error: {e}")

    for _ in range(max(1, timeout)):
        info_raw = client.pending_transaction_info(txid)
        info = _coerce_pending_info(info_raw)
        if int(info.get("confirmed-round", 0)) > 0:
            return info
        client.status_after_block(last_round + 1)
        last_round += 1

    raise AlgorandError(f"Transaction {txid} not confirmed after {timeout} rounds")


def publish_p1_attestation(
    p1_obj: Dict[str, Any],
    *,
    use_flat_fee: bool = True,
    max_note_bytes: int = _DEFAULT_NOTE_MAX,
) -> PublishResult:
    """
    Publish PoVal p1 in the note field using ACJ-1 canonical bytes.
    - No fallback/compaction: if > max_note_bytes, raise AlgorandError.
    - Returns txid, note_size, note_sha256, is_compacted(False), confirmed_round.

    SECURITY:
    - Do not log raw p1 or keys. Log only txid, note_size, note_sha256.
    """
    if not isinstance(p1_obj, dict) or p1_obj.get("s") != "p1":
        raise AlgorandError("Invalid p1 object (missing s='p1').")

    # ACJ-1 canonical bytes (deterministic)
    note_bytes = canonicalize_jcs(p1_obj)
    if len(note_bytes) > max_note_bytes:
        raise AlgorandError(f"p1 note exceeds byte budget ({len(note_bytes)} > {max_note_bytes})")

    client = create_algod_client()
    sender, sk = _require_signing_account()

    try:
        sp = client.suggested_params()
        if use_flat_fee:
            sp.flat_fee = True
            sp.fee = 1000  # min-fee

        txn = transaction.PaymentTxn(sender=sender, receiver=sender, amt=0, sp=sp, note=note_bytes)
        stx = txn.sign(sk)
        txid = client.send_transaction(stx)
        info = wait_for_confirmation(client, txid, timeout=10)

        return PublishResult(
            txid=txid,
            note_size=len(note_bytes),
            note_sha256=hashlib.sha256(note_bytes).hexdigest(),
            is_compacted=False,
            fallback_url_used=False,
            confirmed_round=int(info.get("confirmed-round")) if info.get("confirmed-round") else None,
        )
    except Exception as e:
        raise AlgorandError(f"Algorand publish failed: {e}")


def publish_to_algorand(
    note_dict: Dict[str, Any],
    *,
    fallback_url: Optional[str] = None,
    use_flat_fee: bool = True,
    max_note_bytes: int = 900,
) -> PublishResult:
    """
    LEGACY: self-transfer 0 ALGO with JSON note 'aioracle:v2' (or ref/hash if too large).
    Returns: txid, note_size, note_sha256, is_compacted, fallback_url_used, confirmed_round.
    """
    client = create_algod_client()
    sender, sk = _require_signing_account()

    try:
        sp = client.suggested_params()
        if use_flat_fee:
            sp.flat_fee = True
            sp.fee = 1000  # min-fee

        note_bytes, is_compacted, fallback_used = _ensure_note_bytes(
            note_dict, fallback_url=fallback_url, max_bytes=max_note_bytes
        )

        txn = transaction.PaymentTxn(sender=sender, receiver=sender, amt=0, sp=sp, note=note_bytes)
        stx = txn.sign(sk)
        txid = client.send_transaction(stx)
        info = wait_for_confirmation(client, txid, timeout=10)

        return PublishResult(
            txid=txid,
            note_size=len(note_bytes),
            note_sha256=hashlib.sha256(note_bytes).hexdigest(),
            is_compacted=bool(is_compacted),
            fallback_url_used=bool(fallback_used),
            confirmed_round=int(info.get("confirmed-round")) if info.get("confirmed-round") else None,
        )
    except Exception as e:
        raise AlgorandError(f"Algorand publish failed: {e}")


def create_token_for_asset(
    asset_name: str,
    unit_name: str,
    *,
    metadata_content: Optional[str] = None,
    url: Optional[str] = None,
    default_url: str = "https://example.com/aioracle/metadata.json",
    use_flat_fee: bool = True,
) -> int:
    """
    Create a 1-unit ASA (0 decimals) with manager/reserve/freeze/clawback = sender.
    - metadata_content (optional): used for metadata_hash (SHA-256 of bytes).
    - url: URL written into ASA; falls back to default_url.
    """
    client = create_algod_client()
    sender, sk = _require_signing_account()
    try:
        sp = client.suggested_params()
        if use_flat_fee:
            sp.flat_fee = True
            sp.fee = 1000  # min-fee

        mhash = hashlib.sha256(metadata_content.encode("utf-8")).digest() if metadata_content else None

        txn = transaction.AssetConfigTxn(
            sender=sender,
            sp=sp,
            total=1,
            decimals=0,
            default_frozen=False,
            unit_name=(unit_name or "")[:8],
            asset_name=(asset_name or "")[:32],
            manager=sender,
            reserve=sender,
            freeze=sender,
            clawback=sender,
            url=url or default_url,
            metadata_hash=mhash,
        )

        stx = txn.sign(sk)
        txid = client.send_transaction(stx)
        wait_for_confirmation(client, txid, timeout=10)
        ptx = _coerce_pending_info(client.pending_transaction_info(txid))
        asset_id = ptx.get("asset-index")
        if not isinstance(asset_id, int):
            raise AlgorandError("ASA creation succeeded but asset-index missing in pending info")
        return asset_id
    except Exception as e:
        raise AlgorandError(f"ASA creation failed: {e}")


# =============================================================================
# Tx lookup (Indexer-first with Algod fallback)
# =============================================================================
def get_tx_note_info(txid: str) -> TxNoteInfo:
    """
    Resolve txid → (confirmed_round, note_size, note_sha256, note_json, explorer_url, raw).

    NOTE:
    - Prefer Indexer for historical transactions (with short retry/backoff).
    - Fallback to Algod pending path for just-published transactions.
    - Do not log raw note; compute hash/size and return JSON only to callers explicitly using it.
    """
    # --- 1) Try Indexer (historical) -----------------------------------------
    idx = _create_indexer_client()
    if idx is not None:
        backoff = 0.4
        for _ in range(6):  # ~2.5s total before falling back
            try:
                resp = idx.transaction(txid)  # single tx by id
                # Formats may vary: {"transaction": {...}} or direct fields
                txn = resp.get("transaction", resp)
                confirmed_round = txn.get("confirmed-round") or txn.get("confirmedRound")
                note_b64 = txn.get("note")
                note_bytes, note_json = _decode_note_from_base64(note_b64)
                return TxNoteInfo(
                    confirmed_round=int(confirmed_round) if confirmed_round else None,
                    note_size=(len(note_bytes) if note_bytes else None),
                    note_sha256=(hashlib.sha256(note_bytes).hexdigest() if note_bytes else None),
                    note_json=note_json,
                    explorer_url=explorer_url(txid, get_network()),
                    raw=txn if isinstance(txn, dict) else {},
                )
            except Exception:
                time.sleep(backoff)
                backoff *= 1.5  # PERF: incremental backoff
        # proceed to Algod fallback

    # --- 2) Fallback: Algod pending/immediate -------------------------------
    client = create_algod_client()
    try:
        info = wait_for_confirmation(client, txid, timeout=1)
    except Exception as e:
        # Return minimal structure containing the error (do not crash caller)
        return TxNoteInfo(
            confirmed_round=None,
            note_size=None,
            note_sha256=None,
            note_json=None,
            explorer_url=explorer_url(txid, get_network()),
            raw={"error": str(e)},
        )

    note_b64 = None
    if isinstance(info, dict):
        note_b64 = info.get("note") or (info.get("txn") or {}).get("txn", {}).get("note")

    note_bytes, note_json = _decode_note_from_base64(note_b64)
    return TxNoteInfo(
        confirmed_round=int(info.get("confirmed-round")) if isinstance(info, dict) and info.get("confirmed-round") else None,
        note_size=(len(note_bytes) if note_bytes else None),
        note_sha256=(hashlib.sha256(note_bytes).hexdigest() if note_bytes else None),
        note_json=note_json,
        explorer_url=explorer_url(txid, get_network()),
        raw=info if isinstance(info, dict) else {},
    )
