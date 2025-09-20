"""
Algorand utilities

Funzioni principali:
- create_algod_client(): istanzia il client con config da env (/secrets_manager)
- wait_for_confirmation(txid, timeout): attende la conferma con gestione errori robusta
- publish_p1_attestation(p1_obj): pubblica **PoVal p1** usando i **byte ACJ-1** (fail se > NOTE_MAX_BYTES)
- publish_to_algorand(note_dict, *, fallback_url=None): LEGACY self-transfer 0 ALGO con nota JSON (<= NOTE_MAX_BYTES)
- create_token_for_asset(asset_name, unit_name, metadata_content=None, url=None): crea ASA 1:1 (NFT-like)
- get_tx_note_info(txid): recupera info/nota (decodificata) di una transazione confermata
"""

from __future__ import annotations

import os
import json
import base64
import hashlib
from typing import Any, Dict, Optional, Union, Tuple, TypedDict

from algosdk.v2client import algod  # type: ignore
from algosdk import transaction     # type: ignore

from scripts.secrets_manager import get_algod_config, get_account, get_network
from scripts.canon import canonicalize_jcs  # ACJ-1 per p1

__all__ = [
    "AlgorandError",
    "create_algod_client",
    "wait_for_confirmation",
    "publish_p1_attestation",
    "publish_to_algorand",
    "create_token_for_asset",
    "get_tx_note_info",
    "explorer_url"
]

# ---------------------------------------------------------------------
# Costanti / Env
# ---------------------------------------------------------------------
_DEFAULT_NOTE_MAX = int(os.getenv("NOTE_MAX_BYTES", "1024"))  # per p1 garantiamo <1KB

# ---------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------
class AlgorandError(Exception):
    """Custom exception for Algorand operations."""
    pass

# ---------------------------------------------------------------------
# Tipi di ritorno
# ---------------------------------------------------------------------
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
    raw: Dict[str, Any]

# ---------------------------------------------------------------------
# Client & account helpers
# ---------------------------------------------------------------------
def create_algod_client() -> algod.AlgodClient:
    """
    Crea un client Algod usando la configurazione da env (Algonode/Sandbox/Custom).
    Accetta token via headers per compatibilità con Algonode.
    """
    cfg = get_algod_config()
    token = ""  # token passato via headers
    headers = cfg.headers or {}
    return algod.AlgodClient(token, cfg.algod_url, headers)

def _require_signing_account() -> Tuple[str, bytes]:
    """
    Ritorna (address, private_key). Solleva se i segreti non ci sono.
    """
    acc = get_account(require_signing=True)
    if not acc.address or not acc.private_key:
        raise AlgorandError("Signing material not available (address/private_key). Check env secrets.")
    return acc.address, acc.private_key

# ---------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------
def _json_min(obj: Dict[str, Any]) -> bytes:
    """Serializza JSON minificato UTF-8 (deterministico)."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def _build_compact_note(full_payload: Dict[str, Any], *, fallback_url: Optional[str]) -> Dict[str, Any]:
    """
    LEGACY: nota compatta per payload troppo grandi (aioracle:v2).
    """
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
    max_bytes: int
) -> Tuple[bytes, bool, bool]:
    """
    LEGACY: se eccede, sostituisce con ref/hash (+url se presente).
    """
    raw = _json_min(payload)
    if len(raw) <= max_bytes:
        return raw, False, False

    compact = _build_compact_note(payload, fallback_url=fallback_url)
    cbytes = _json_min(compact)
    fallback_used = "url" in compact

    if len(cbytes) > max_bytes and "url" in compact:
        del compact["url"]
        cbytes = _json_min(compact)
        fallback_used = False

    if len(cbytes) > max_bytes:
        minimal = {"ref": "aioracle:v2", "hash": compact["hash"]}
        cbytes = _json_min(minimal)

    return cbytes, True, fallback_used

def _coerce_pending_info(raw: Union[Dict[str, Any], bytes]) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise AlgorandError(f"Invalid JSON bytes from pending_transaction_info: {e}")
    raise AlgorandError("Unexpected type from pending_transaction_info")

# ---------------------------------------------------------------------
# Explorer helper
# ---------------------------------------------------------------------
def explorer_url(txid: str, network: Optional[str] = None) -> Optional[str]:
    """
    Costruisce l'URL dell'explorer coerente con il network.
    Ritorna None per network sandbox/custom senza explorer pubblico.
    """
    if not txid:
        return None
    net = (network or get_network() or "").lower()
    if net == "mainnet":
        return f"https://explorer.perawallet.app/tx/{txid}"
    if net == "testnet":
        return f"https://testnet.explorer.perawallet.app/tx/{txid}"
    # sandbox/custom → nessun explorer garantito
    return None

# ---------------------------------------------------------------------
# Core ops
# ---------------------------------------------------------------------
def wait_for_confirmation(client: algod.AlgodClient, txid: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Attende che la transazione sia confermata (o timeout in N round).
    Ritorna il pending info finale (dict) alla conferma.
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
    Pubblica **PoVal p1** in nota, usando i **byte canonici ACJ-1**.
    - NON effettua fallback: se supera `max_note_bytes` solleva AlgorandError.
    - Ritorna: txid, note_size, note_sha256, is_compacted(False), confirmed_round.
    """
    if not isinstance(p1_obj, dict) or p1_obj.get("s") != "p1":
        raise AlgorandError("Invalid p1 object (missing s='p1').")

    # Byte canonici (ACJ-1)
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
    **LEGACY**: self-transfer 0 ALGO con nota JSON 'aioracle:v2' (o ref/hash se troppo grande).
    Ritorna: txid, note_size, note_sha256, is_compacted, fallback_url_used, confirmed_round.
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
    Crea un ASA (1 unità, 0 decimali) con manager/reserve/freeze/clawback = sender.
    - metadata_content (opz.): usato per metadata_hash (SHA-256 dei byte)
    - url: URL scritto nell'ASA; se assente usa default_url
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

def get_tx_note_info(txid: str) -> TxNoteInfo:
    """
    Recupera info (round) + nota decodificata. Se la nota è JSON, la ritorna come dict.
    """
    client = create_algod_client()
    info = wait_for_confirmation(client, txid, timeout=1)

    note_b64 = None
    if isinstance(info, dict):
        note_b64 = info.get("note") or (info.get("txn") or {}).get("txn", {}).get("note")

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

    return TxNoteInfo(
        confirmed_round=int(info.get("confirmed-round")) if isinstance(info, dict) and info.get("confirmed-round") else None,
        note_size=(len(note_bytes) if note_bytes else None),
        note_sha256=(hashlib.sha256(note_bytes).hexdigest() if note_bytes else None),
        note_json=note_json,
        raw=info if isinstance(info, dict) else {},
    )
