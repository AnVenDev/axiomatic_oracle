"""
Algorand utilities

Funzioni principali:
- create_algod_client(): istanzia il client con config da env (/secrets_manager)
- wait_for_confirmation(txid, timeout): attende la conferma con gestione errori robusta
- publish_to_algorand(note_dict, *, fallback_url=None): self-transfer 0 ALGO con nota JSON (≤1KB)
- create_token_for_asset(asset_name, unit_name, metadata_content=None, url=None): crea ASA 1:1 (NFT-like)
"""

from __future__ import annotations

import json
import hashlib
from typing import Any, Dict, Optional, Union

from algosdk.v2client import algod      # type: ignore
from algosdk import transaction         # type: ignore

from scripts.secrets_manager import (
    get_algod_config,
    get_account,
)

# -----------------------------------------------------------------------------
# Error type
# -----------------------------------------------------------------------------
class AlgorandError(Exception):
    """Custom exception for Algorand operations."""
    pass

# -----------------------------------------------------------------------------
# Client & account helpers
# -----------------------------------------------------------------------------
def create_algod_client() -> algod.AlgodClient:
    """
    Crea un client Algod usando la configurazione da env.
    Supporta Algonode/Sandbox/Custom e token via header.
    """
    cfg = get_algod_config()
    # Il client accetta token come stringa o headers; passiamo headers per compatibilità Algonode
    token = ""  # vuoto: usiamo solo headers
    headers = cfg.headers or {}
    return algod.AlgodClient(token, cfg.algod_url, headers)


def _require_signing_account() -> tuple[str, bytes]:
    """
    Ritorna (address, private_key). Solleva se i segreti non ci sono.
    """
    acc = get_account(require_signing=True)
    if not acc.address or not acc.private_key:
        raise AlgorandError("Signing material not available (address/private_key). Check env secrets.")
    return acc.address, acc.private_key


# -----------------------------------------------------------------------------
# Internal utils
# -----------------------------------------------------------------------------
def _ensure_note_bytes(payload: Dict[str, Any], *, fallback_url: Optional[str] = None, max_bytes: int = 1000) -> bytes:
    """
    Converte il dict in JSON minificato e si assicura che rientri nel limite note (≈1KB).
    Se eccede, sostituisce con un riferimento hash+url.
    """
    # 1) serializza minificato
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(raw) <= max_bytes:
        return raw

    # 2) se troppo lungo: crea riferimenti compatti
    h = hashlib.sha256(raw).hexdigest()
    compact = {
        "ref": "aioracle:v2",
        "schema": payload.get("schema_version"),
        "asset_id": payload.get("asset_id"),
        "hash": h,
    }
    if fallback_url:
        compact["url"] = fallback_url
    return json.dumps(compact, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _coerce_pending_info(raw: Union[Dict[str, Any], bytes]) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            raise AlgorandError("Invalid JSON bytes from pending_transaction_info")
    raise AlgorandError("Unexpected type from pending_transaction_info")


# -----------------------------------------------------------------------------
# Core ops
# -----------------------------------------------------------------------------
def wait_for_confirmation(client: algod.AlgodClient, txid: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Attende che la transazione sia confermata (o timeout in N round).
    Ritorna il pending info finale (dict) alla conferma.
    """
    try:
        status = client.status()
        last_round = status.get("last-round") if isinstance(status, dict) else 0
    except Exception as e:
        raise AlgorandError(f"Algod status error: {e}")

    for _ in range(timeout):
        info_raw = client.pending_transaction_info(txid)
        info = _coerce_pending_info(info_raw)
        if info.get("confirmed-round", 0) > 0:
            return info
        client.status_after_block(last_round + 1)
        last_round += 1

    raise AlgorandError(f"Transaction {txid} not confirmed after {timeout} rounds")


def publish_to_algorand(
    note_dict: Dict[str, Any],
    *,
    fallback_url: Optional[str] = None,
    use_flat_fee: bool = True,
) -> str:
    """
    Esegue un self-transfer 0 ALGO con nota JSON (≤ ~1KB).
    - fallback_url: opzionale, incluso nella nota compatta se la payload eccede il limite
    - use_flat_fee: se True, forza la min fee (utile per determinismo)
    Ritorna il txid.
    """
    client = create_algod_client()
    sender, sk = _require_signing_account()
    try:
        sp = client.suggested_params()
        if use_flat_fee:
            sp.flat_fee = True
            sp.fee = 1000  # min-fee

        note_bytes = _ensure_note_bytes(note_dict, fallback_url=fallback_url)
        txn = transaction.PaymentTxn(sender=sender, receiver=sender, amt=0, sp=sp, note=note_bytes)
        stx = txn.sign(sk)
        txid = client.send_transaction(stx)
        wait_for_confirmation(client, txid)
        return txid
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
    Ritorna l'asset_id (int) o solleva AlgorandError.

    - metadata_content (opz.): se presente, viene usato per metadata_hash (32-byte)
    - url: URL da scrivere nell'ASA; se assente usa default_url
    """
    client = create_algod_client()
    sender, sk = _require_signing_account()
    try:
        sp = client.suggested_params()
        if use_flat_fee:
            sp.flat_fee = True
            sp.fee = 1000  # min-fee

        mhash = hashlib.sha256(metadata_content.encode()).digest() if metadata_content else None

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
        wait_for_confirmation(client, txid)
        ptx = _coerce_pending_info(client.pending_transaction_info(txid))
        asset_id = ptx.get("asset-index")
        if not isinstance(asset_id, int):
            raise AlgorandError("ASA creation succeeded but asset-index missing in pending info")
        return asset_id
    except Exception as e:
        raise AlgorandError(f"ASA creation failed: {e}")