from __future__ import annotations
"""
axiomatic_proofkit.publish — pubblicazione p1 come nota Algorand.

API simmetrica a TS:
- from_addr  ≈ "from"
- sign(unsigned) -> (bytes | base64 str)

Dipendenze runtime: opzionale `algosdk` (installare con `pip install py-algorand-sdk`)
"""
import base64
from typing import Any, Dict, Optional, Callable, Union
from .build import assert_note_size_ok, canonical_note_bytes_p1

class PublishError(RuntimeError):
    pass

# Tipi accettati dallo signer
BytesLike = Union[bytes, bytearray, memoryview]
SignedLike = Union[BytesLike, str]


def _require_algosdk():
    try:
        import algosdk  # type: ignore
        from algosdk import transaction  # type: ignore
        return algosdk, transaction
    except Exception as e:
        raise PublishError("algosdk is required for publish; install with `pip install py-algorand-sdk`") from e


def _algod_client(network: str, algod: Any | None):
    if algod:
        return algod
    algosdk, _ = _require_algosdk()
    base = f"https://node.{'' if network == 'mainnet' else 'testnet.'}algoexplorerapi.io"
    return algosdk.v2client.algod.AlgodClient("", base)


def _as_bytes(x: BytesLike) -> bytes:
    """Normalizza qualsiasi bytes-like in bytes."""
    if isinstance(x, bytes):
        return x
    if isinstance(x, bytearray):
        return bytes(x)
    if isinstance(x, memoryview):
        return x.tobytes()
    raise TypeError("Expected bytes-like (bytes|bytearray|memoryview)")


def publish_p1(
    p1: Dict[str, Any],
    *,
    network: str = "testnet",
    algod: Any | None = None,
    from_addr: Optional[str] = None,                                  # ≈ "from" (TS)
    sign: Optional[Callable[[bytes], SignedLike]] = None,             # unsigned msgpack bytes -> (signed bytes | base64 str)
    wait_rounds: int = 4,
) -> Dict[str, Any]:
    """
    Pubblica p1 come nota (0 ALGO to self).

    Parametri:
      - from_addr: indirizzo mittente
      - sign: funzione che accetta i BYTES della transazione NON firmata (msgpack)
              e ritorna *o* i BYTES della transazione firmata *o* la stringa base64.
      - algod: client Algod opzionale già configurato
      - network: "testnet" | "mainnet"
      - wait_rounds: round di attesa per la conferma

    Ritorna:
      { txid, explorer_url, note_sha256, note_size, network }
    """
    algosdk, transaction = _require_algosdk()

    # Bytes canonici + size + sha (da build.py)
    assert_note_size_ok(p1)
    note_bytes, sha_hex, size = canonical_note_bytes_p1(p1)

    client = _algod_client(network, algod)
    params = client.suggested_params()

    if not from_addr:
        raise PublishError("from_addr is required")
    if not callable(sign):
        raise PublishError("sign function is required (sign(unsigned_tx_bytes)->signed_tx)")

    # Costruisci tx: 0 ALGO a se stessi con nota p1
    txn = transaction.PaymentTxn(
        sender=from_addr,
        sp=params,
        receiver=from_addr,
        amt=0,
        note=note_bytes,
    )

    # Encode unsigned txn -> msgpack bytes (lo passeremo allo signer)
    unsigned_b64 = algosdk.encoding.msgpack_encode(txn)      # base64 str dell'UNSIGNED
    unsigned_bytes = algosdk.encoding.base64.b64decode(unsigned_b64)

    # Firma esterna: può restituire bytes-like *oppure* base64 str
    signed_like: SignedLike = sign(unsigned_bytes)

    # Normalizza sempre a base64 str per compatibilità con send_raw_transaction
    if isinstance(signed_like, str):
        signed_b64 = signed_like
    elif isinstance(signed_like, (bytes, bytearray, memoryview)):
        signed_b64 = base64.b64encode(_as_bytes(signed_like)).decode("ascii")
    else:
        raise PublishError("sign must return base64 str or bytes-like")

    # Invia (SDK può ritornare str oppure dict)
    try:
        send_res = client.send_raw_transaction(signed_b64)
        if isinstance(send_res, str):
            txid = send_res
        elif isinstance(send_res, dict):
            txid = send_res.get("txId") or send_res.get("txid") or send_res.get("txID")
            if not txid:
                for v in send_res.values():
                    if isinstance(v, str):
                        txid = v
                        break
            if not txid:
                raise PublishError(f"Unexpected send_raw_transaction response: {send_res!r}")
        else:
            raise PublishError(f"Unexpected send_raw_transaction response type: {type(send_res)}")
    except Exception as e:
        # Se è "already in ledger", calcola il txid dall'UNSIGNED e prosegui
        msg = str(e)
        if "already in ledger" in msg or "already in ledger" in getattr(e, "message", ""):
            # Il txid su Algorand dipende dall'UNSIGNED (non dalla firma)
            txid = txn.get_txid()
        else:
            raise

    # --- Attesa conferma (compat SDK) ---
    try:
        from algosdk import transaction as _txn
        _txn.wait_for_confirmation(client, txid, wait_rounds)
    except Exception:
        # fallback: poll manuale
        status = client.status()
        last = status.get("last-round") or status.get("lastRound") or 0
        deadline = last + int(wait_rounds)
        while last <= deadline:
            info = client.pending_transaction_info(txid)
            confirmed = info.get("confirmed-round") or info.get("confirmedRound") or 0
            if confirmed and confirmed > 0:
                break
            last += 1
            client.status_after_block(last)


    explorer = f"https://{'' if network=='mainnet' else 'testnet.'}algoexplorer.io/tx/{txid}"
    return {
        "txid": txid,
        "explorer_url": explorer,
        "note_sha256": sha_hex,
        "note_size": size,
        "network": network,
    }