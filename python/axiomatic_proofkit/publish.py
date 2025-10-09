from __future__ import annotations
"""
axiomatic_proofkit.publish — pubblicazione p1 come nota Algorand.

API simmetrica a TS:
- from_addr  ≈ "from"
- sign(unsigned: bytes-like) -> bytes  ≈ sign(unsigned: Uint8Array) -> Uint8Array

Dipendenze runtime: opzionale `algosdk` (installare con `pip install py-algorand-sdk`)
"""

from typing import Any, Dict, Optional, Callable, Union

from .build import assert_note_size_ok, canonical_note_bytes_p1


class PublishError(RuntimeError):
    pass


# Bytes-like type (compat con Uint8Array lato TS)
BytesLike = Union[bytes, bytearray, memoryview]


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
    from_addr: Optional[str] = None,                  # ≈ "from" (TS)
    sign: Optional[Callable[[BytesLike], BytesLike]] = None,  # ≈ sign(unsigned: Uint8Array) -> Uint8Array
    wait_rounds: int = 4,
) -> Dict[str, Any]:
    """
    Pubblica p1 come nota (0 ALGO to self).

    Parametri:
      - from_addr: indirizzo mittente (obbligatorio se non si fornisce un client con wallet integrato)
      - sign: funzione che accetta i BYTES della transazione NON firmata (msgpack)
              e ritorna i BYTES della transazione FIRMATA.
              (Compat con Uint8Array lato TS: qui usi bytes/bytearray/memoryview)
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
        raise PublishError("sign function is required (sign(unsigned_tx_bytes)->signed_tx_bytes)")

    # Costruisci tx: 0 ALGO a se stessi con nota p1
    txn = transaction.PaymentTxn(
        sender=from_addr,
        sp=params,
        receiver=from_addr,
        amt=0,
        note=note_bytes,
    )

    # Encode unsigned txn -> msgpack bytes
    unsigned_b64 = algosdk.encoding.msgpack_encode(txn)  # base64 str
    unsigned_bytes = algosdk.encoding.base64.b64decode(unsigned_b64)

    # Firma esterna (bytes-like in / bytes out)
    signed_bytes_like = sign(unsigned_bytes)
    signed_bytes = _as_bytes(signed_bytes_like)
    if not isinstance(signed_bytes, (bytes, bytearray)):
        raise PublishError("sign must return signed transaction bytes")

    # Invia & attendi conferma
    txid = client.send_raw_transaction(signed_bytes)["txid"]
    algosdk.v2client.algod.wait_for_confirmation(client, txid, wait_rounds)

    explorer = f"https://{'' if network=='mainnet' else 'testnet.'}algoexplorer.io/tx/{txid}"
    return {
        "txid": txid,
        "explorer_url": explorer,
        "note_sha256": sha_hex,
        "note_size": size,
        "network": network,
    }