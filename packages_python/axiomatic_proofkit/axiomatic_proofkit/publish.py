from __future__ import annotations
import base64
from typing import Any, Dict, Optional, Callable, Union
from .build import assert_note_size_ok, canonical_note_bytes_p1

class PublishError(RuntimeError):
    pass

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

    base = (
        "https://mainnet-api.algonode.cloud"
        if network == "mainnet"
        else "https://testnet-api.algonode.cloud"
    )

    return algosdk.v2client.algod.AlgodClient("", base)


def _as_bytes(x: BytesLike) -> bytes:
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
    from_addr: Optional[str] = None,
    sign: Optional[Callable[[bytes], SignedLike]] = None,
    wait_rounds: int = 4,
) -> Dict[str, Any]:
    algosdk, transaction = _require_algosdk()

    assert_note_size_ok(p1)
    note_bytes, sha_hex, size = canonical_note_bytes_p1(p1)

    client = _algod_client(network, algod)
    params = client.suggested_params()

    if not from_addr:
        raise PublishError("from_addr is required")
    if not callable(sign):
        raise PublishError("sign function is required (sign(unsigned_tx_bytes)->signed_tx)")

    txn = transaction.PaymentTxn(
        sender=from_addr,
        sp=params,
        receiver=from_addr,
        amt=0,
        note=note_bytes,
    )

    unsigned_b64 = algosdk.encoding.msgpack_encode(txn)
    unsigned_bytes = algosdk.encoding.base64.b64decode(unsigned_b64)

    signed_like: SignedLike = sign(unsigned_bytes)

    if isinstance(signed_like, str):
        signed_b64 = signed_like
    elif isinstance(signed_like, (bytes, bytearray, memoryview)):
        signed_b64 = base64.b64encode(_as_bytes(signed_like)).decode("ascii")
    else:
        raise PublishError("sign must return base64 str or bytes-like")

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
        msg = str(e)
        if "already in ledger" in msg or "already in ledger" in getattr(e, "message", ""):
            txid = txn.get_txid()
        else:
            raise

    try:
        from algosdk import transaction as _txn
        _txn.wait_for_confirmation(client, txid, wait_rounds)
    except Exception:
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

    explorer = (
    f"https://explorer.perawallet.app/tx/{txid}"
    if network == "mainnet"
    else f"https://testnet.explorer.perawallet.app/tx/{txid}"
    )
    
    return {
        "txid": txid,
        "explorer_url": explorer,
        "note_sha256": sha_hex,
        "note_size": size,
        "network": network,
    }