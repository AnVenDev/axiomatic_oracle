import base64
import types

import pytest

from axiomatic_proofkit import build_p1
import axiomatic_proofkit.publish as pub #type: ignore


class FakeAlgodClient:
    """
    Minimal stub for algosdk.v2client.algod.AlgodClient used in publish_p1.
    """

    def __init__(self, token: str, base: str) -> None:
        self.token = token
        self.base = base
        self.sent = []

    def suggested_params(self):
        # publish_p1 only needs an object to pass as sp=
        return object()

    def send_raw_transaction(self, signed_b64: str):
        self.sent.append(signed_b64)
        # Emulate algosdk behavior returning txid string
        return "FAKE_TXID_123"

    def status(self):
        return {"last-round": 1}

    def status_after_block(self, rnd: int):
        # No-op for the stub
        return {}

    def pending_transaction_info(self, txid: str):
        # Emulate a confirmed transaction
        return {"confirmed-round": 2}


class FakePaymentTxn:
    """
    Minimal stub for algosdk.transaction.PaymentTxn.
    """

    def __init__(self, sender, sp, receiver, amt, note):
        self.sender = sender
        self.sp = sp
        self.receiver = receiver
        self.amt = amt
        self.note = note

    def get_txid(self) -> str:
        # Only used in rare error paths; provide a stable value
        return "FAKE_TXID_FALLBACK"


def make_fake_algosdk_modules():
    """
    Build fake (algosdk, transaction) pair compatible with publish_p1 internals.
    """
    # encoding namespace: msgpack_encode + base64 module
    encoding = types.SimpleNamespace(
        msgpack_encode=lambda txn: base64.b64encode(b"unsigned-txn").decode("ascii"),
        base64=base64,
    )

    v2client = types.SimpleNamespace(
        algod=types.SimpleNamespace(AlgodClient=FakeAlgodClient)
    )

    fake_algosdk = types.SimpleNamespace(
        encoding=encoding,
        v2client=v2client,
    )

    fake_transaction = types.SimpleNamespace(PaymentTxn=FakePaymentTxn)

    return fake_algosdk, fake_transaction


@pytest.fixture(autouse=True)
def patch_algosdk(monkeypatch):
    """
    Automatically patch _require_algosdk in axiomatic_proofkit.publish for all tests
    in this module, so no real py-algorand-sdk or network is required.
    """
    fake_algosdk, fake_transaction = make_fake_algosdk_modules()
    monkeypatch.setattr(pub, "_require_algosdk", lambda: (fake_algosdk, fake_transaction))
    return fake_algosdk, fake_transaction


def _make_sample_p1() -> dict:
    ih = "77" * 32
    return build_p1(
        model_version="v1",
        input_hash_hex=ih,
        value_eur=123_000.0,
        uncertainty_low_eur=100_000.0,
        uncertainty_high_eur=140_000.0,
        timestamp_epoch=1_700_000_000,
    )


def test_publish_p1_success_with_fake_algosdk():
    p1 = _make_sample_p1()

    def dummy_sign(unsigned_bytes: bytes) -> bytes:
        # Return some deterministic bytes; publish_p1 will base64-encode them.
        assert isinstance(unsigned_bytes, (bytes, bytearray, memoryview))
        return b"signed-txn"

    res = pub.publish_p1(
        p1,
        network="testnet",
        from_addr="FAKEADDR123",
        sign=dummy_sign,
        wait_rounds=2,
    )

    assert res["txid"] == "FAKE_TXID_123"
    assert res["network"] == "testnet"
    assert "testnet.explorer.perawallet.app/tx/" in res["explorer_url"]
    assert isinstance(res["note_sha256"], str)
    assert isinstance(res["note_size"], int)
    assert res["note_size"] > 0


def test_publish_p1_requires_from_addr():
    p1 = _make_sample_p1()

    with pytest.raises(pub.PublishError):
        pub.publish_p1(
            p1,
            network="testnet",
            from_addr=None,  # type: ignore[arg-type]
            sign=lambda b: b"signed",
        )


def test_publish_p1_requires_sign_callable():
    p1 = _make_sample_p1()

    with pytest.raises(pub.PublishError):
        pub.publish_p1(  # type: ignore[call-arg]
            p1,
            network="testnet",
            from_addr="FAKEADDR123",
            sign=None,
        )
