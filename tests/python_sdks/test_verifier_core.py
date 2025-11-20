import base64
import json

import pytest

from axiomatic_verifier import verify_tx, to_jcs_bytes, sha256_hex
import axiomatic_verifier.verifier as vmod
from axiomatic_proofkit.jcs import to_jcs_bytes as pk_to_jcs_bytes  # for cross-check


def test_jcs_consistency_between_proofkit_and_verifier():
    """
    JCS implementation in verifier must be compatible with the one in proofkit
    for typical objects (dicts/lists/primitive types).
    """
    obj = {
        "b": 2,
        "a": 1,
        "nested": {"z": 1, "m": 2},
        "list": [3, 2, 1],
        "flag": True,
        "none": None,
    }

    bytes_verifier = to_jcs_bytes(obj)
    bytes_proofkit = pk_to_jcs_bytes(obj)

    assert bytes_verifier == bytes_proofkit
    assert sha256_hex(bytes_verifier) == sha256_hex(bytes_proofkit)


def test_verify_tx_tx_not_found(monkeypatch):
    """
    When _fetch_tx raises, verify_tx should return a structured error.
    """

    def fake_fetch(txid, network, indexer_url=None):
        raise RuntimeError("tx_not_found:404")

    monkeypatch.setattr(vmod, "_fetch_tx", fake_fetch)

    res = verify_tx("FAKE_TX", network="testnet")
    assert res["txid"] == "FAKE_TX"
    assert res["verified"] is False
    assert res["mode"] == "unknown"
    assert "tx_not_found" in res["reason"]


def test_verify_tx_note_missing(monkeypatch):
    """
    Transaction without a note should yield reason='note_missing'.
    """

    def fake_fetch(txid, network, indexer_url=None):
        return {
            "transaction": {
                "id": txid,
                "confirmed-round": 123,
                # no 'note' field
            }
        }

    monkeypatch.setattr(vmod, "_fetch_tx", fake_fetch)

    res = verify_tx("FAKE_TX", network="testnet")
    assert res["verified"] is False
    assert res["mode"] == "unknown"
    assert res["reason"] == "note_missing"
    assert "/tx/" in res["explorer_url"]


def test_verify_tx_note_not_json(monkeypatch):
    """
    Note that decodes but is not a JSON object (e.g. JSON string) should yield
    reason='note_not_json'.
    """
    note_payload = json.dumps("just-a-string").encode("utf-8")
    note_b64 = base64.b64encode(note_payload).decode("ascii")

    def fake_fetch(txid, network, indexer_url=None):
        return {
            "transaction": {
                "id": txid,
                "confirmed-round": 123,
                "note": note_b64,
            }
        }

    monkeypatch.setattr(vmod, "_fetch_tx", fake_fetch)

    res = verify_tx("FAKE_TX", network="testnet")
    assert res["verified"] is False
    assert res["mode"] == "unknown"
    assert res["reason"] == "note_not_json"


def test_verify_tx_legacy_mode(monkeypatch):
    """
    Notes with 'ref' or 'schema_version' are treated as 'legacy' and considered
    verified=True but mode='legacy'.
    """

    def fake_fetch(txid, network, indexer_url=None):
        return {
            "transaction": {
                "id": txid,
                "confirmed-round": 123,
                "note": "ignored",
            }
        }

    def fake_decode(note_b64):
        # Emulate a legacy note (dict with 'ref')
        note = {"ref": "legacy-valuation"}
        return b"{}", note

    monkeypatch.setattr(vmod, "_fetch_tx", fake_fetch)
    monkeypatch.setattr(vmod, "_decode_note", fake_decode)

    res = verify_tx("FAKE_TX", network="testnet")
    assert res["verified"] is True
    assert res["mode"] == "legacy"
    assert res["reason"] is None


def _make_p1_note(ts: int, with_hash: bool = False, bad_hash: bool = False) -> dict:
    note = {
        "s": "p1",
        "a": "re:EUR",
        "mv": "v1",
        "mh": "",
        "ih": "88" * 32,
        "v": 100_000.0,
        "u": [90_000.0, 110_000.0],
        "ts": ts,
    }
    if with_hash:
        canonical_bytes = to_jcs_bytes(note)
        good_sha = sha256_hex(canonical_bytes)
        note["note_sha256"] = "deadbeef" * 8 if bad_hash else good_sha
    return note


def test_verify_tx_p1_happy_path(monkeypatch):
    """
    P1 note with matching hash and in-window ts should verify successfully.
    """
    fake_now = 1_700_000_000
    note = _make_p1_note(ts=fake_now, with_hash=True, bad_hash=False)

    def fake_fetch(txid, network, indexer_url=None):
        return {
            "transaction": {
                "id": txid,
                "confirmed-round": 321,
                "note": "ignored",
            }
        }

    def fake_decode(note_b64):
        return b"{}", note

    monkeypatch.setattr(vmod, "_fetch_tx", fake_fetch)
    monkeypatch.setattr(vmod, "_decode_note", fake_decode)
    monkeypatch.setattr(vmod.time, "time", lambda: fake_now)

    res = verify_tx("FAKE_TX", network="testnet")
    assert res["verified"] is True
    assert res["mode"] == "p1"
    assert res["reason"] is None
    assert res["confirmed_round"] == 321
    assert res["note"] == note
    assert res["note_sha256"] == res["rebuilt_sha256"]


def test_verify_tx_p1_onchain_hash_mismatch(monkeypatch):
    """
    If note_sha256 (or sha256) is present and does not match the rebuilt hash,
    verification should fail with reason='onchain_hash_mismatch'.
    """
    fake_now = 1_700_000_000
    note = _make_p1_note(ts=fake_now, with_hash=True, bad_hash=True)

    def fake_fetch(txid, network, indexer_url=None):
        return {
            "transaction": {
                "id": txid,
                "confirmed-round": 321,
                "note": "ignored",
            }
        }

    def fake_decode(note_b64):
        return b"{}", note

    monkeypatch.setattr(vmod, "_fetch_tx", fake_fetch)
    monkeypatch.setattr(vmod, "_decode_note", fake_decode)
    monkeypatch.setattr(vmod.time, "time", lambda: fake_now)

    res = verify_tx("FAKE_TX", network="testnet")
    assert res["verified"] is False
    assert res["mode"] == "p1"
    assert res["reason"] == "onchain_hash_mismatch"
    assert res["note_sha256"] != res["rebuilt_sha256"]


def test_verify_tx_p1_ts_out_of_window(monkeypatch):
    """
    P1 note with ts outside allowed skew window should fail with
    reason='ts_out_of_window'.
    """
    fake_now = 1_700_000_000
    # Put ts well in the past beyond default skew
    ts_past = fake_now - (vmod.DEFAULT_SKEW_PAST_SEC + 10)
    note = _make_p1_note(ts=ts_past, with_hash=False)

    def fake_fetch(txid, network, indexer_url=None):
        return {
            "transaction": {
                "id": txid,
                "confirmed-round": 321,
                "note": "ignored",
            }
        }

    def fake_decode(note_b64):
        return b"{}", note

    monkeypatch.setattr(vmod, "_fetch_tx", fake_fetch)
    monkeypatch.setattr(vmod, "_decode_note", fake_decode)
    monkeypatch.setattr(vmod.time, "time", lambda: fake_now)

    res = verify_tx("FAKE_TX", network="testnet")
    assert res["verified"] is False
    assert res["mode"] == "p1"
    assert res["reason"] == "ts_out_of_window"
