import hashlib
import json
import time

import pytest

from axiomatic_proofkit import (
    build_p1,
    canonical_note_bytes_p1,
    assert_note_size_ok,
    build_canonical_input,
    compute_input_hash,
)


def _dummy_hash() -> str:
    # 64 hex chars
    return "ab" * 32


def test_build_p1_happy_path():
    input_hash = "11" * 32  # 64 hex chars
    ts = 1_700_000_000

    p1 = build_p1(
        asset_tag="re:EUR",
        model_version="v1",
        model_hash_hex=_dummy_hash(),
        input_hash_hex=input_hash,
        value_eur=100_000.0,
        uncertainty_low_eur=90_000.0,
        uncertainty_high_eur=110_000.0,
        timestamp_epoch=ts,
    )

    assert p1["s"] == "p1"
    assert p1["a"] == "re:EUR"
    assert p1["mv"] == "v1"
    assert p1["mh"] == _dummy_hash()
    assert p1["ih"] == input_hash
    assert p1["v"] == pytest.approx(100_000.0)
    assert p1["u"] == [pytest.approx(90_000.0), pytest.approx(110_000.0)]
    assert p1["ts"] == ts


def test_build_p1_auto_timestamp_is_int(monkeypatch):
    input_hash = "22" * 32
    fake_now = 1_800_000_000

    monkeypatch.setattr(time, "time", lambda: fake_now)

    p1 = build_p1(
        model_version="v1",
        input_hash_hex=input_hash,
        value_eur=50_000.0,
        uncertainty_low_eur=40_000.0,
        uncertainty_high_eur=60_000.0,
    )
    assert isinstance(p1["ts"], int)
    assert p1["ts"] == fake_now


def test_build_p1_validation_errors():
    good_hash = "33" * 32

    # low > high
    with pytest.raises(ValueError):
        build_p1(
            model_version="v1",
            input_hash_hex=good_hash,
            value_eur=100_000.0,
            uncertainty_low_eur=110_000.0,
            uncertainty_high_eur=90_000.0,
        )

    # invalid input_hash_hex length
    with pytest.raises(ValueError):
        build_p1(
            model_version="v1",
            input_hash_hex="short",
            value_eur=100_000.0,
            uncertainty_low_eur=90_000.0,
            uncertainty_high_eur=110_000.0,
        )

    # invalid model_hash_hex length (when provided)
    with pytest.raises(ValueError):
        build_p1(
            model_version="v1",
            model_hash_hex="zz",  # not 64 chars
            input_hash_hex=good_hash,
            value_eur=100_000.0,
            uncertainty_low_eur=90_000.0,
            uncertainty_high_eur=110_000.0,
        )

    # non-finite value_eur
    with pytest.raises(ValueError):
        build_p1(
            model_version="v1",
            input_hash_hex=good_hash,
            value_eur=float("inf"),
            uncertainty_low_eur=90_000.0,
            uncertainty_high_eur=110_000.0,
        )


def test_canonical_note_bytes_and_size():
    input_hash = "44" * 32
    p1 = build_p1(
        model_version="v1",
        input_hash_hex=input_hash,
        value_eur=100_000.0,
        uncertainty_low_eur=90_000.0,
        uncertainty_high_eur=110_000.0,
        timestamp_epoch=1_700_000_000,
    )

    note_bytes, sha_hex, size = canonical_note_bytes_p1(p1)

    assert isinstance(note_bytes, (bytes, bytearray))
    assert isinstance(sha_hex, str)
    assert isinstance(size, int)
    assert size == len(note_bytes)

    expected_sha = hashlib.sha256(note_bytes).hexdigest()
    assert sha_hex == expected_sha


def test_canonical_note_bytes_requires_p1_schema():
    with pytest.raises(ValueError):
        canonical_note_bytes_p1({"s": "other"})


def test_assert_note_size_ok_passes_for_small_note():
    input_hash = "55" * 32
    p1 = build_p1(
        model_version="v1",
        input_hash_hex=input_hash,
        value_eur=10_000.0,
        uncertainty_low_eur=9_000.0,
        uncertainty_high_eur=11_000.0,
    )

    # Use default max_bytes (internal constant) – should not raise
    assert_note_size_ok(p1)


def test_assert_note_size_ok_raises_when_too_large():
    input_hash = "66" * 32
    p1 = build_p1(
        model_version="v1",
        input_hash_hex=input_hash,
        value_eur=10_000.0,
        uncertainty_low_eur=9_000.0,
        uncertainty_high_eur=11_000.0,
    )

    # Force a tiny max_bytes to trigger the failure
    with pytest.raises(ValueError):
        assert_note_size_ok(p1, max_bytes=10)


def test_build_canonical_input_and_compute_input_hash():
    rec = {
        "b": 2,
        "a": 1,
        "c": None,
        "extra": 999,
    }
    allowed_keys = ["a", "b", "c"]
    cin = build_canonical_input(rec, allowed_keys=allowed_keys, strip_none=True)

    # c is None and strip_none=True, so we only keep a and b
    assert set(cin.keys()) == {"a", "b"}
    assert cin["a"] == 1
    assert cin["b"] == 2

    # Expected canonical string and hash
    expected_json = '{"a":1,"b":2}'
    expected_sha = "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777"

    s = json.dumps(cin, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    assert s == expected_json

    ih = compute_input_hash(rec, allowed_keys=allowed_keys)
    assert ih == expected_sha
