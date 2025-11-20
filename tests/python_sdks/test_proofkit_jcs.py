import math

import pytest

from axiomatic_proofkit import to_jcs_bytes, sha256_hex


def test_jcs_deterministic_order_and_hash():
    """
    JCS must produce a deterministic, canonical JSON representation
    (sorted keys, no extra whitespace) and a stable SHA-256.
    """
    obj = {
        "b": 2,
        "a": 1,
        "nested": {"z": 1, "m": 2},
        "list": [3, 2, 1],
        "flag": True,
        "none": None,
    }

    # Expected canonical JSON string (keys sorted, compact separators)
    expected_json = (
        '{"a":1,"b":2,"flag":true,"list":[3,2,1],'
        '"nested":{"m":2,"z":1},"none":null}'
    )

    b = to_jcs_bytes(obj)
    assert b.decode("utf-8") == expected_json

    expected_sha = "b422257a403826b661ba7317cf8f251ae1e5c2dab2d2b2194cab6c2c18b7324f"
    assert sha256_hex(b) == expected_sha


def test_jcs_rejects_non_finite_floats():
    """NaN / Inf must be rejected in canonical JSON."""
    with pytest.raises(ValueError):
        to_jcs_bytes({"x": float("inf")})

    with pytest.raises(ValueError):
        to_jcs_bytes({"x": float("-inf")})

    with pytest.raises(ValueError):
        to_jcs_bytes({"x": math.nan})
