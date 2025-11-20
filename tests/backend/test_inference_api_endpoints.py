import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from fastapi.testclient import TestClient

from scripts import inference_api as api


# ---------------------------------------------------------------------
# Dummy pipeline + registry to avoid loading real models / Algorand
# ---------------------------------------------------------------------
class DummyTree:
    """Simple tree-like estimator used to emulate per-tree predictions."""

    def __init__(self, value: float) -> None:
        self._value = float(value)

    def predict(self, X) -> np.ndarray:  # noqa: N803
        # Ignore X, always return a single value
        return np.array([self._value], dtype=float)


class DummyModel:
    """Simple ensemble-like model with estimators_ attribute."""

    def __init__(self) -> None:
        # Three slightly different predictions so that std > 0
        self.estimators_ = [
            DummyTree(190.0),
            DummyTree(210.0),
            DummyTree(200.0),
        ]

    def predict(self, X) -> np.ndarray:  # noqa: N803
        return np.array([200.0], dtype=float)


class DummyPipeline:
    """
    Minimal pipeline-like object exposing:
    - .predict(...)
    - .estimator (so that _unwrap_final_estimator can find DummyModel)
    """

    def __init__(self) -> None:
        self.estimator = DummyModel()

    def predict(self, X) -> np.ndarray:  # noqa: N803
        return self.estimator.predict(X)


class DummyRegistry:
    """Minimal AttestationRegistry replacement for tests."""

    def __init__(self) -> None:
        self.seen_calls = []
        self.record_calls = []

    def seen(self, p1_sha: str, asset_id: str) -> bool:
        self.seen_calls.append((p1_sha, asset_id))
        # Never mark as replay in tests
        return False

    def record(self, *args: Any, **kwargs: Any) -> None:
        self.record_calls.append((args, kwargs))


# ---------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------
@pytest.fixture(autouse=True)
def disable_auth_and_reset_rate_limit(monkeypatch):
    """
    Ensure the API runs in "open" mode for tests and reset rate limiter.
    """
    # Disable Bearer token requirement
    api.API_KEY = None  # type: ignore[attr-defined]

    # Reset rate limit bucket between tests
    if hasattr(api, "_rate_bucket"):
        api._rate_bucket.clear()  # type: ignore[attr-defined]

    yield

    if hasattr(api, "_rate_bucket"):
        api._rate_bucket.clear()  # type: ignore[attr-defined]


@pytest.fixture
def client(monkeypatch, tmp_path) -> TestClient:
    """
    TestClient with all heavy dependencies mocked so that:
    - no real model files are required
    - no Algorand / network calls are performed
    - no real files are written outside a temp directory
    """
    # ---- Environment / paths isolation ----
    monkeypatch.setenv("OUTPUTS_DIR", str(tmp_path))
    api.API_LOG_PATH = tmp_path / "api_inference_log.jsonl"  # type: ignore[attr-defined]

    # ---- Model registry / pipeline mocks ----
    dummy_meta: Dict[str, Any] = {
        "model_version": "1.0.0-test",
        "model_class": "DummyModel",
        "metrics": {"MAE": 10.0, "R2": 0.9},
        # A small but realistic feature_order
        "feature_order": [
            "location",
            "size_m2",
            "rooms",
            "bathrooms",
            "year_built",
            "floor",
            "building_floors",
            "has_elevator",
            "has_garden",
            "has_balcony",
            "has_garage",
            "energy_class",
            "humidity_level",
            "temperature_avg",
            "noise_level",
            "air_quality_index",
            "age_years",
            "listing_month",
            "city",
            "region",
            "zone",
            "public_transport_nearby",
        ],
    }

    def fake_get_pipeline(asset_type: str, task: str) -> DummyPipeline:
        return DummyPipeline()

    def fake_get_model_paths(asset_type: str, task: str) -> Dict[str, str]:
        return {
            "pipeline": "dummy.joblib",
            "manifest": "",
        }

    def fake_get_model_metadata(asset_type: str, task: str) -> Dict[str, Any]:
        return dict(dummy_meta)

    monkeypatch.setattr(api, "get_pipeline", fake_get_pipeline)
    monkeypatch.setattr(api, "get_model_paths", fake_get_model_paths)
    monkeypatch.setattr(api, "get_model_metadata", fake_get_model_metadata)

    # ---- Validation / explanation / pricing helpers ----
    def fake_validate_property(base: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True, "warnings": [], "errors": []}

    def fake_explain_price(base: Dict[str, Any]) -> Dict[str, Any]:
        return {"components": []}

    def fake_price_benchmark(location: Any, valuation_k: float) -> Dict[str, Any]:
        return {
            "location": location,
            "valuation_k": valuation_k,
            "out_of_band": False,
        }

    monkeypatch.setattr(api, "validate_property", fake_validate_property)
    monkeypatch.setattr(api, "explain_price", fake_explain_price)
    monkeypatch.setattr(api, "price_benchmark", fake_price_benchmark)

    # ---- PoVal builder / canonicalization ----
    def fake_build_p1_from_response(response: Dict[str, Any], allowed_input_keys: Any):
        p1 = {
            "s": "p1",
            "v": 200.0,
            "u": [180.0, 220.0],
            "ts": 1_700_000_000,
        }
        dbg = {"ih": "fake_input_hash"}
        return p1, dbg

    def fake_canonical_note_bytes_p1(p1: Dict[str, Any]):
        # bytes, sha256, size
        return b"{}", "fake_p1_sha256", 128

    monkeypatch.setattr(api, "build_p1_from_response", fake_build_p1_from_response)
    monkeypatch.setattr(api, "canonical_note_bytes_p1", fake_canonical_note_bytes_p1)

    # ---- Audit bundle / registry / network helpers ----
    dummy_registry = DummyRegistry()
    api.registry = dummy_registry  # type: ignore[attr-defined]

    def fake_save_audit_bundle(bundle_dir: Path, **kwargs: Any) -> None:
        # Do nothing; we just want the call to succeed
        return None

    def fake_publish_ai_prediction(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            "blockchain_txid": "FAKE_TXID",
            "asa_id": 12345,
            "note_size": 128,
            "note_sha256": "fake_p1_sha256",
            "is_compacted": True,
            "confirmed_round": 42,
        }

    def fake_get_network() -> str:
        return "testnet"

    def fake_get_tx_note_info(txid: str) -> Dict[str, Any]:
        # Only used when publish=True; here we simulate an indexer response
        return {
            "note_json": {"s": "p1"},
            "note_sha256": "fake_p1_sha256",
            "note_size": 128,
            "confirmed_round": 42,
            "explorer_url": f"https://fake.explorer/tx/{txid}",
        }

    monkeypatch.setattr(api, "save_audit_bundle", fake_save_audit_bundle)
    monkeypatch.setattr(api, "publish_ai_prediction", fake_publish_ai_prediction)
    monkeypatch.setattr(api, "get_network", fake_get_network)
    monkeypatch.setattr(api, "get_tx_note_info", fake_get_tx_note_info)

    # Health / cache helpers (used by /health)
    def fake_health_check_model(asset_type: str, task: str) -> Dict[str, Any]:
        return {"status": "healthy", "asset_type": asset_type, "task": task}

    def fake_cache_stats() -> Dict[str, Any]:
        return {"hits": 0, "misses": 0}

    monkeypatch.setattr(api, "health_check_model", fake_health_check_model)
    monkeypatch.setattr(api, "cache_stats", fake_cache_stats)

    return TestClient(api.app)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
def test_health_endpoint_ok(client: TestClient):
    """Basic sanity check for /health."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] in ("ok", "degraded")
    assert "model_health" in data
    assert "cache_stats" in data
    assert "asset_types" in data
    assert "schema_version" in data
    assert "api_version" in data


def test_predict_property_happy_path(client: TestClient):
    """
    /predict/property with minimal payload should return:
    - HTTP 200
    - valuation metrics
    - attestation info (p1 + sha)
    - audit bundle id
    """
    payload = {
        "location": "Milan",
        "size_m2": 80,
        "rooms": 3,
        "bathrooms": 2,
        "year_built": 2005,
        "floor": 2,
        "building_floors": 5,
        "has_elevator": 1,
    }

    resp = client.post("/predict/property?publish=false&attestation_only=false", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert data["asset_type"] == "property"
    assert "asset_id" in data
    assert "metrics" in data
    assert "attestation" in data
    assert "audit_bundle" in data

    metrics = data["metrics"]
    assert "valuation_k" in metrics
    assert "confidence_low_k" in metrics
    assert "confidence_high_k" in metrics
    assert metrics["valuation_k"] > 0


def test_predict_property_attestation_only(client: TestClient):
    """
    /predict/property with attestation_only=true should return
    a compact structure focused on the attestation.
    """
    payload = {
        "location": "Rome",
        "size_m2": 60,
        "rooms": 2,
        "bathrooms": 1,
        "year_built": 2010,
    }

    resp = client.post("/predict/property?publish=false&attestation_only=true", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    # Compact response: should NOT contain full metrics
    assert "asset_id" in data
    assert "attestation" in data
    assert "attestation_sha256" in data
    assert "attestation_size" in data
    assert "txid" in data
    assert data.get("published") is False
    assert "metrics" not in data  # ensure we really returned the compact form


def test_verify_missing_txid_returns_422(client: TestClient):
    """POST /verify without 'txid' should return 422."""
    resp = client.post("/verify", json={})
    assert resp.status_code == 422
