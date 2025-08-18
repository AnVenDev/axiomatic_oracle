# tests/test_api.py
from __future__ import annotations

import os
import pathlib
import sys
from typing import List
from unittest.mock import patch
import numpy as np  # type: ignore
import pytest  # type: ignore
from fastapi.testclient import TestClient  # type: ignore
from jsonschema import ValidationError  # type: ignore
from jsonschema import validate as jsonschema_validate  # type: ignore

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_nbdir = ROOT / "notebooks"
if _nbdir.exists() and str(_nbdir) not in sys.path:
    sys.path.insert(0, str(_nbdir))

from scripts.inference_api import OUTPUT_SCHEMA, SCHEMA_VERSION, app  # noqa: E402

# ---------------------------------------------------------------------
# Dummy pipeline (replica l’interfaccia usata dall’endpoint)
# ---------------------------------------------------------------------
class _DummyLast:
    def __init__(self) -> None:
        self.feature_name_: list[str] = []

    def predict(self, X):
        return np.full(len(X), 123.456)

class _DummySlice:
    def __init__(self, last: _DummyLast) -> None:
        self._last = last

    def transform(self, df):
        self._last.feature_name_ = list(df.columns)
        return df.to_numpy()

class _DummyPipeline:
    def __init__(self) -> None:
        self._last = _DummyLast()

    def __getitem__(self, key):
        if isinstance(key, slice):
            return _DummySlice(self._last)
        if key == -1:
            return self._last
        raise IndexError

def _meta():
    return {
        "model_version": "vTest",
        "model_class": "DummyRegressor",
        "model_path": "dummy",
        "metrics": {"r2": 0.9, "mae_k": 1.0, "rmse_k": 2.0},
    }

def _health():
    return {
        "status": "healthy",
        "model_path": "dummy",
        "size_mb": 0.1,
        "last_modified": "now",
        "metadata_valid": True,
        "metrics": {"r2": 0.9, "mae_k": 1.0, "rmse_k": 2.0},
    }

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _pick_valuation(metrics: dict):
    if not isinstance(metrics, dict):
        return None
    return metrics.get("valuation_k") or metrics.get("valuation_base_k")

def _maybe_validate_schema(resp: dict):
    if not OUTPUT_SCHEMA:
        return
    resp_ver = str(resp.get("schema_version") or "").lower()
    schema_ver = str(SCHEMA_VERSION or "").lower()
    if resp_ver and schema_ver and resp_ver == schema_ver:
        try:
            jsonschema_validate(instance=resp, schema=OUTPUT_SCHEMA)
        except ValidationError as ve:
            pytest.fail(f"Schema validation failed: {ve.message}")

def _patch_expected_features(monkeypatch, expected: List[str]):
    """
    Stuba *qualsiasi* risoluzione features usata dall'API:
    - funzioni tipo resolve_expected_features / get_expected_features / resolve_artifacts / load_artifacts
    - costanti tipo ALL_EXPECTED / EXPECTED_FEATURES
    Così evitiamo l’errore “Artifacts error: Empty expected features”.
    """
    import scripts.inference_api as api  # import inside to ensure module loaded

    cat = [f for f in expected if f in ("location", "energy_class")]
    num = [f for f in expected if f not in cat]

    # funzioni possibili (le patchiamo solo se esistono)
    func_stubs = {
        "resolve_expected_features": lambda *a, **k: {
            "categorical_expected": cat,
            "numeric_expected": num,
            "all_expected": expected,
        },
        "get_expected_features": lambda *a, **k: expected,
        "resolve_artifacts": lambda *a, **k: {
            "features_categorical": cat,
            "features_numeric": num,
        },
        "load_artifacts": lambda *a, **k: {
            "features_categorical": cat,
            "features_numeric": num,
        },
        "resolve_feature_config": lambda *a, **k: {
            "categorical": cat,
            "numeric": num,
        },
        "load_feature_config": lambda *a, **k: {
            "categorical": cat,
            "numeric": num,
        },
    }
    for name, stub in func_stubs.items():
        if hasattr(api, name):
            monkeypatch.setattr(api, name, stub, raising=False)

    # costanti/variabili possibili
    for name in ("ALL_EXPECTED", "EXPECTED_FEATURES", "FEATURES_ALL"):
        if hasattr(api, name):
            monkeypatch.setattr(api, name, expected, raising=False)

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_payload():
    return {
        "location": "Milan",
        "size_m2": 95,
        "rooms": 4,
        "bathrooms": 2,
        "year_built": 1999,
        "floor": 2,
        "building_floors": 6,
        "has_elevator": 1,
        "has_garden": 0,
        "has_balcony": 1,
        "garage": 1,
        "energy_class": "B",
        "humidity_level": 50.0,
        "temperature_avg": 20.5,
        "noise_level": 40,
        "air_quality_index": 70,
        "age_years": 26,
    }

@pytest.fixture
def payload_no_age(sample_payload):
    payload = dict(sample_payload)
    payload.pop("age_years", None)
    return payload

# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d.get("status") in ("ok", "degraded")
    assert "asset_types" in d
    assert "schema_version" in d
    assert "api_version" in d

def test_predict_property(client: TestClient, sample_payload, monkeypatch):
    _patch_expected_features(monkeypatch, list(sample_payload.keys()))
    with patch("scripts.inference_api.get_pipeline", return_value=_DummyPipeline()), \
         patch("scripts.inference_api.get_model_metadata", return_value=_meta()), \
         patch("scripts.inference_api.health_check_model", return_value=_health()):
        r = client.post("/predict/property", json=sample_payload)
    assert r.status_code == 200, r.text
    data = r.json()
    for key in ("schema_version", "asset_id", "asset_type", "timestamp", "metrics", "flags", "model_meta"):
        assert key in data
    val = _pick_valuation(data["metrics"])
    assert isinstance(val, (int, float))
    assert isinstance(data["flags"], dict) and "anomaly" in data["flags"] and "needs_review" in data["flags"]
    _maybe_validate_schema(data)
    assert "schema_validation_error" in data

def test_predict_property_autofill_age(client: TestClient, payload_no_age, monkeypatch):
    # usa le stesse feature della request (age_years verrà autocalcolato dal validator pydantic)
    expected_keys = list(payload_no_age.keys()) + ["age_years"]
    _patch_expected_features(monkeypatch, expected_keys)
    with patch("scripts.inference_api.get_pipeline", return_value=_DummyPipeline()), \
         patch("scripts.inference_api.get_model_metadata", return_value=_meta()), \
         patch("scripts.inference_api.health_check_model", return_value=_health()):
        r = client.post("/predict/property", json=payload_no_age)
    assert r.status_code == 200, r.text
    data = r.json()
    val = _pick_valuation(data["metrics"])
    assert isinstance(val, (int, float))
    _maybe_validate_schema(data)

def test_predict_with_publish(client: TestClient, sample_payload, monkeypatch):
    _patch_expected_features(monkeypatch, list(sample_payload.keys()))
    mocked_pub = {"asset_id": "mocked_asset_123", "blockchain_txid": "mocked_txid_12345", "asa_id": 999999}
    with patch("scripts.inference_api.get_pipeline", return_value=_DummyPipeline()), \
         patch("scripts.inference_api.get_model_metadata", return_value=_meta()), \
         patch("scripts.inference_api.health_check_model", return_value=_health()), \
         patch("scripts.inference_api.publish_ai_prediction", return_value=mocked_pub) as mock_publish:
        r = client.post("/predict/property?publish=true", json=sample_payload)
    assert r.status_code == 200, r.text
    d = r.json()
    assert "publish" in d
    pub = d["publish"]
    assert pub.get("status") in {"success", "error"}
    if pub.get("status") == "success":
        assert pub.get("txid") == mocked_pub["blockchain_txid"]
    _maybe_validate_schema(d)
    mock_publish.assert_called_once()

def test_list_models(client: TestClient):
    r = client.get("/models/property")
    assert r.status_code == 200
    data = r.json()
    assert data["asset_type"] == "property"
    assert "tasks" in data and isinstance(data["tasks"], list)
    assert "discovered_models" in data

def test_refresh_model_cache(client: TestClient):
    r = client.post("/models/property/value_regressor/refresh")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "cache_refreshed"
    assert data["asset_type"] == "property"

def test_model_health(client: TestClient):
    r = client.get("/models/property/value_regressor/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    if data.get("status") == "healthy":
        assert "model_path" in data and "size_mb" in data
