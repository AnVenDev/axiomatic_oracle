import sys
import pathlib
import pytest
from jsonschema import validate as jsonschema_validate, ValidationError
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.inference_api import app, OUTPUT_SCHEMA

client = TestClient(app)

# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

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
        "age_years": 26
    }

@pytest.fixture
def payload_no_age(sample_payload):
    p = dict(sample_payload)
    p.pop("age_years", None)
    return p

# -------------------------------------------------------------------
# Tests - Health & Predict
# -------------------------------------------------------------------

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d.get("status") in ("ok", "degraded")
    assert "asset_types" in d
    assert "schema_version" in d
    assert "api_version" in d


def test_predict_property(sample_payload):
    r = client.post("/predict/property", json=sample_payload)
    assert r.status_code == 200, f"Response code: {r.status_code}, Body: {r.text}"
    data = r.json()

    # Top-level keys
    for key in ("schema_version", "asset_id", "asset_type", "timestamp", "metrics", "flags", "model_meta"):
        assert key in data, f"Missing key in response: {key}"

    assert isinstance(data["metrics"]["valuation_base_k"], (int, float))
    assert data["flags"] == {"anomaly": False, "needs_review": False}

    if "dataset_hash" in data.get("model_meta", {}):
        assert isinstance(data["model_meta"]["dataset_hash"], str)

    # Schema validation
    try:
        jsonschema_validate(instance=data, schema=OUTPUT_SCHEMA)
    except ValidationError as ve:
        pytest.fail(f"Schema validation failed: {ve.message}")


def test_predict_property_derive_age(payload_no_age):
    r = client.post("/predict/property", json=payload_no_age)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "valuation_base_k" in data["metrics"]
    try:
        jsonschema_validate(instance=data, schema=OUTPUT_SCHEMA)
    except ValidationError as ve:
        pytest.fail(f"Schema validation failed (derive age): {ve.message}")


# -------------------------------------------------------------------
# Tests - Validation Errors
# -------------------------------------------------------------------

def test_predict_validation_error(sample_payload):
    bad = dict(sample_payload)
    bad["floor"] = bad["building_floors"]  # invalid: floor must be < building_floors
    r = client.post("/predict/property", json=bad)
    assert r.status_code == 422
    assert "Invalid payload" in r.json().get("detail", "")


def test_predict_energy_class_error(sample_payload):
    bad = dict(sample_payload)
    bad["energy_class"] = "Z"  # not in accepted values
    r = client.post("/predict/property", json=bad)
    assert r.status_code == 422
    assert "energy_class must be one of" in r.json().get("detail", "")


def test_predict_missing_required_field(sample_payload):
    bad = dict(sample_payload)
    bad.pop("location")
    r = client.post("/predict/property", json=bad)
    assert r.status_code == 422


def test_predict_wrong_type_field(sample_payload):
    bad = dict(sample_payload)
    bad["size_m2"] = "large"
    r = client.post("/predict/property", json=bad)
    assert r.status_code == 422


# -------------------------------------------------------------------
# Test with Publish Simulated
# -------------------------------------------------------------------

def test_predict_with_publish(sample_payload):
    r = client.post("/predict/property?publish=true", json=sample_payload)
    assert r.status_code == 200
    d = r.json()
    assert "publish" in d
    pub = d["publish"]
    assert pub.get("status") in ("simulated", "success")
    assert isinstance(pub.get("txid"), str)
    try:
        jsonschema_validate(instance=d, schema=OUTPUT_SCHEMA)
    except ValidationError as ve:
        pytest.fail(f"Schema validation failed (publish): {ve.message}")