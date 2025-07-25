from scripts.inference_api import app, OUTPUT_SCHEMA
import sys
import pathlib
import pytest
from jsonschema import validate as jsonschema_validate, ValidationError
from fastapi.testclient import TestClient
from unittest.mock import patch

# -------------------------------------------------------------------
# Set sys.path to root for absolute imports
# -------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    payload = dict(sample_payload)
    payload.pop("age_years", None)
    return payload

# -------------------------------------------------------------------
# Tests
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
    assert r.status_code == 200, r.text
    data = r.json()

    # Structural checks
    for key in ("schema_version", "asset_id", "asset_type", "timestamp", "metrics", "flags", "model_meta"):
        assert key in data, f"Missing {key} in response"

    assert "valuation_base_k" in data["metrics"]
    assert isinstance(data["metrics"]["valuation_base_k"], (int, float))
    assert data["flags"] == {"anomaly": False, "needs_review": False}

    # Optional: validate dataset hash format
    if "dataset_hash" in data.get("model_meta", {}):
        assert isinstance(data["model_meta"]["dataset_hash"], str)

    # JSON schema validation
    try:
        jsonschema_validate(instance=data, schema=OUTPUT_SCHEMA)
    except ValidationError as ve:
        pytest.fail(f"Schema validation failed: {ve.message}")

    assert "schema_validation_error" not in data


def test_predict_property_derive_age(payload_no_age):
    r = client.post("/predict/property", json=payload_no_age)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "valuation_base_k" in data["metrics"]
    try:
        jsonschema_validate(instance=data, schema=OUTPUT_SCHEMA)
    except ValidationError as ve:
        pytest.fail(f"Schema validation failed (derive age): {ve.message}")


def test_predict_validation_error(sample_payload):
    bad = {**sample_payload, "floor": sample_payload["building_floors"]}  # floor >= building_floors → invalid
    r = client.post("/predict/property", json=bad)
    assert r.status_code == 422
    assert "Invalid payload" in r.json().get("detail", "")


def test_predict_energy_class_error(sample_payload):
    bad = {**sample_payload, "energy_class": "Z"}  # Invalid class
    r = client.post("/predict/property", json=bad)
    assert r.status_code == 422
    assert "energy_class must be one of" in r.json().get("detail", "")


def test_predict_with_publish(sample_payload):
    mocked_response = {
    "asset_id": "mocked_asset_123",
    "blockchain_txid": "mocked_txid_12345",
    "asa_id": 999999
    }

    with patch("scripts.inference_api.publish_ai_prediction", return_value=mocked_response) as mock_publish:
        r = client.post("/predict/property?publish=true", json=sample_payload)
        assert r.status_code == 200, r.text
        d = r.json()

        assert "publish" in d
        pub = d["publish"]
        assert pub.get("status") == "success"
        assert isinstance(pub.get("txid"), str)
        assert pub.get("txid") == mocked_response["blockchain_txid"]

        assert "schema_version" in d
        try:
            jsonschema_validate(instance=d, schema=OUTPUT_SCHEMA)
        except ValidationError as ve:
            pytest.fail(f"Schema validation failed (mocked publish): {ve.message}")

        mock_publish.assert_called_once()