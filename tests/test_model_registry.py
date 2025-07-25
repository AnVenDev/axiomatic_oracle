import pytest
import joblib
from pathlib import Path
from unittest.mock import patch, mock_open
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import scripts.model_registry as model_registry

@pytest.fixture
def fake_model_path():
    return Path("models/property/value_regressor_v1.joblib")


def test_get_pipeline_with_real_model(tmp_path, monkeypatch):
    # Crea una pipeline reale
    real_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    model_file = tmp_path / "value_regressor_v1.joblib"
    joblib.dump(real_pipeline, model_file)

    # Patch registri e path
    monkeypatch.setitem(model_registry.MODEL_REGISTRY, "property", {
        "value_regressor": model_file.name
    })
    monkeypatch.setattr(model_registry, "MODELS_BASE", tmp_path)

    result = model_registry.get_pipeline("property", "value_regressor")

    assert isinstance(result, Pipeline)
    assert isinstance(result.named_steps["model"], LinearRegression)


@patch("builtins.open", new_callable=mock_open, read_data='{"model_version": "v1", "metrics": {"r2": 0.95}}')
@patch("scripts.model_registry._metadata_path_for", return_value=Path("models/property/value_regressor_v1_meta.json"))
@patch("scripts.model_registry.Path.exists", return_value=True)
def test_get_model_metadata(mock_exists, mock_meta_path, mock_file):
    meta = model_registry.get_model_metadata("property", "value_regressor")
    assert meta is not None
    assert meta["model_version"] == "v1"
    assert "model_hash" in meta


def test_list_asset_types():
    types = model_registry.list_asset_types()
    assert isinstance(types, list)
    assert "property" in types


def test_list_tasks():
    tasks = model_registry.list_tasks("property")
    assert isinstance(tasks, list)
    assert "value_regressor" in tasks


def test_list_tasks_invalid():
    with pytest.raises(model_registry.RegistryLookupError):
        model_registry.list_tasks("unknown_asset")


def test_model_exists_true():
    with patch("scripts.model_registry._resolve_path", return_value=Path("models/property/value_regressor_v1.joblib")):
        assert model_registry.model_exists("property", "value_regressor")


def test_model_exists_false():
    with patch("scripts.model_registry._resolve_path", side_effect=model_registry.ModelNotFoundError()):
        assert not model_registry.model_exists("property", "unknown_model")


@patch("scripts.model_registry.get_last_modified", return_value="Today")
@patch("scripts.model_registry.get_model_size", return_value=1.5)
@patch("scripts.model_registry.get_model_metadata", return_value={"metrics": {"r2": 0.85}})
@patch("scripts.model_registry._resolve_path", return_value=Path("models/property/value_regressor_v1.joblib"))
@patch("scripts.model_registry.get_pipeline", return_value=True)
def test_health_check_model(mock_pipe, mock_resolve, mock_meta, mock_size, mock_mod):
    result = model_registry.health_check_model("property", "value_regressor")
    assert result["status"] == "healthy"
    assert "model_path" in result
    assert "size_mb" in result