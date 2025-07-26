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

def test_suggest_similar_models():
    result = model_registry._suggest_similar_models("property", "value_regresor")
    assert "value_regressor" in result

def test_resolve_path_invalid_key():
    with pytest.raises(model_registry.RegistryLookupError):
        model_registry._resolve_path("invalid_type", "value_regressor")

@patch("scripts.model_registry.latest_version_filename", return_value="value_regressor_v1.joblib")
def test_resolve_path_with_fallback(mock_latest, tmp_path, monkeypatch):
    path = tmp_path / "property"
    path.mkdir()
    fallback_model = path / "value_regressor_v1.joblib"
    fallback_model.write_text("fake model")
    
    monkeypatch.setattr(model_registry, "MODELS_BASE", tmp_path)
    monkeypatch.setitem(model_registry.MODEL_REGISTRY, "property", {
        "value_regressor": "non_existing_path.joblib"
    })
    
    resolved = model_registry._resolve_path("property", "value_regressor", fallback_latest=True)
    assert resolved.exists()

def test_file_hash_sha256(tmp_path):
    file = tmp_path / "model.joblib"
    file.write_text("test content")
    h = model_registry._file_hash_sha256(file)
    assert isinstance(h, str)
    assert len(h) == 64

def test_refresh_cache_invalid():
    # Should not raise
    model_registry.refresh_cache("invalid", "task")

def test_cache_stats_format():
    stats = model_registry.cache_stats()
    assert "pipelines_cached" in stats
    assert "models" in stats

def test_discover_and_parse(tmp_path, monkeypatch):
    model_dir = tmp_path / "property"
    model_dir.mkdir()
    model_file = model_dir / "value_regressor_v1.joblib"
    model_file.write_text("data")
    monkeypatch.setattr(model_registry, "MODELS_BASE", tmp_path)

    models = model_registry.discover_models_for_asset("property")
    assert len(models) == 1

    parsed = model_registry.parse_task_and_version(model_file.name)
    assert parsed == ("value_regressor", "v1")

def test_suggest_and_latest_version(tmp_path, monkeypatch):
    model_dir = tmp_path / "property"
    model_dir.mkdir()
    (model_dir / "value_regressor_v1.joblib").write_text("x")
    (model_dir / "value_regressor_v2.joblib").write_text("y")
    monkeypatch.setattr(model_registry, "MODELS_BASE", tmp_path)

    versions = model_registry.suggest_task_versions("property", "value_regressor")
    assert "value_regressor_v2.joblib" in versions

    latest = model_registry.latest_version_filename("property", "value_regressor")
    assert latest == "value_regressor_v2.joblib"

@pytest.mark.asyncio
async def test_remote_model_registry_methods():
    registry = model_registry.RemoteModelRegistry()
    with pytest.raises(NotImplementedError):
        await registry.download_model("property", "value_regressor", "v1")
    available = await registry.check_remote_availability("property", "value_regressor", "v1")
    assert available is False

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