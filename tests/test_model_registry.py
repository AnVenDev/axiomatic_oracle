# tests/test_model_registry.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np                                      # type: ignore
import joblib                                           # type: ignore
import pytest                                           # type: ignore
from sklearn.pipeline import Pipeline                   # type: ignore
from sklearn.preprocessing import StandardScaler        # type: ignore
from sklearn.linear_model import LinearRegression       # type: ignore

import scripts.model_registry as model_registry


# -----------------------------------------------------------------------------
# Helpers per creare un modello FITTED + meta + manifest nel tmp path
# -----------------------------------------------------------------------------
def _make_fitted_pipeline() -> Pipeline:
    # Pipeline semplicissima ma FITTED (requisito del registry)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pipe.fit(X, y)
    return pipe


def _write_artifacts(tmp_path: Path) -> dict:
    """
    Crea:
      tmp_path/property/value_regressor_v1.joblib
      tmp_path/property/value_regressor_v1_meta.json
      tmp_path/property/training_manifest.json
    """
    at_dir = tmp_path / "property"
    at_dir.mkdir(parents=True, exist_ok=True)

    # Modello fitted
    model_path = at_dir / "value_regressor_v1.joblib"
    joblib.dump(_make_fitted_pipeline(), model_path)

    # Meta sidecar
    meta = {
        "model_version": "v1",
        "model_class": "LinearRegression",
        "features_categorical": ["energy_class"],
        "features_numeric": ["size_m2", "rooms"],
        "metrics": {"rmse": 1.2, "mae": 0.9, "r2": 0.8},
        "model_path": str(model_path),
    }
    meta_path = at_dir / "value_regressor_v1_meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    # Manifest di training (preferito per feature/metrics)
    manifest = {
        "model": {
            "feature_list": {
                "categorical": ["energy_class"],
                "numeric": ["size_m2", "rooms"],
            }
        },
        "metrics": {"validation": {"rmse": 1.3, "mae": 1.0, "r2": 0.79}},
    }
    manifest_path = at_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    return {
        "models_base": tmp_path,
        "asset_dir": at_dir,
        "model_path": model_path,
        "meta_path": meta_path,
        "manifest_path": manifest_path,
    }


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def tmp_models(monkeypatch, tmp_path):
    # Prepara artefatti e forza il registry a usare questa radice
    art = _write_artifacts(tmp_path)
    monkeypatch.setattr(model_registry, "MODELS_BASE", art["models_base"])
    return art


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_get_pipeline_with_real_model(tmp_models):
    pipe = model_registry.get_pipeline("property", "value_regressor")
    assert isinstance(pipe, Pipeline)
    # Dev'essere fitted (il registry carica solo fitted)
    # (LinearRegression fitted -> ha coef_):
    assert hasattr(pipe.named_steps["model"], "coef_")


def test_get_model_metadata(tmp_models):
    meta = model_registry.get_model_metadata("property", "value_regressor")
    assert isinstance(meta, dict)
    assert meta.get("model_version") == "v1"
    assert "model_hash" in meta  # arricchito
    assert "model_size_mb" in meta
    assert "last_modified" in meta
    # metriche possono venire dal manifest
    assert "metrics" in meta


def test_list_asset_types(tmp_models):
    types = model_registry.list_asset_types()
    assert isinstance(types, list)
    assert "property" in types


def test_list_tasks(tmp_models):
    tasks = model_registry.list_tasks("property")
    assert isinstance(tasks, list)
    assert "value_regressor" in tasks


def test_list_tasks_invalid(tmp_models):
    with pytest.raises(model_registry.RegistryLookupError):
        model_registry.list_tasks("unknown_asset")


def test_model_exists_true(tmp_models):
    assert model_registry.model_exists("property", "value_regressor")


def test_model_exists_false(tmp_models):
    assert not model_registry.model_exists("property", "unknown_model")


def test_suggest_similar_models(tmp_models):
    # Usa il MODEL_REGISTRY interno per suggerimenti sul nome task
    result = model_registry._suggest_similar_models("property", "value_regresor")
    assert "value_regressor" in result


def test_resolve_path_invalid_key(tmp_models):
    # Asset inesistente → nessun discovery → ModelNotFoundError
    with pytest.raises(model_registry.ModelNotFoundError):
        model_registry._resolve_path("invalid_type", "value_regressor")


def test_resolve_path_discovery(tmp_path, monkeypatch):
    """
    Verifica che _resolve_path trovi via discovery un modello fitted anche
    se non è presente in MODEL_REGISTRY.
    """
    at_dir = tmp_path / "property"
    at_dir.mkdir()
    # crea un modello fitted reale
    model_file = at_dir / "value_regressor_v1.joblib"
    joblib.dump(_make_fitted_pipeline(), model_file)
    (at_dir / "value_regressor_v1_meta.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(model_registry, "MODELS_BASE", tmp_path)
    p = model_registry._resolve_path("property", "value_regressor")
    assert p.exists()
    assert p.name == "value_regressor_v1.joblib"


def test_file_sha256(tmp_path):
    file = tmp_path / "model.joblib"
    file.write_text("test content", encoding="utf-8")
    h = model_registry._file_sha256(file)
    assert isinstance(h, str) and len(h) == 64


def test_refresh_cache_invalid(tmp_models):
    # Non deve sollevare
    model_registry.refresh_cache("invalid", "task")


def test_cache_stats_format(tmp_models):
    _ = model_registry.get_pipeline("property", "value_regressor")  # warm cache
    stats = model_registry.cache_stats()
    assert "pipelines_cached" in stats
    assert "models" in stats


def test_discover_and_parse(tmp_path, monkeypatch):
    model_dir = tmp_path / "property"
    model_dir.mkdir()
    model_file = model_dir / "value_regressor_v1.joblib"
    joblib.dump(_make_fitted_pipeline(), model_file)
    monkeypatch.setattr(model_registry, "MODELS_BASE", tmp_path)

    models = model_registry.discover_models_for_asset("property")
    assert any(p.name == "value_regressor_v1.joblib" for p in models)

    parsed = model_registry.parse_task_and_version(model_file.name)
    assert parsed == ("value_regressor", "v1")


def test_suggest_and_latest_version(tmp_path, monkeypatch):
    model_dir = tmp_path / "property"
    model_dir.mkdir()
    joblib.dump(_make_fitted_pipeline(), model_dir / "value_regressor_v1.joblib")
    joblib.dump(_make_fitted_pipeline(), model_dir / "value_regressor_v2.joblib")
    monkeypatch.setattr(model_registry, "MODELS_BASE", tmp_path)

    versions = model_registry.suggest_task_versions("property", "value_regressor")
    assert "value_regressor_v2.joblib" in versions

    latest = model_registry.latest_version_filename("property", "value_regressor")
    assert latest == "value_regressor_v2.joblib"


@pytest.mark.asyncio
async def test_remote_model_registry_methods():
    if not hasattr(model_registry, "RemoteModelRegistry"):
        pytest.skip("RemoteModelRegistry not implemented in this build")
    registry = model_registry.RemoteModelRegistry()
    with pytest.raises(NotImplementedError):
        await registry.download_model("property", "value_regressor", "v1")
    available = await registry.check_remote_availability("property", "value_regressor", "v1")
    assert available is False



def test_health_check_model(tmp_models):
    result = model_registry.health_check_model("property", "value_regressor")
    assert result["status"] == "healthy"
    assert "model_path" in result
    assert "size_mb" in result
    assert result["fitted"] is True


def test_expected_features_from_manifest_or_meta(tmp_models):
    paths = model_registry.get_model_paths("property", "value_regressor")
    meta = model_registry.get_model_metadata("property", "value_regressor") or {}
    cat, num = model_registry.expected_features(meta, paths["manifest"], asset_type="property")
    assert isinstance(cat, list) and isinstance(num, list)
    assert len(cat) + len(num) > 0
    # no overlap
    assert not (set(cat) & set(num))