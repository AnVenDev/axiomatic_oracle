"""
model_registry.py
Centralized registry & loader for AI Oracle model pipelines (multi-RWA ready).

Responsibilities:
- Map (asset_type, task) -> local filesystem path
- Lazy-load & cache pipelines (joblib)
- Provide metadata loading helper (if available)
- Offer utility to list available tasks and versions
- Prepare for future remote / versioned storage (e.g. S3, IPFS)

Usage:
    from scripts.model_registry import get_pipeline, get_model_metadata

    pipe = get_pipeline("property", "value_regressor")
    meta = get_model_metadata("property", "value_regressor")

Conventions:
- Each asset_type gets its own subfolder inside /models
- File naming: <task>_<version>.joblib
- Metadata file: <task>_<version>_meta.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, List

import joblib

# Base directory for models (can be overridden via env)
MODELS_BASE = Path(__file__).resolve().parent.parent / "models"

# Registry definition:
# Each asset_type maps to tasks with relative paths
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "property": {
        "value_regressor": "property/value_regressor_v1.joblib",
        # Future example:
        # "anomaly_model": "property/anomaly_iforest_v0.joblib"
    },
    # "art": {
    #     "valuation_model": "art/valuation_regressor_v1.joblib",
    #     "authenticity_model": "art/auth_classifier_v1.joblib"
    # },
}

# In-memory cache to avoid re-loading
_PIPELINE_CACHE: Dict[Path, object] = {}
_METADATA_CACHE: Dict[Path, dict] = {}


class ModelNotFoundError(Exception):
    """Raised when a model path does not exist on disk."""
    pass


class RegistryLookupError(Exception):
    """Raised when (asset_type, task) pair is not defined in MODEL_REGISTRY."""
    pass


def _resolve_path(asset_type: str, task: str) -> Path:
    try:
        rel_path = MODEL_REGISTRY[asset_type][task]
    except KeyError:
        raise RegistryLookupError(
            f"Unknown asset_type/task combination: '{asset_type}' / '{task}'. "
            f"Defined asset_types: {list(MODEL_REGISTRY.keys())}"
        )
    full_path = MODELS_BASE / rel_path
    if not full_path.exists():
        raise ModelNotFoundError(f"Model file not found at: {full_path}")
    return full_path


def _metadata_path_for(model_path: Path) -> Path:
    """
    Given: property/value_regressor_v1.joblib
    Expect metadata: property/value_regressor_v1_meta.json
    """
    stem = model_path.name.replace(".joblib", "")
    meta_name = f"{stem}_meta.json"
    return model_path.parent / meta_name


def get_pipeline(asset_type: str, task: str = "value_regressor"):
    """
    Return a loaded (and cached) model pipeline.

    :param asset_type: e.g. "property"
    :param task: e.g. "value_regressor"
    :return: scikit-learn Pipeline object
    """
    model_path = _resolve_path(asset_type, task)

    if model_path not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[model_path] = joblib.load(model_path)

    return _PIPELINE_CACHE[model_path]


def get_model_metadata(asset_type: str, task: str = "value_regressor") -> Optional[dict]:
    """
    Return metadata dict if side-car JSON exists, else None.
    """
    model_path = _resolve_path(asset_type, task)
    meta_path = _metadata_path_for(model_path)

    if not meta_path.exists():
        return None

    if meta_path not in _METADATA_CACHE:
        with meta_path.open("r", encoding="utf-8") as f:
            _METADATA_CACHE[meta_path] = json.load(f)

    return _METADATA_CACHE[meta_path]


def list_asset_types() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def list_tasks(asset_type: str) -> List[str]:
    if asset_type not in MODEL_REGISTRY:
        raise RegistryLookupError(f"Unknown asset_type: {asset_type}")
    return list(MODEL_REGISTRY[asset_type].keys())


def model_exists(asset_type: str, task: str = "value_regressor") -> bool:
    try:
        _resolve_path(asset_type, task)
        return True
    except (RegistryLookupError, ModelNotFoundError):
        return False


def refresh_cache(asset_type: Optional[str] = None, task: Optional[str] = None) -> None:
    """
    Clear cached entries (all or selected) so next call reloads from disk.
    """
    if asset_type and task:
        try:
            path = _resolve_path(asset_type, task)
            _PIPELINE_CACHE.pop(path, None)
            meta_path = _metadata_path_for(path)
            _METADATA_CACHE.pop(meta_path, None)
        except (RegistryLookupError, ModelNotFoundError):
            pass
    else:
        _PIPELINE_CACHE.clear()
        _METADATA_CACHE.clear()


if __name__ == "__main__":
    # Simple diagnostics when running directly
    print("Available asset types:", list_asset_types())
    for at in list_asset_types():
        print(f"  {at}: tasks -> {list_tasks(at)}")
        for t in list_tasks(at):
            print(f"    - {t}: exists on disk? {model_exists(at, t)}")
            if model_exists(at, t):
                meta = get_model_metadata(at, t)
                if meta:
                    print(f"      version: {meta.get('model_version')} | R²: {meta.get('metrics', {}).get('r2')}")
