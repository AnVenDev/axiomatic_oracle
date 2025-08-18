# tests/test_logger_utils.py
from __future__ import annotations

import importlib
import json
import os
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest   # type: ignore

def _reload_logger_utils(tmp_dir: Path):
    # Indica al modulo dove scrivere
    with patch.dict(os.environ, {"AI_ORACLE_LOG_DIR": str(tmp_dir)}, clear=False):
        import scripts.logger_utils as lu
        importlib.reload(lu)
        return lu

@pytest.fixture
def sample_data():
    # le funzioni non richiedono un payload specifico
    return {"asset_id": "test_001", "note": "ok"}

def test_log_asset_publication(tmp_path: Path, sample_data):
    lu = _reload_logger_utils(tmp_path)
    # abilita log anche se l’utente ha AI_ORACLE_DISABLE_API_LOG=1 nell’ambiente
    with patch.dict(os.environ, {"AI_ORACLE_DISABLE_API_LOG": "0"}, clear=False):
        lu.log_asset_publication(sample_data)

    # Il modulo scrive in <AI_ORACLE_LOG_DIR>/published_assets.json
    log_file = tmp_path / "published_assets.json"
    assert log_file.exists(), "Log file not created"
    content = json.loads(log_file.read_text(encoding="utf-8"))
    assert content and content[0]["asset_id"] == "test_001"
    assert "logged_at" in content[0]

def test_save_publications_to_json(tmp_path: Path):
    lu = _reload_logger_utils(tmp_path)
    sample_batch = [{"asset_id": "batch_1"}]
    with patch.dict(os.environ, {"AI_ORACLE_DISABLE_API_LOG": "0"}, clear=False):
        lu.save_publications_to_json(sample_batch, filename=tmp_path / "published_assets.json")

    content = json.loads((tmp_path / "published_assets.json").read_text(encoding="utf-8"))
    assert isinstance(content, list)
    assert "logged_at" in content[0]

def test_compute_file_hash(tmp_path: Path):
    lu = _reload_logger_utils(tmp_path)
    file = tmp_path / "test.json"
    content = '{"a":1}'
    file.write_text(content, encoding="utf-8")

    hash_val = lu.compute_file_hash(file)
    assert isinstance(hash_val, str)
    assert len(hash_val) == 64
    assert hash_val == hashlib.sha256(content.encode("utf-8")).hexdigest()

def test_save_prediction_detail(tmp_path: Path):
    lu = _reload_logger_utils(tmp_path)
    sample_pred = {"asset_id": "abc123", "metrics": {"valuation_k": 55.5}, "flags": {}}
    with patch.dict(os.environ, {"AI_ORACLE_DISABLE_API_LOG": "0"}, clear=False):
        hash_val = lu.save_prediction_detail(sample_pred)

    expected_file = tmp_path / "detail_reports" / "abc123.json"
    assert expected_file.exists()
    assert isinstance(hash_val, str) and len(hash_val) == 64
