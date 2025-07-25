import json
import hashlib
import pytest
from unittest.mock import patch
from scripts.logger_utils import (
    log_asset_publication,
    save_publications_to_json,
    compute_file_hash,
    save_prediction_detail,
)

@pytest.fixture
def sample_data():
    return {"asset_id": "test_001", "valuation_base_k": 123.45}


def test_log_asset_publication(tmp_path, sample_data):
    log_file = tmp_path / "published_assets.json"

    with patch("scripts.logger_utils.LOG_FILE", log_file), patch("scripts.logger_utils.LOG_PATH", tmp_path):
        log_asset_publication(sample_data)

        assert log_file.exists()
        content = json.loads(log_file.read_text())
        assert content[0]["asset_id"] == "test_001"
        assert "logged_at" in content[0]


def test_save_publications_to_json(tmp_path):
    sample_data = [{"asset_id": "batch_1", "valuation_base_k": 999.0}]
    json_file = tmp_path / "published_assets.json"

    with patch("scripts.logger_utils.LOG_FILE", json_file), patch("scripts.logger_utils.LOG_PATH", tmp_path):
        save_publications_to_json(sample_data, filename=json_file)

        content = json.loads(json_file.read_text())
        assert isinstance(content, list)
        assert "logged_at" in content[0]


def test_compute_file_hash(tmp_path):
    file = tmp_path / "test.json"
    content = '{"a":1}'
    file.write_text(content)

    hash_val = compute_file_hash(file)
    assert isinstance(hash_val, str)
    assert len(hash_val) == 64
    assert hash_val == hashlib.sha256(content.encode("utf-8")).hexdigest()


def test_save_prediction_detail(tmp_path):
    sample_data = {"asset_id": "abc123", "valuation_base_k": 55.5, "flags": {}}
    expected_file = tmp_path / "abc123.json"

    with patch("scripts.logger_utils.DETAIL_DIR", tmp_path):
        hash_val = save_prediction_detail(sample_data)

        assert expected_file.exists()
        assert len(hash_val) == 64
