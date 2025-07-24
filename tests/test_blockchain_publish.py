import sys
import pathlib
import pytest
from unittest.mock import patch

# Aggiunge la root del progetto a sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.blockchain_publisher import publish_ai_prediction, batch_publish_predictions

# Mocked sample inputs
sample_response = {
    "asset_id": "property_1234",
    "asset_type": "property",
    "timestamp": "2025-07-23T15:00:00Z",
    "metrics": {"valuation_base_k": 4.8},
    "flags": {"anomaly": False, "needs_review": False},
    "model_meta": {
        "value_model_version": "v1",
        "value_model_name": "MockModel",
        "model_hash": "abcd1234abcd1234abcd1234abcd1234"
    },
    "schema_version": "v1"
}

multiple_samples = [sample_response, {**sample_response, "asset_id": "property_5678"}]


@patch("scripts.blockchain_publisher.publish_to_algorand", return_value="mocked_txid")
@patch("scripts.blockchain_publisher.create_token_for_asset", return_value=987654)
@patch("scripts.blockchain_publisher.log_asset_publication")
def test_publish_ai_prediction(mock_log, mock_token, mock_publish):
    result = publish_ai_prediction(sample_response)

    assert isinstance(result, dict)
    assert result["asset_id"] == "property_1234"
    assert result["blockchain_txid"] == "mocked_txid"
    assert result["asa_id"] == 987654

    mock_publish.assert_called_once()
    mock_token.assert_called_once()
    mock_log.assert_called_once()


@patch("scripts.blockchain_publisher.publish_to_algorand", return_value="mocked_txid")
@patch("scripts.blockchain_publisher.create_token_for_asset", return_value=123456)
@patch("scripts.blockchain_publisher.log_asset_publication")
@patch("scripts.blockchain_publisher.save_publications_to_json")
def test_batch_publish_predictions(mock_save, mock_log, mock_token, mock_publish):
    results = batch_publish_predictions(multiple_samples)

    assert isinstance(results, list)
    assert len(results) == 2
    for res in results:
        assert "asset_id" in res
        assert "blockchain_txid" in res
        assert "asa_id" in res

    assert mock_publish.call_count == 2
    assert mock_token.call_count == 2
    assert mock_log.call_count == 2
    mock_save.assert_called_once_with(results)
