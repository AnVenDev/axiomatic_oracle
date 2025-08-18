# tests/test_blockchain_publish.py
import pathlib
import sys
from unittest.mock import patch

from scripts.blockchain_publisher import (
    batch_publish_predictions,
    publish_ai_prediction,
)

# Aggiunge la root del progetto a sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Mocked sample inputs (schema v1: il publisher supporta v1 e v2)
sample_response = {
    "asset_id": "property_1234",
    "asset_type": "property",
    "timestamp": "2025-07-23T15:00:00Z",
    "metrics": {"valuation_base_k": 4.8},
    "flags": {"anomaly": False, "needs_review": False},
    "model_meta": {
        "value_model_version": "v1",
        "value_model_name": "MockModel",
        "model_hash": "abcd1234abcd1234abcd1234abcd1234",
    },
    "schema_version": "v1",
}

multiple_samples = [
    sample_response,
    {**sample_response, "asset_id": "property_5678"},
]


@patch("scripts.blockchain_publisher.save_prediction_detail", return_value="deadbeef" * 8)
@patch("scripts.blockchain_publisher.publish_to_algorand", return_value="mocked_txid")
@patch("scripts.blockchain_publisher.create_token_for_asset", return_value=987654)
@patch("scripts.blockchain_publisher.log_asset_publication")
def test_publish_ai_prediction(mock_log, mock_token, mock_publish, mock_save_detail):
    result = publish_ai_prediction(sample_response)

    assert isinstance(result, dict)
    assert result["asset_id"] == "property_1234"
    assert result["blockchain_txid"] == "mocked_txid"
    assert result["asa_id"] == 987654

    mock_save_detail.assert_called_once()       # niente I/O reale
    mock_publish.assert_called_once()           # notarizzazione on-chain
    mock_token.assert_called_once()             # ASA creato
    mock_log.assert_called_once()               # log pubblicazione


@patch("scripts.blockchain_publisher.save_prediction_detail", return_value="deadbeef" * 8)
@patch("scripts.blockchain_publisher.publish_to_algorand", return_value="mocked_txid")
@patch("scripts.blockchain_publisher.create_token_for_asset", return_value=123456)
@patch("scripts.blockchain_publisher.log_asset_publication")
@patch("scripts.blockchain_publisher.save_publications_to_json")
def test_batch_publish_predictions(mock_save_json, mock_log, mock_token, mock_publish, mock_save_detail):
    results = batch_publish_predictions(multiple_samples)

    assert isinstance(results, list)
    assert len(results) == 2
    for res in results:
        assert "asset_id" in res
        assert "blockchain_txid" in res
        assert "asa_id" in res

    # due asset → due call per publish/token/log/detail
    assert mock_publish.call_count == 2
    assert mock_token.call_count == 2
    assert mock_log.call_count == 2
    assert mock_save_detail.call_count == 2

    # salva il JSON array finale con tutti i risultati
    mock_save_json.assert_called_once_with(results)