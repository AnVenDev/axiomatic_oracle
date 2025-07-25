import pytest
from unittest.mock import patch, MagicMock

from scripts.algorand_utils import (
    wait_for_confirmation,
    publish_to_algorand,
    create_token_for_asset,
    AlgorandError,
)

# --- wait_for_confirmation ---
@patch("scripts.algorand_utils.client")
def test_wait_for_confirmation_success(mock_client):
    mock_client.status.return_value = {"last-round": 1}
    mock_client.pending_transaction_info.return_value = {"confirmed-round": 2}
    mock_client.status_after_block.return_value = None

    result = wait_for_confirmation("fake_txid")
    assert result["confirmed-round"] == 2


@patch("scripts.algorand_utils.client")
def test_wait_for_confirmation_timeout(mock_client):
    mock_client.status.return_value = {"last-round": 1}
    mock_client.pending_transaction_info.return_value = {}
    mock_client.status_after_block.return_value = None

    with pytest.raises(AlgorandError):
        wait_for_confirmation("fake_txid", timeout=1)


# --- publish_to_algorand ---
@patch("scripts.algorand_utils.client")
@patch("scripts.algorand_utils.wait_for_confirmation")
@patch("scripts.algorand_utils.transaction.PaymentTxn")
def test_publish_to_algorand_success(mock_txn, mock_wait, mock_client):
    signed_txn = MagicMock()
    mock_txn.return_value.sign.return_value = signed_txn
    mock_client.send_transaction.return_value = "tx123"

    txid = publish_to_algorand({"hello": "world"})
    assert txid == "tx123"


# --- create_token_for_asset ---
@patch("scripts.algorand_utils.client")
@patch("scripts.algorand_utils.wait_for_confirmation")
@patch("scripts.algorand_utils.transaction.AssetConfigTxn")
def test_create_token_for_asset_success(mock_txn, mock_wait, mock_client):
    signed_txn = MagicMock()
    mock_txn.return_value.sign.return_value = signed_txn
    mock_client.send_transaction.return_value = "tx456"
    mock_client.pending_transaction_info.return_value = {"asset-index": 999}

    asset_id = create_token_for_asset("Prop", "PRP")
    assert asset_id == 999


@patch("scripts.algorand_utils.client")
@patch("scripts.algorand_utils.wait_for_confirmation")
@patch("scripts.algorand_utils.transaction.AssetConfigTxn")
def test_create_token_for_asset_error(mock_txn, mock_wait, mock_client):
    signed_txn = MagicMock()
    mock_txn.return_value.sign.return_value = signed_txn
    mock_client.send_transaction.side_effect = Exception("fail")

    with pytest.raises(AlgorandError):
        create_token_for_asset("Prop", "PRP")