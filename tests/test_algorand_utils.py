# tests/test_algorand_utils.py
from __future__ import annotations

import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from scripts.algorand_utils import (
    wait_for_confirmation,
    publish_to_algorand,
    create_token_for_asset,
    AlgorandError,
)


# ---------------------------------------------------------------------
# wait_for_confirmation
# ---------------------------------------------------------------------
def test_wait_for_confirmation_success():
    mock_client = MagicMock()
    mock_client.status.return_value = {"last-round": 1}
    mock_client.pending_transaction_info.return_value = {"confirmed-round": 2}
    mock_client.status_after_block.return_value = None

    result = wait_for_confirmation(mock_client, "fake_txid")
    assert result["confirmed-round"] == 2


def test_wait_for_confirmation_timeout():
    mock_client = MagicMock()
    mock_client.status.return_value = {"last-round": 1}
    mock_client.pending_transaction_info.return_value = {}
    mock_client.status_after_block.return_value = None

    with pytest.raises(AlgorandError):
        wait_for_confirmation(mock_client, "fake_txid", timeout=1)


# ---------------------------------------------------------------------
# publish_to_algorand
# ---------------------------------------------------------------------
@patch("scripts.algorand_utils.wait_for_confirmation", return_value={"confirmed-round": 123})
@patch("scripts.algorand_utils.transaction.PaymentTxn")
@patch("scripts.algorand_utils.get_account")
@patch("scripts.algorand_utils.create_algod_client")
def test_publish_to_algorand_success(mock_client_factory, mock_get_account, mock_payment, mock_wait):
    # client factory → mock client
    mock_client = MagicMock()
    mock_client_factory.return_value = mock_client

    # suggested params (MagicMock accetta assignment di flat_fee/fee)
    mock_client.suggested_params.return_value = MagicMock()

    # account con address/pk
    mock_get_account.return_value = SimpleNamespace(address="ADDR_TEST", private_key=b"pk", mnemonic=None)

    # txn.sign → signed blob & send_transaction → txid
    signed_txn = MagicMock()
    mock_payment.return_value.sign.return_value = signed_txn
    mock_client.send_transaction.return_value = "tx123"

    txid = publish_to_algorand({"hello": "world"})
    assert txid == "tx123"

    # controlli di base sulle chiamate
    mock_payment.assert_called_once()
    mock_client.send_transaction.assert_called_once()
    mock_wait.assert_called_once()


# ---------------------------------------------------------------------
# create_token_for_asset
# ---------------------------------------------------------------------
@patch("scripts.algorand_utils.transaction.AssetConfigTxn")
@patch("scripts.algorand_utils.get_account")
@patch("scripts.algorand_utils.create_algod_client")
def test_create_token_for_asset_success(mock_client_factory, mock_get_account, mock_asset_txn):
    mock_client = MagicMock()
    mock_client_factory.return_value = mock_client
    mock_client.suggested_params.return_value = MagicMock()
    mock_client.pending_transaction_info.return_value = {"asset-index": 999}
    mock_client.send_transaction.return_value = "tx456"

    mock_get_account.return_value = SimpleNamespace(address="ADDR_TEST", private_key=b"pk", mnemonic=None)

    # Evita real wait: facciamo sì che la funzione interna non sollevi
    with patch("scripts.algorand_utils.wait_for_confirmation", return_value={"confirmed-round": 1}):
        asset_id = create_token_for_asset("Prop", "PRP")
        assert asset_id == 999

    # ha creato e firmato la transazione
    assert mock_asset_txn.called
    assert mock_client.send_transaction.called


@patch("scripts.algorand_utils.transaction.AssetConfigTxn")
@patch("scripts.algorand_utils.get_account")
@patch("scripts.algorand_utils.create_algod_client")
def test_create_token_for_asset_error(mock_client_factory, mock_get_account, mock_asset_txn):
    mock_client = MagicMock()
    mock_client_factory.return_value = mock_client
    mock_client.suggested_params.return_value = MagicMock()
    mock_client.send_transaction.side_effect = Exception("fail")

    mock_get_account.return_value = SimpleNamespace(address="ADDR_TEST", private_key=b"pk", mnemonic=None)

    with patch("scripts.algorand_utils.wait_for_confirmation", return_value={"confirmed-round": 1}):
        with pytest.raises(AlgorandError):
            create_token_for_asset("Prop", "PRP")