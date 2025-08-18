# tests/test_secrets_manager.py
from __future__ import annotations

import importlib
import sys
from unittest.mock import patch
import pytest  # type: ignore

ZERO_ADDR = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAY5HFKQ"

@pytest.fixture(autouse=True)
def clear_secrets_module():
    if "scripts.secrets_manager" in sys.modules:
        del sys.modules["scripts.secrets_manager"]

@patch.dict("os.environ", {"CI": "true"}, clear=True)
def test_ci_environment():
    with patch("scripts.secrets_manager.load_dotenv", lambda *a, **k: None):
        import scripts.secrets_manager as sm
        importlib.reload(sm)

    assert sm.get_network() == "testnet"
    algod = sm.get_algod_config(); idx = sm.get_indexer_config()
    assert algod.network == "testnet" and "testnet" in (algod.algod_url or "")
    assert idx.network == "testnet" and "testnet" in (idx.indexer_url or "")

    acc = sm.get_account(require_signing=False)
    assert acc.address == ""
    assert acc.mnemonic is None and acc.private_key is None

    with pytest.raises(ValueError):
        sm.get_account(require_signing=True)

@patch.dict(
    "os.environ",
    {
        "CI": "false",
        "ALGORAND_NETWORK": "mainnet",
        "ALGORAND_WALLET_ADDRESS": ZERO_ADDR,
        "ALGORAND_MNEMONIC": "",
        "ALGORAND_PRIVATE_KEY": "",
    },
    clear=True,
)
def test_local_env_mainnet_readonly():
    with patch("scripts.secrets_manager.load_dotenv", lambda *a, **k: None):
        import scripts.secrets_manager as sm
        importlib.reload(sm)

    assert sm.get_network() == "mainnet"
    algod = sm.get_algod_config(); idx = sm.get_indexer_config()
    assert "mainnet" in (algod.algod_url or "") and "mainnet" in (idx.indexer_url or "")
    acc = sm.get_account(require_signing=False)
    assert acc.address == ZERO_ADDR
    assert acc.private_key is None and acc.mnemonic is None

@patch.dict(
    "os.environ",
    {
        "CI": "false",
        "ALGORAND_NETWORK": "custom",
        "ALGORAND_ALGOD_URL": "http://localhost:4001",
        "ALGORAND_ALGOD_TOKEN": "ALGOD_TOKEN_ABC",
        "ALGORAND_INDEXER_URL": "http://localhost:8980",
        "ALGORAND_INDEXER_TOKEN": "INDEXER_TOKEN_XYZ",
    },
    clear=True,
)
def test_custom_urls_and_tokens():
    with patch("scripts.secrets_manager.load_dotenv", lambda *a, **k: None):
        import scripts.secrets_manager as sm
        importlib.reload(sm)

    algod = sm.get_algod_config(); idx = sm.get_indexer_config()
    assert algod.algod_url == "http://localhost:4001"
    assert idx.indexer_url == "http://localhost:8980"
    assert algod.headers.get("X-Algo-API-Token") == "ALGOD_TOKEN_ABC"
    assert idx.headers.get("X-Algo-API-Token") == "INDEXER_TOKEN_XYZ"

@patch.dict(
    "os.environ",
    {
        "CI": "false",
        "ALGORAND_NETWORK": "testnet",
        "ALGORAND_WALLET_ADDRESS": ZERO_ADDR,
        "ALGORAND_MNEMONIC": "",
        "ALGORAND_PRIVATE_KEY": "",
    },
    clear=True,
)
def test_safe_config_summary_redaction():
    with patch("scripts.secrets_manager.load_dotenv", lambda *a, **k: None):
        import scripts.secrets_manager as sm
        importlib.reload(sm)

    summary = sm.get_safe_config_summary()
    assert isinstance(summary["wallet_address"], str)
    assert summary["wallet_address"] != ZERO_ADDR
    assert ("…" in summary["wallet_address"]) or ("..." in summary["wallet_address"]) or ("*" in summary["wallet_address"])
    assert summary["has_mnemonic"] in {"True", "False"}
    assert summary["has_private_key"] in {"True", "False"}