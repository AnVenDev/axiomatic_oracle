import importlib
import sys
from unittest.mock import patch
import pytest

@pytest.fixture(autouse=True)
def clear_secrets_module():
    """Assicura che il modulo venga ricaricato da zero a ogni test."""
    if "scripts.secrets_manager" in sys.modules:
        del sys.modules["scripts.secrets_manager"]


@patch.dict("os.environ", {"CI": "true"})
def test_ci_environment():
    import scripts.secrets_manager as sm
    importlib.reload(sm)

    assert sm.ALGORAND_WALLET_ADDRESS == "FAKE_WALLET"
    assert sm.ALGORAND_MNEMONIC.startswith("abandon abandon")
    assert sm.ALGORAND_NETWORK == "testnet"
    assert sm.ALGORAND_ADDRESS == "https://testnet-api.algonode.cloud"


@patch.dict(
    "os.environ",
    {
        "CI": "false",
        "ALGORAND_WALLET_ADDRESS": "TEST_WALLET",
        "ALGORAND_MNEMONIC": "apple " * 24 + "zebra",
        "ALGORAND_NETWORK": "mainnet",
    },
)
def test_local_env():
    import scripts.secrets_manager as sm
    importlib.reload(sm)

    assert sm.ALGORAND_WALLET_ADDRESS == "TEST_WALLET"
    assert sm.ALGORAND_MNEMONIC.startswith("apple apple")
    assert sm.ALGORAND_NETWORK == "mainnet"
    assert sm.ALGORAND_ADDRESS == "https://mainnet-api.algonode.cloud"
