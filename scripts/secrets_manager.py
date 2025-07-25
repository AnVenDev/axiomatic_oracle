import os

from dotenv import load_dotenv

IS_CI = os.getenv("CI", "false").lower() == "true"

if IS_CI:
    ALGORAND_ADDRESS = "FAKE_ADDRESS"
    ALGORAND_MNEMONIC = "FAKE_MNEMONIC"
    ALGORAND_WALLET_ADDRESS = "FAKE_WALLET"
else:
    load_dotenv()
    ALGORAND_ADDRESS = os.getenv("ALGORAND_ADDRESS")
    ALGORAND_MNEMONIC = os.getenv("ALGORAND_MNEMONIC")
    ALGORAND_WALLET_ADDRESS = os.getenv("ALGORAND_WALLET_ADDRESS")

if not ALGORAND_WALLET_ADDRESS or not ALGORAND_MNEMONIC:
    raise ValueError("Algorand credentials not properly set.")

# Network URLs
ALGO_NETWORKS = {
    "testnet": "https://testnet-api.algonode.cloud",
    "mainnet": "https://mainnet-api.algonode.cloud",
}
ALGORAND_ADDRESS = ALGO_NETWORKS.get(ALGORAND_NETWORK)
