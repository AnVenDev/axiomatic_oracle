import os
from dotenv import load_dotenv

IS_CI = os.getenv("CI", "false").lower() == "true"

# Network URLs
ALGO_NETWORKS = {
    "testnet": "https://testnet-api.algonode.cloud",
    "mainnet": "https://mainnet-api.algonode.cloud",
}

if IS_CI:
    ALGORAND_WALLET_ADDRESS = "FAKE_WALLET"
    ALGORAND_MNEMONIC = "FAKE_MNEMONIC"
    ALGORAND_NETWORK = "testnet"  # valore valido per evitare errori
else:
    load_dotenv()
    ALGORAND_WALLET_ADDRESS = os.getenv("ALGORAND_WALLET_ADDRESS")
    ALGORAND_MNEMONIC = os.getenv("ALGORAND_MNEMONIC")
    ALGORAND_NETWORK = os.getenv("ALGORAND_NETWORK", "testnet")

if not ALGORAND_WALLET_ADDRESS or not ALGORAND_MNEMONIC:
    raise ValueError("Algorand credentials not properly set.")

# Valid ALGORAND_ADDRESS based on network
ALGORAND_ADDRESS = ALGO_NETWORKS.get(ALGORAND_NETWORK)