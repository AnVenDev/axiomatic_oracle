import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Load secrets
ALGORAND_WALLET_ADDRESS = os.getenv("ALGORAND_WALLET_ADDRESS")
ALGORAND_MNEMONIC = os.getenv("ALGORAND_MNEMONIC")
ALGORAND_NETWORK = os.getenv("ALGORAND_NETWORK", "testnet")

if not ALGORAND_WALLET_ADDRESS or not ALGORAND_MNEMONIC:
    raise ValueError("Algorand credentials not properly set.")

# Network URLs
ALGO_NETWORKS = {
    "testnet": "https://testnet-api.algonode.cloud",
    "mainnet": "https://mainnet-api.algonode.cloud"
}
ALGORAND_ADDRESS = ALGO_NETWORKS.get(ALGORAND_NETWORK)