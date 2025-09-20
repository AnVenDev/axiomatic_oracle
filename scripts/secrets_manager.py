# scripts/secrets_manager.py
from __future__ import annotations

"""
Secrets & configuration loader for Algorand clients and account.

Public API
- get_network() -> str
- get_algod_config() -> AlgodConfig
- get_indexer_config() -> IndexerConfig
- get_account(require_signing=False) -> Account
- get_safe_config_summary() -> Dict[str, str]

Notes
- Uses Algonode endpoints by default; supports Sandbox/custom via env.
- Never logs secrets; `get_safe_config_summary` redacts values.
"""

import os
import json
from dataclasses import dataclass
from typing import Optional, Dict

try:
    # Optional dependency; do not hard-fail if missing
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


__all__ = [
    "AlgodConfig",
    "IndexerConfig",
    "Account",
    "get_network",
    "get_algod_config",
    "get_indexer_config",
    "get_account",
    "get_safe_config_summary",
]


# =============================================================================
# Defaults (Algonode)
# =============================================================================
ALGO_ALGOD_URLS: Dict[str, str] = {
    "testnet": "https://testnet-api.algonode.cloud",
    "mainnet": "https://mainnet-api.algonode.cloud",
    "betanet": "https://betanet-api.algonode.cloud",
}
ALGO_INDEXER_URLS: Dict[str, str] = {
    "testnet": "https://testnet-idx.algonode.cloud",
    "mainnet": "https://mainnet-idx.algonode.cloud",
    "betanet": "https://betanet-idx.algonode.cloud",
}


# =============================================================================
# Data classes
# =============================================================================
@dataclass(frozen=True)
class AlgodConfig:
    network: str
    algod_url: str
    headers: Dict[str, str]


@dataclass(frozen=True)
class IndexerConfig:
    network: str
    indexer_url: str
    headers: Dict[str, str]


@dataclass(frozen=True)
class Account:
    address: str
    mnemonic: Optional[str]
    private_key: Optional[bytes]  # None if not derivable (e.g., algosdk missing)


# =============================================================================
# Helpers
# =============================================================================
def _is_truthy(val: Optional[str]) -> bool:
    return str(val or "").lower() in {"1", "true", "yes", "y"}


def load_env(dotenv: bool = True) -> None:
    """
    Load .env from CWD if present and not running in CI.
    Harmless if file is missing or python-dotenv is not installed.
    """
    if not _is_truthy(os.getenv("CI")) and load_dotenv:
        load_dotenv(override=False)


def _redact(s: Optional[str], keep: int = 4) -> str:
    if not s:
        return ""
    if len(s) <= keep:
        return "*" * len(s)
    return s[:keep] + "…" + "*" * max(0, len(s) - keep - 1)


# =============================================================================
# Public API
# =============================================================================
def get_network() -> str:
    """
    Returns the configured Algorand network.
    Allowed: testnet (default), mainnet, betanet, sandbox, custom.
    """
    net = (os.getenv("ALGORAND_NETWORK") or "testnet").strip().lower()
    return net if net in {"testnet", "mainnet", "betanet", "sandbox", "custom"} else "testnet"


def get_algod_config() -> AlgodConfig:
    """
    Build Algod configuration.
    - URL order: ALGORAND_ALGOD_URL → defaults map (Algonode) → sandbox fallback
    - Auth header precedence:
        ALGOD_API_KEY / ALGORAND_ALGOD_API_KEY  ->  X-API-Key
        ALGORAND_ALGOD_TOKEN / ALGOD_TOKEN      ->  X-Algo-API-Token
    - Extra headers via ALGOD_EXTRA_HEADERS (JSON)
    """
    load_env()
    net = get_network()
    url = os.getenv("ALGORAND_ALGOD_URL") or ALGO_ALGOD_URLS.get(net)
    if not url:
        # classic sandbox (docker)
        url = "http://localhost:4001" if net == "sandbox" else ALGO_ALGOD_URLS["testnet"]

    api_key = os.getenv("ALGOD_API_KEY") or os.getenv("ALGORAND_ALGOD_API_KEY") or ""
    token = os.getenv("ALGORAND_ALGOD_TOKEN") or os.getenv("ALGOD_TOKEN") or ""

    headers: Dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key
    elif token:
        headers["X-Algo-API-Token"] = token

    extra = os.getenv("ALGOD_EXTRA_HEADERS")
    if extra:
        try:
            headers.update(json.loads(extra))
        except Exception:
            pass

    return AlgodConfig(network=net, algod_url=url, headers=headers)


def get_indexer_config() -> IndexerConfig:
    """
    Build Indexer configuration.
    - URL order: ALGORAND_INDEXER_URL → defaults map (Algonode) → sandbox fallback
    - Auth header precedence:
        INDEXER_API_KEY / ALGORAND_INDEXER_API_KEY  ->  X-API-Key
        ALGORAND_INDEXER_TOKEN / INDEXER_TOKEN      ->  X-Algo-API-Token
    - Extra headers via INDEXER_EXTRA_HEADERS (JSON)
    """
    load_env()
    net = get_network()
    url = os.getenv("ALGORAND_INDEXER_URL") or ALGO_INDEXER_URLS.get(net)
    if not url:
        url = "http://localhost:8980" if net == "sandbox" else ALGO_INDEXER_URLS["testnet"]

    api_key = os.getenv("INDEXER_API_KEY") or os.getenv("ALGORAND_INDEXER_API_KEY") or ""
    token = os.getenv("ALGORAND_INDEXER_TOKEN") or os.getenv("INDEXER_TOKEN") or ""

    headers: Dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key
    elif token:
        headers["X-Algo-API-Token"] = token

    extra = os.getenv("INDEXER_EXTRA_HEADERS")
    if extra:
        try:
            headers.update(json.loads(extra))
        except Exception:
            pass

    return IndexerConfig(network=net, indexer_url=url, headers=headers)


def get_account(require_signing: bool = False) -> Account:
    """
    Return configured account info.
    Supports:
      - ALGORAND_MNEMONIC (25 words)
      - ALGORAND_PRIVATE_KEY (base64)
      - Optional ALGORAND_WALLET_ADDRESS for consistency checks
    If require_signing=True and a private key cannot be derived → raises.
    """
    load_env()

    is_ci = _is_truthy(os.getenv("CI"))

    addr_env = os.getenv("ALGORAND_WALLET_ADDRESS") or ""
    mnem_env = os.getenv("ALGORAND_MNEMONIC") or ""
    pkey_b64 = os.getenv("ALGORAND_PRIVATE_KEY") or ""

    if not (mnem_env or pkey_b64):
        if require_signing:
            raise ValueError("Missing signing material: set ALGORAND_MNEMONIC or ALGORAND_PRIVATE_KEY.")
        return Account(address=addr_env, mnemonic=None, private_key=None)

    try:
        from algosdk import mnemonic, account as algo_account  # type: ignore
        from algosdk.encoding import is_valid_address          # type: ignore
        from base64 import b64decode

        derived_addr: Optional[str] = None
        private_key: Optional[bytes] = None

        if mnem_env:
            private_key = mnemonic.to_private_key(mnem_env)
            derived_addr = algo_account.address_from_private_key(private_key)
        elif pkey_b64:
            private_key = b64decode(pkey_b64)
            derived_addr = algo_account.address_from_private_key(private_key)

        if addr_env:
            if not is_valid_address(addr_env):
                raise ValueError("ALGORAND_WALLET_ADDRESS is not a valid Algorand address.")
            if derived_addr and addr_env != derived_addr and not is_ci:
                raise ValueError("Configured address does not match the one derived from the secret material.")

        address = addr_env or (derived_addr or "")
        if not address:
            raise ValueError("Unable to determine Algorand address from provided secrets.")

        return Account(address=address, mnemonic=mnem_env or None, private_key=private_key)

    except ImportError:
        if require_signing:
            raise ValueError("algosdk not installed: cannot sign transactions. Install 'py-algorand-sdk'.")
        return Account(address=addr_env, mnemonic=mnem_env or None, private_key=None)


# =============================================================================
# Safe debug summary (never prints secrets)
# =============================================================================
def get_safe_config_summary() -> Dict[str, str]:
    """
    Small redacted summary for troubleshooting.
    Does not include secrets in clear text.
    """
    algod = get_algod_config()
    idx = get_indexer_config()
    net = get_network()
    addr = os.getenv("ALGORAND_WALLET_ADDRESS") or ""
    has_mn = bool(os.getenv("ALGORAND_MNEMONIC"))
    has_pk = bool(os.getenv("ALGORAND_PRIVATE_KEY"))
    return {
        "network": net,
        "algod_url": algod.algod_url,
        "indexer_url": idx.indexer_url,
        "wallet_address": _redact(addr, keep=6),
        "has_mnemonic": str(has_mn),
        "has_private_key": str(has_pk),
    }
