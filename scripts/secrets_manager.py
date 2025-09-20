# scripts/secrets_manager.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict
import json

try:
    # Carichiamo dotenv solo se disponibile (non hard-dep)
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
# Costanti & mappe di default (Algonode)
# =============================================================================

ALGO_ALGOD_URLS: Dict[str, str] = {
    "testnet": "https://testnet-api.algonode.cloud",
    "mainnet": "https://mainnet-api.algonode.cloud",
    "betanet": "https://betanet-api.algonode.cloud"

}
ALGO_INDEXER_URLS: Dict[str, str] = {
    "testnet": "https://testnet-idx.algonode.cloud",
    "mainnet": "https://mainnet-idx.algonode.cloud",
    "betanet": "https://betanet-idx.algonode.cloud"
}

# =============================================================================
# Dataclasses di configurazione
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
    private_key: Optional[bytes]  # None se la libreria non è disponibile o non fornita


# =============================================================================
# Helpers
# =============================================================================

def _is_truthy(val: Optional[str]) -> bool:
    return str(val or "").lower() in {"1", "true", "yes", "y"}


def load_env(dotenv: bool = True) -> None:
    """Carica .env nella working dir se non siamo in CI e se disponibile."""
    if not _is_truthy(os.getenv("CI")) and load_dotenv:
        # Non fallisce se .env non esiste
        load_dotenv(override=False)


def _redact(s: Optional[str], keep: int = 4) -> str:
    if not s:
        return ""
    if len(s) <= keep:
        return "*" * len(s)
    return s[:keep] + "…" + "*" * max(0, len(s) - keep - 1)


def get_network() -> str:
    net = (os.getenv("ALGORAND_NETWORK") or "testnet").strip().lower()
    if net in {"testnet", "mainnet", "betanet", "sandbox", "custom"}:  # aggiungi betanet
        return net
    return "testnet"

def get_algod_config() -> AlgodConfig:
    """
    Restituisce configurazione dell'algod:
    - URL: da ALGORAND_ALGOD_URL o mappa di default (Algonode)
    - Header auth (ordine): ALGOD_API_KEY → X-API-Key, altrimenti ALGORAND_ALGOD_TOKEN/ALGOD_TOKEN → X-Algo-API-Token
    - Extra header opzionali: ALGOD_EXTRA_HEADERS (JSON)
    """
    load_env()
    net = get_network()
    url = os.getenv("ALGORAND_ALGOD_URL") or ALGO_ALGOD_URLS.get(net)
    if not url:
        # sandbox classico (docker): http://localhost:4001
        url = "http://localhost:4001" if net == "sandbox" else ALGO_ALGOD_URLS["testnet"]

    # auth precedence: API key → Token → none
    api_key = os.getenv("ALGOD_API_KEY") or os.getenv("ALGORAND_ALGOD_API_KEY") or ""
    token = os.getenv("ALGORAND_ALGOD_TOKEN") or os.getenv("ALGOD_TOKEN") or ""
    headers: Dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key
    elif token:
        headers["X-Algo-API-Token"] = token

    # merge extra headers (JSON string)
    extra = os.getenv("ALGOD_EXTRA_HEADERS")
    if extra:
        try:
            headers.update(json.loads(extra))
        except Exception:
            pass
    return AlgodConfig(network=net, algod_url=url, headers=headers)


def get_indexer_config() -> IndexerConfig:
    """
    Restituisce configurazione dell'indexer:
     - URL: da ALGORAND_INDEXER_URL o mappa di default (Algonode)
     - Header auth (ordine): INDEXER_API_KEY → X-API-Key, altrimenti ALGORAND_INDEXER_TOKEN/INDEXER_TOKEN → X-Algo-API-Token
     - Extra header opzionali: INDEXER_EXTRA_HEADERS (JSON)
    """
    load_env()
    net = get_network()
    url = os.getenv("ALGORAND_INDEXER_URL") or ALGO_INDEXER_URLS.get(net)
    if not url:
        # sandbox classico (docker): http://localhost:8980
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
    Restituisce l'account configurato.
    - Supporta ALGORAND_MNEMONIC (25 parole) oppure ALGORAND_PRIVATE_KEY (base64)
    - ALGORAND_WALLET_ADDRESS è opzionale: se presente, viene verificata la consistenza
    - Se require_signing=True e non si riesce a derivare la private key → solleva errore
    """
    load_env()

    # CI: se vuoi avere credenziali fake, setta CI=true e ALGORAND_MNEMONIC finto
    is_ci = _is_truthy(os.getenv("CI"))

    addr_env = os.getenv("ALGORAND_WALLET_ADDRESS") or ""
    mnem_env = os.getenv("ALGORAND_MNEMONIC") or ""
    pkey_b64 = os.getenv("ALGORAND_PRIVATE_KEY") or ""  # opzionale (base64)

    # Nessun segreto → se serve firmare, errore
    if not (mnem_env or pkey_b64):
        if require_signing:
            raise ValueError("Missing signing material: set ALGORAND_MNEMONIC or ALGORAND_PRIVATE_KEY.")
        return Account(address=addr_env, mnemonic=None, private_key=None)

    # Proviamo ad usare algosdk se presente
    try:
        from algosdk import mnemonic, account as algo_account  # type: ignore
        from algosdk.encoding import is_valid_address  # type: ignore
        from base64 import b64decode

        derived_addr: Optional[str] = None
        private_key: Optional[bytes] = None

        if mnem_env:
            private_key = mnemonic.to_private_key(mnem_env)
            derived_addr = algo_account.address_from_private_key(private_key)
        elif pkey_b64:
            private_key = b64decode(pkey_b64)
            derived_addr = algo_account.address_from_private_key(private_key)

        # Se abbiamo anche l'address esplicito, verifichiamo consistenza
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
# Convenience per debug sicuro (non logga segreti)
# =============================================================================

def get_safe_config_summary() -> Dict[str, str]:
    """
    Restituisce un piccolo sommario della config (redatto) per debug.
    Non include mai segreti in chiaro.
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
