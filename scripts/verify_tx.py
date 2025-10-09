# scripts/verify_tx.py
"""
Smoke test Verifier (Python): verifica una TX p1 via Indexer.

Prerequisiti:
- pip install -e python\axiomatic_verifier
- .env con (facoltativi):
    ALGORAND_NETWORK=testnet|mainnet
    INDEXER_URL=https://testnet-idx.algonode.cloud
Uso:
  python scripts\verify_tx.py <TXID>
"""

import os, sys, json
from pathlib import Path

def load_env(p: Path):
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line or line.strip().startswith("#") or "=" not in line: 
                continue
            k, v = line.split("=", 1)
            if k and (k not in os.environ):
                os.environ[k.strip()] = v.strip()
    except FileNotFoundError:
        pass

ROOT = Path(__file__).resolve().parents[1]
load_env(ROOT / ".env")

if len(sys.argv) < 2:
    print("Usage: python scripts\\verify_tx.py <TXID>", file=sys.stderr)
    sys.exit(1)

txid = sys.argv[1].strip()
network = (os.getenv("ALGORAND_NETWORK") or "testnet").strip()
indexer_url = (os.getenv("INDEXER_URL") or 
               ("https://mainnet-idx.algonode.cloud" if network=="mainnet" else "https://testnet-idx.algonode.cloud")).strip()

from axiomatic_verifier.verifier import verify_tx  # noqa

res = verify_tx(
    txid=txid,
    network=network,
    indexer_url=indexer_url,  # opzionale ma utile per override
    # niente algod_url: non è previsto
)

print(json.dumps(res, indent=2, ensure_ascii=False))
