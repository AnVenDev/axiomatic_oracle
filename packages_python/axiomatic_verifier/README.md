**Axiomatic Verifier (Python)** is the SDK for independently verifying on-chain **p1** attestations (and related formats) published on Algorand.

Given a transaction ID, it:

1. Fetches the transaction via an Algorand Indexer.
2. Extracts and parses the note.
3. Re-canonicalizes the JSON (JCS/ACJ-style).
4. Recomputes the SHA-256.
5. Applies lightweight rules (schema, hash, timestamp).
6. Returns a structured JSON-style result.

No dependency on Axiomatic backend: anyone can verify.

---

## Installation

Requires **Python 3.10+**.

```bash
pip install axiomatic_verifier
```

(`requests` is installed automatically as a dependency.)

---

## Exposed API

From `axiomatic_verifier`:

* `verify_tx(txid, network="testnet", indexer_url=None, max_skew_past_sec=..., max_skew_future_sec=...)`

Utility helpers:

* `to_jcs_bytes(obj)`
* `sha256_hex(b)`

---

## Quickstart: verify a p1 transaction

This example mirrors the internal smoke test and is suitable as a public snippet.

### 1. Environment (optional)

Create a `.env` file next to your script (optional):

```env
ALGORAND_NETWORK=testnet
# INDEXER_URL=https://testnet-idx.algonode.cloud
```

If `INDEXER_URL` is not set, the SDK will default to Algonode
(`https://mainnet-idx.algonode.cloud` or `https://testnet-idx.algonode.cloud`).

### 2. Example script (`verify_tx_example.py`)

```python
import os
import sys
import json
from pathlib import Path

from axiomatic_verifier import verify_tx

ROOT = Path(__file__).resolve().parent


def load_env(env_path: Path) -> None:
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if not line or line.strip().startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if k and (k not in os.environ):
                os.environ[k] = v
    except FileNotFoundError:
        pass


load_env(ROOT / ".env")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:", file=sys.stderr)
        print("  python verify_tx_example.py <TXID>", file=sys.stderr)
        sys.exit(1)

    txid = sys.argv[1].strip()
    network = (os.getenv("ALGORAND_NETWORK") or "testnet").strip()

    # Default Indexer: Algonode (can be overridden via INDEXER_URL)
    indexer_url = (
        os.getenv("INDEXER_URL")
        or (
            "https://mainnet-idx.algonode.cloud"
            if network == "mainnet"
            else "https://testnet-idx.algonode.cloud"
        )
    ).strip()

    res = verify_tx(
        txid=txid,
        network=network,
        indexer_url=indexer_url,
        # For examples: accept p1 issued within the last hour
        max_skew_past_sec=3600,
        max_skew_future_sec=300,
    )

    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

Run:

```bash
python verify_tx_example.py <TXID>
```

Use a txid produced by `axiomatic_proofkit` on the same network.

---

## Response model

`verify_tx` always returns a dict-like structure. Typical shapes:

### Valid p1

```json
{
  "txid": "...",
  "verified": true,
  "mode": "p1",
  "reason": null,
  "note_sha256": "...",
  "rebuilt_sha256": "...",
  "confirmed_round": 57318625,
  "explorer_url": "https://testnet.explorer.perawallet.app/tx/...",
  "note": {
    "s": "p1",
    "a": "re:EUR",
    "mv": "v2",
    "mh": "",
    "ih": "...",
    "v": 550000.0,
    "u": [520000.0, 580000.0],
    "ts": 1762609210
  }
}
```

### Stale / out-of-window p1

```json
{
  "txid": "...",
  "verified": false,
  "mode": "p1",
  "reason": "ts_out_of_window",
  "note_sha256": "...",
  "rebuilt_sha256": "...",
  "explorer_url": "..."
}
```

Meaning: structurally and cryptographically correct p1, but `ts` is outside the allowed time window (`max_skew_past_sec` / `max_skew_future_sec`).

### Unsupported / missing note

```json
{
  "txid": "...",
  "verified": false,
  "mode": "unknown",
  "reason": "unsupported_or_empty_note",
  "explorer_url": "..."
}
```

---

## Canonical JSON helpers (optional)

You can use the bundled JCS/ACJ-style helpers for your own golden tests:

```python
from axiomatic_verifier import to_jcs_bytes, sha256_hex
import json

obj = json.loads(open("p1.json", "r", encoding="utf-8").read())
b = to_jcs_bytes(obj)
print(sha256_hex(b))
```

This lets you confirm that your own tooling matches the canonicalization used on-chain.

---

This is an early-access verifier: please treat responses as structured signals to plug into your own risk/validation logic.