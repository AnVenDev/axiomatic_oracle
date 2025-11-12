`@axiomatic_oracle/verifier` is the TypeScript/JavaScript SDK to independently verify **p1** attestations (and selected legacy formats) published on Algorand.

Given a transaction ID, it:

1. Fetches the transaction via an Algorand indexer.
2. Reads and decodes the note.
3. Re-canonicalizes the JSON (JCS-style).
4. Recomputes the SHA-256.
5. Applies basic validation rules (schema, hash, timestamp).
6. Returns a structured result that you can feed into your own logic.

It is designed to work with notes produced by:

- `@axiomatic_oracle/proofkit` (Node),
- `axiomatic_proofkit` (Python),
- and compatible Axiomatic Oracle integrations.

---

## Installation

```bash
npm install @axiomatic_oracle/verifier
```

Requirements:

* Node.js **18+** or a runtime with:

  * `fetch`
  * `TextDecoder`
  * `crypto.subtle` or a compatible SHA-256 implementation.

If you run on older Node.js versions, you will need to polyfill `fetch` and related globals.

---

## Exports

From `@axiomatic_oracle/verifier`:

* `verifyTx(options)` → `Promise<VerifyResult>`
* `toJcsBytes(obj)` (utility)
* `sha256Hex(bytes)` (utility)
* `jcsSha256(obj)` (utility)

```ts
type Network = "testnet" | "mainnet" | "betanet";

interface VerifyResult {
  verified: boolean;
  reason?: string | null;
  mode: "p1" | "legacy" | "unknown";
  noteSha256?: string;
  rebuiltSha256?: string;
  confirmedRound?: number;
  explorerUrl?: string;
  note?: any;
}
```

---

## Indexer and explorer defaults

* Indexer:

  * mainnet: `https://mainnet-idx.algonode.cloud`
  * testnet: `https://testnet-idx.algonode.cloud`
  * betanet: `https://betanet-idx.algonode.cloud`
* Explorer:

  * mainnet: `https://explorer.perawallet.app/tx/{txid}`
  * testnet: `https://testnet.explorer.perawallet.app/tx/{txid}`

You can override `indexerUrl` if you run your own infrastructure.

---

## Quickstart: verify a p1 from Node.js

```ts
import "dotenv/config";
import { verifyTx } from "@axiomatic_oracle/verifier";

async function main() {
  const txid = process.argv[2];
  if (!txid) {
    console.error("Usage: node verify_p1.js <TXID>");
    process.exit(1);
  }

  const network = (process.env.ALGORAND_NETWORK || "testnet").trim();
  const indexerUrl =
    process.env.INDEXER_URL ||
    (network === "mainnet"
      ? "https://mainnet-idx.algonode.cloud"
      : "https://testnet-idx.algonode.cloud");

  const res = await verifyTx({
    txid,
    network,
    indexerUrl,
    // Example: accept attestations up to 1 hour old,
    // and up to 5 minutes in the future (clock skew).
    maxSkewPastSec: 3600,
    maxSkewFutureSec: 300,
  });

  console.log(JSON.stringify(res, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
```

Use a `txid` produced by `@axiomatic/proofkit` (or the Python ProofKit) on the same network.

---

## Result semantics

`verifyTx` always returns a `VerifyResult`:

### Valid and fresh p1

```json
{
  "verified": true,
  "mode": "p1",
  "reason": null,
  "noteSha256": "...",
  "rebuiltSha256": "...",
  "confirmedRound": 57318625,
  "explorerUrl": "https://testnet.explorer.perawallet.app/tx/...",
  "note": {
    "s": "p1",
    "a": "re:EUR",
    "mv": "v2",
    "mh": "",
    "ih": "...",
    "v": 550000,
    "u": [520000, 580000],
    "ts": 1762609210
  }
}
```

### p1 structurally valid but stale / outside time window

```json
{
  "verified": false,
  "mode": "p1",
  "reason": "ts_out_of_window",
  "noteSha256": "...",
  "rebuiltSha256": "...",
  "explorerUrl": "..."
}
```

This means: the attestation is well-formed and hash-consistent, but its `ts` is outside the allowed window (`maxSkewPastSec` / `maxSkewFutureSec`). You can tune these thresholds according to your policy.

### Unsupported or empty note

```json
{
  "verified": false,
  "mode": "unknown",
  "reason": "unsupported_or_empty_note",
  "explorerUrl": "..."
}
```

### Legacy notes

If the note matches selected legacy formats (`ref`, `schema_version`, etc.), the verifier will return:

```json
{
  "verified": true,
  "mode": "legacy",
  "reason": null,
  "note": { ... }
}
```

This allows you to keep backward compatibility while progressively migrating to p1.

---

## Canonical JSON helpers

You can also use the bundled helpers to validate your own canonicalization:

```ts
import { toJcsBytes, jcsSha256 } from "@axiomatic_oracle/verifier";

const obj = { b: 2, a: 1 };
const bytes = toJcsBytes(obj);
const hash = await jcsSha256(obj);

console.log(hash);
```

They implement the same JCS-style normalization used by the p1 verifier and the ProofKit SDKs.

---

`@axiomatic_oracle/verifier` is intended as a low-level, auditable building block. Integrators are encouraged to layer their own business rules (e.g. allowed models, assets, issuers) on top of the returned `VerifyResult`.