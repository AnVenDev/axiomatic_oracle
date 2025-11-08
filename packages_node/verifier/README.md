# @axiomatic/verifier

`@axiomatic/verifier` is a tiny TypeScript/Node.js SDK to **trustlessly verify p1 attestations on Algorand**.

It lets you:

- Fetch a transaction by `txid`
- Decode the note
- Rebuild canonical JSON bytes (JCS-style)
- Recompute `sha256` and compare with on-chain data
- Enforce a time window on `ts`
- Get a clean `verified / reason / note / explorerUrl` result

No secrets, no backend required.  
Works together with `@axiomatic/proofkit` and the Axiomatic backend.

---

## Install

```bash
npm install @axiomatic/verifier
# or
pnpm add @axiomatic/verifier
```

Requirements:

* Node.js ≥ 18 (for built-in `fetch` & `TextEncoder`/`TextDecoder`)
* Public Indexer endpoint (default uses AlgoExplorer-style; configurable)

---

## API

```ts
import {
  verifyTx,
  toJcsBytes,
  sha256Hex,
  jcsSha256,
  type Network,
} from "@axiomatic/verifier";
```

### `verifyTx(options)`

```ts
type Network = "testnet" | "mainnet" | "betanet";

interface VerifyOptions {
  txid: string;
  network?: Network;        // default: "testnet"
  indexerUrl?: string;      // override default indexer
  maxSkewPastSec?: number;  // default: 600  (10 min)
  maxSkewFutureSec?: number;// default: 120  (2 min)
}

interface VerifyResult {
  verified: boolean;
  reason?: string | null;
  mode: "p1" | "legacy" | "unknown";
  noteSha256?: string;
  rebuiltSha256?: string;
  confirmedRound?: number;
  explorerUrl?: string;
  note?: any;               // parsed JSON note (p1) for inspection/debug
}
```

---

## Basic usage

Verify a `p1` attestation given its transaction id:

```ts
import { verifyTx } from "@axiomatic/verifier";

const txid = "ZBDWDAILCTKDXINHUVJCXN2VKNIDTQREGOQ6HB3OHIO6ED6ATBIQ";

const res = await verifyTx({
  txid,
  network: "testnet",
  // indexerUrl: "https://testnet-idx.algonode.cloud", // optional override
});

if (res.verified && res.mode === "p1") {
  console.log("✅ Verified p1 attestation");
  console.log("Note hash:", res.noteSha256);
  console.log("Explorer:", res.explorerUrl);
  console.log("Decoded note:", res.note);
} else {
  console.log("❌ Not verified");
  console.log("Mode:", res.mode);
  console.log("Reason:", res.reason);
}
```

This will:

1. Fetch the transaction from the configured indexer
2. Read the `note` field (base64)
3. Parse JSON
4. Check `s === "p1"` (otherwise `mode: "legacy" | "unknown"`)
5. Rebuild canonical bytes via JCS helper
6. Recompute `sha256`
7. If the note contains a `note_sha256`/`sha256` field, compare against the rebuilt hash
8. Optionally enforce a time window on `ts`
9. Return a structured `VerifyResult`

---

## Modes

* `mode: "p1"`

  * Note has `s: "p1"` and passes structural checks
* `mode: "legacy"`

  * Note looks like an old / non-p1 schema (best-effort, implementation-dependent)
* `mode: "unknown"`

  * Missing note, invalid JSON, unsupported format, or fetch issue

You can use this to **progressively migrate** to `p1` while still handling legacy data if needed.

---

## Explorer URL

`verifyTx` returns an `explorerUrl` convenience link, e.g.:

```txt
https://testnet.explorer.perawallet.app/tx/<TXID>
```

(or equivalent explorer domain depending on configuration/version).

This is only for humans; verification logic does **not** depend on it.

---

## JCS / Hash Utilities

The same helpers used internally are exported for your own checks:

```ts
import { toJcsBytes, sha256Hex, jcsSha256 } from "@axiomatic/verifier";

const obj = { a: 1, b: ["x", 2], z: 3 };
const bytes = toJcsBytes(obj);       // canonical JSON bytes
const hash = await sha256Hex(bytes); // hex string
const hash2 = await jcsSha256(obj);  // convenience: JCS(obj) then sha256
```

Properties:

* Deterministic (sorted keys, no whitespace)
* Rejects non-finite numbers (NaN/Inf)
* Aligned with:

  * `@axiomatic/proofkit`
  * `axiomatic_proofkit` (Python)
  * `axiomatic_verifier` (Python)
  * Axiomatic backend `canon.py`

---

## Typical Flow with ProofKit

1. **Producer** (valuation engine / platform):

   * Builds canonical input
   * Computes `input_hash`
   * Builds `p1`
   * Publishes on-chain via `@axiomatic/proofkit`

2. **Verifier** (investor, bank, marketplace, auditor):

   * Calls `@axiomatic/verifier.verifyTx({ txid })`
   * Confirms:

     * Note is well-formed `p1`
     * Hash matches canonical bytes
     * Timestamp is sane
   * Optionally replays the valuation off-chain using provided inputs

No need to trust Axiomatic’s servers.
Everything needed to verify is on-chain + in your stack.
