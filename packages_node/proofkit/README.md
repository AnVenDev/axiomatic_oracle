# @axiomatic/proofkit

ProofKit is a minimal TypeScript/Node.js SDK to build and publish
**canonical on-chain p1 attestations** for valuations on Algorand.

It handles:

- Canonical JSON (ACJ-1-style) serialization
- Deterministic input hashing
- Building stable `p1` notes
- Publishing `p1` as a 0-ALGO self-payment with note
- Returning `{ txid, explorerUrl }` for verification

Used together with `@axiomatic/verifier` for full end-to-end trustless checks.

---

## Install

```bash
npm install @axiomatic/proofkit algosdk
# or
pnpm add @axiomatic/proofkit algosdk
```

Requirements:

* Node.js ≥ 18
* An Algorand account on TestNet/MainNet with enough ALGO for fees

---

## Core Concepts

A `p1` attestation encodes:

* `a` – `asset_tag`, e.g. `"re:EUR"`
* `mv` – `model_version`, e.g. `"v2"`
* `mh` – `model_hash_hex` (optional, 64-hex)
* `ih` – `input_hash_hex` (64-hex) over canonical input
* `v` – valuation (EUR)
* `u` – `[low, high]` interval (EUR)
* `ts` – Unix timestamp (seconds)

ProofKit **does not** run your valuation model.
It standardizes **how you commit** to the result on-chain in a deterministic, verifiable way.

---

## API Surface

From `@axiomatic/proofkit` you get:

```ts
import {
  NOTE_MAX_BYTES,
  DEFAULT_ASSET_TAG,
  buildP1,
  canonicalNoteBytesP1,
  assertNoteSizeOK,
  buildCanonicalInput,
  computeInputHash,
  publishP1,
  type Network,
  type P1,
} from "@axiomatic/proofkit";
```

Key pieces:

* `buildCanonicalInput(rec, allowedKeys, stripNone?)`
* `computeInputHash(rec, allowedKeys)`
* `buildP1({...})`
* `canonicalNoteBytesP1(p1)`
* `assertNoteSizeOK(p1, maxBytes?)`
* `publishP1({ p1, from, sign, network, pera?, algod?, waitRounds? })`

---

## 1. Canonical input hash

You decide which raw fields count as input.
ProofKit gives you a deterministic hash over that subset.

```ts
import { buildCanonicalInput, computeInputHash } from "@axiomatic/proofkit";

const allowedKeys = [
  "surface_sqm",
  "rooms",
  "zip_code",
  "year_built",
  // ...
];

const rawInput = {
  surface_sqm: 85,
  rooms: 3,
  zip_code: "20121",
  year_built: 1995,
  ignored: "not included",
};

const ih = await computeInputHash(rawInput, allowedKeys);
// ih: 64-hex string, stable across SDKs / backend
```

If you need the canonical subset explicitly:

```ts
const canonicalInput = buildCanonicalInput(rawInput, allowedKeys);
```

---

## 2. Build a `p1` attestation

```ts
import { buildP1 } from "@axiomatic/proofkit";

const p1 = buildP1({
  assetTag: "re:EUR",          // optional, default "re:EUR"
  modelVersion: "v2",
  modelHashHex: "",            // optional, 64-hex if used
  inputHashHex: ih,            // from computeInputHash
  valueEUR: 550_000,
  uncertaintyLowEUR: 520_000,
  uncertaintyHighEUR: 580_000,
  // timestampEpochSec: optional, defaults to now()
});
```

This will throw if:

* `uncertaintyLowEUR > uncertaintyHighEUR`
* any numeric field is non-finite
* `inputHashHex` / `modelHashHex` wrong length

---

## 3. Inspect canonical note bytes (optional)

Useful for debugging, audits, and golden tests.

```ts
import { canonicalNoteBytesP1, assertNoteSizeOK } from "@axiomatic/proofkit";

const { bytes, sha256, size } = await canonicalNoteBytesP1(p1);
await assertNoteSizeOK(p1); // uses NOTE_MAX_BYTES (1024) by default

console.log("note.size =", size);
console.log("note.sha256 =", sha256);
```

The same `sha256` is what a verifier will recompute independently.

---

## 4. Publish on Algorand (TestNet / MainNet)

You keep your keys. ProofKit only needs a **signer**.

Here’s a minimal Node.js example with `algosdk` and a mnemonic:

```ts
import algosdk from "algosdk";
import { buildP1, publishP1 } from "@axiomatic/proofkit";

const MNEMONIC = process.env.ALGORAND_MNEMONIC!;
const NETWORK = (process.env.ALGORAND_NETWORK || "testnet") as "testnet" | "mainnet";

const { addr, sk } = algosdk.mnemonicToSecretKey(MNEMONIC);
const from = addr.toString();

const p1 = buildP1({
  modelVersion: "v2",
  modelHashHex: "",
  inputHashHex: "a".repeat(64), // demo; use real ih in production
  valueEUR: 550000,
  uncertaintyLowEUR: 520000,
  uncertaintyHighEUR: 580000,
});

const sign = async (unsignedBytes: Uint8Array): Promise<Uint8Array> => {
  const tx = algosdk.decodeUnsignedTransaction(unsignedBytes);
  const { blob } = algosdk.signTransaction(tx, sk);
  return blob;
};

const { txid, explorerUrl } = await publishP1({
  p1,
  from,
  sign,
  network: NETWORK,
  // optional: algod client override, pera wallet in browser, waitRounds, ...
});

console.log("Published p1:", txid);
console.log("Explorer:", explorerUrl);
```

Default behavior:

* Creates a **0 ALGO** payment from `from` to `from`
* Attaches the canonical `p1` as `note`
* Waits for confirmation
* Returns `{ txid, explorerUrl }`
* Uses Algonode/AlgoExplorer-style public endpoints by default (can be overridden with your own `algod` client)

---

## 5. Using Pera Wallet (browser)

ProofKit can also work with Pera Wallet in a browser context:

```ts
import { publishP1 } from "@axiomatic/proofkit";
// import PeraWalletConnect from "@perawallet/connect";

const pera = new PeraWalletConnect();
const accounts = await pera.connect();
const from = accounts[0].address;

const sign = async (unsignedBytes: Uint8Array): Promise<Uint8Array> => {
  const [signed] = await pera.signTransactions([unsignedBytes]);
  return signed;
};

const { txid, explorerUrl } = await publishP1({
  p1,
  from,
  pera,   // or use `sign`, depending on your integration
  sign,
  network: "testnet",
});
```

---

## Guarantees & Interop

* JCS / hashing aligned with:

  * Axiomatic backend implementation
  * `axiomatic-proofkit` (Python)
  * `@axiomatic/verifier` (Node/Python)
* `p1` schema is intentionally minimal and versioned.
* No private keys leave your environment.
* Any third party can verify your published valuations using Verifier SDKs.
