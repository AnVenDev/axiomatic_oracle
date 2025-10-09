import { toJcsBytes, sha256Hex, jcsSha256 } from "./jcs.js";

export const NOTE_MAX_BYTES = 1024;
export const DEFAULT_ASSET_TAG = "re:EUR";

export type Network = "testnet" | "mainnet" | "betanet";

export interface BuildP1Opts {
  assetTag?: string;
  modelVersion: string;
  modelHashHex?: string;
  inputHashHex: string;
  valueEUR: number;
  uncertaintyLowEUR: number;
  uncertaintyHighEUR: number;
  timestampEpochSec?: number;
}

export interface P1 {
  s: "p1";
  a: string;
  mv: string;
  mh: string;
  ih: string;
  v: number;
  u: [number, number];
  ts: number;
}

function assertFinite(n: number, name: string) {
  if (typeof n !== "number" || !Number.isFinite(n)) throw new Error(`${name} must be a finite number`);
}

export function buildP1(opts: BuildP1Opts): P1 {
  const {
    assetTag = DEFAULT_ASSET_TAG,
    modelVersion,
    modelHashHex = "",
    inputHashHex,
    valueEUR,
    uncertaintyLowEUR,
    uncertaintyHighEUR,
    timestampEpochSec = Math.floor(Date.now() / 1000),
  } = opts;

  if (uncertaintyLowEUR > uncertaintyHighEUR) throw new Error("uncertaintyLowEUR > uncertaintyHighEUR");
  assertFinite(valueEUR, "valueEUR");
  assertFinite(uncertaintyLowEUR, "uncertaintyLowEUR");
  assertFinite(uncertaintyHighEUR, "uncertaintyHighEUR");
  if (!modelVersion?.trim()) throw new Error("modelVersion is required");
  if (inputHashHex.length !== 64) throw new Error("inputHashHex must be 64 hex chars");
  if (modelHashHex && modelHashHex.length !== 64) throw new Error("modelHashHex must be 64 hex chars (or empty)");

  const p1: P1 = {
    s: "p1",
    a: String(assetTag),
    mv: String(modelVersion),
    mh: String(modelHashHex || ""),
    ih: String(inputHashHex),
    v: Number(valueEUR),
    u: [Number(uncertaintyLowEUR), Number(uncertaintyHighEUR)],
    ts: Number(timestampEpochSec),
  };

  void toJcsBytes(p1); // ACJ-1 sanity
  return p1;
}

export async function canonicalNoteBytesP1(p1: P1): Promise<{ bytes: Uint8Array; sha256: string; size: number }> {
  if (!p1 || p1.s !== "p1") throw new Error("Invalid p1 object");
  const bytes = toJcsBytes(p1);
  const sha256 = await sha256Hex(bytes);
  return { bytes, sha256, size: bytes.length };
}

export async function assertNoteSizeOK(p1: P1, max = NOTE_MAX_BYTES): Promise<void> {
  const { size } = await canonicalNoteBytesP1(p1);
  if (size > max) throw new Error(`p1 note too large: ${size} bytes (max ${max})`);
}

export interface PublishOpts {
  p1: P1;
  network?: Network;
  pera?: any; // PeraWalletConnect (browser)
  algod?: any; // algosdk.Algodv2
  from?: string;
  sign?: (txnBytes: Uint8Array) => Promise<Uint8Array>; // custom signer (Node)
  waitRounds?: number;
}

export async function publishP1(opts: PublishOpts): Promise<{ txid: string; explorerUrl: string }> {
  const { p1, network = "testnet", pera, algod, from, sign, waitRounds = 4 } = opts;

  // import dinamico (ESM friendly)
  const algosdkMod = await import("algosdk");
  const algosdk: any = (algosdkMod as any).default ?? algosdkMod;

  await assertNoteSizeOK(p1);
  const { bytes } = await canonicalNoteBytesP1(p1);

  const client =
    algod ??
    new algosdk.Algodv2(
      "",
      `https://node.${network === "mainnet" ? "" : "testnet."}algoexplorerapi.io`,
      ""
    );

  // 1) Suggested params (cast sicuro per evitare mismatch di tipi in TS)
  const sp = await client.getTransactionParams().do();
  const suggestedParams: any = {
    fee: sp.fee,
    flatFee: sp.flatFee,
    firstRound: sp.firstRound,
    lastRound: sp.lastRound,
    genesisID: sp.genesisID,
    genesisHash: sp.genesisHash,
  };

  // 2) mittente
  let sender = from;
  if (!sender && pera) {
    const accounts = await pera.connect();
    sender = accounts?.[0]?.address;
  }
  if (!sender) throw new Error("Missing `from` or Pera wallet connection");

  // 3) Tx: usa *FromObject* (compat con la tua versione di algosdk)
  let txn: any;
  try {
    txn = algosdk.makePaymentTxnWithSuggestedParamsFromObject({
      sender,              
      receiver: sender,    
      amount: 0,
      note: bytes,
      suggestedParams: suggestedParams as any,
    });
  } catch (e1) {
    txn = algosdk.makePaymentTxnWithSuggestedParamsFromObject({
      from: sender,
      to: sender,
      amount: 0,
      note: bytes,
      suggestedParams: suggestedParams as any,
    });
  }

  // 4) Firma
  const unsignedBytes: Uint8Array = algosdk.encodeUnsignedTransaction(txn);
  let signed: Uint8Array;

  if (pera) {
    // ✅ Pera: API corretta (niente 'signers')
    const signedBlobs: Uint8Array[] = await (pera as any).signTransactions([unsignedBytes]);
    signed = signedBlobs[0];
  } else if (sign) {
    signed = await sign(unsignedBytes);
  } else {
    throw new Error("Provide `pera` or a custom `sign` function to sign the transaction");
  }

  // 5) Invio & conferma
  const sendRes = await client.sendRawTransaction(signed).do();
  const txId: string = sendRes.txId;
  await algosdk.waitForConfirmation(client, txId, waitRounds);

  const explorerUrl = `https://${network === "mainnet" ? "" : "testnet."}algoexplorer.io/tx/${txId}`;
  return { txid: txId, explorerUrl };
}

export function buildCanonicalInput(rec: Record<string, any>, allowedKeys: string[], stripNone = true): Record<string, any> {
  const allowed = new Set(allowedKeys.map(String));
  const out: Record<string, any> = {};
  for (const [kRaw, v] of Object.entries(rec)) {
    const k = String(kRaw);
    if (!allowed.has(k)) continue;
    if (stripNone && (v === null || typeof v === "undefined")) continue;
    out[k] = v;
  }
  return out;
}

export async function computeInputHash(rec: Record<string, any>, allowedKeys: string[]): Promise<string> {
  const cin = buildCanonicalInput(rec, allowedKeys);
  return jcsSha256(cin);
}
