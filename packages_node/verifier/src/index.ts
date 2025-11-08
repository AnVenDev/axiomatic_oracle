import { toJcsBytes, sha256Hex, jcsSha256 } from "./jcs.js";

export type Network = "testnet" | "mainnet" | "betanet";

export interface VerifyResult {
  verified: boolean;
  reason?: string | null;
  mode: "p1" | "legacy" | "unknown";
  noteSha256?: string;
  rebuiltSha256?: string;
  confirmedRound?: number;
  explorerUrl?: string;
  note?: any;
}

function defaultIndexerBase(network: Network): string {
  const net = (network || "testnet").toLowerCase();
  // Default: Algonode indexer
  if (net === "mainnet") return "https://mainnet-idx.algonode.cloud";
  if (net === "betanet") return "https://betanet-idx.algonode.cloud";
  return "https://testnet-idx.algonode.cloud";
}

function explorerUrlFor(txid: string, network: Network): string {
  const net = (network || "testnet").toLowerCase();
  if (net === "mainnet") {
    return `https://explorer.perawallet.app/tx/${txid}`;
  }
  return `https://testnet.explorer.perawallet.app/tx/${txid}`;
}

// base64 → bytes (Node + browser safe-ish)
declare const Buffer: any;
function base64ToBytes(b64: string): Uint8Array {
  if (typeof atob === "function") {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  if (typeof Buffer !== "undefined") {
    const buf = Buffer.from(b64, "base64");
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  }
  throw new Error("No base64 decoder available for base64ToBytes");
}

export async function verifyTx(opts: {
  txid: string;
  network?: Network;
  indexerUrl?: string;
  maxSkewPastSec?: number;
  maxSkewFutureSec?: number;
}): Promise<VerifyResult> {
  const {
    txid,
    network = "testnet",
    indexerUrl,
    maxSkewPastSec = 600,
    maxSkewFutureSec = 120,
  } = opts;

  const explorerUrl = explorerUrlFor(txid, network);
  const base = indexerUrl || defaultIndexerBase(network);

  const res = await fetch(`${base}/v2/transactions/${txid}`);
  if (!res.ok) {
    return {
      verified: false,
      reason: `tx_not_found:${res.status}`,
      mode: "unknown",
      explorerUrl,
    };
  }

  const data = await res.json();
  const tx = data.transaction ?? data;
  const noteB64 = tx?.note;

  if (!noteB64) {
    return {
      verified: false,
      reason: "note_missing",
      mode: "unknown",
      explorerUrl,
    };
  }

  let p1: any;
  try {
    const bytes = base64ToBytes(noteB64);
    p1 = JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    return {
      verified: false,
      reason: "note_not_json",
      mode: "unknown",
      explorerUrl,
    };
  }

  // Mode detection (p1 / legacy / unknown)
  const mode: VerifyResult["mode"] =
    p1?.s === "p1"
      ? "p1"
      : (p1 && (p1.ref || p1.schema_version)) ? "legacy" : "unknown";

  if (mode !== "p1") {
    return {
      verified: mode === "legacy",
      reason: mode === "legacy" ? null : "unsupported_or_empty_note",
      mode,
      explorerUrl,
      note: p1,
    };
  }

  // == Hash parity (JCS) ==
  const rebuilt = await jcsSha256(p1);
  const onchain =
    (p1 && (p1.note_sha256 as string)) ||
    (p1 && (p1.sha256 as string)) ||
    undefined;

  if (onchain && onchain !== rebuilt) {
    return {
      verified: false,
      reason: "onchain_hash_mismatch",
      mode: "p1",
      noteSha256: onchain,
      rebuiltSha256: rebuilt,
      explorerUrl,
      note: p1,
    };
  }

  // == Time-window (ts epoch seconds) ==
  const nowSec = Math.floor(Date.now() / 1000);
  const ts = Number.isFinite(p1?.ts) ? Number(p1.ts) : NaN;
  if (Number.isFinite(ts)) {
    if (ts < nowSec - maxSkewPastSec || ts > nowSec + maxSkewFutureSec) {
      return {
        verified: false,
        reason: "ts_out_of_window",
        mode: "p1",
        noteSha256: onchain ?? rebuilt,
        rebuiltSha256: rebuilt,
        explorerUrl,
        note: p1,
      };
    }
  }

  return {
    verified: true,
    reason: null,
    mode: "p1",
    noteSha256: onchain ?? rebuilt,
    rebuiltSha256: rebuilt,
    confirmedRound: tx["confirmed-round"],
    explorerUrl,
    note: p1,
  };
}

// re-export util
export { toJcsBytes, sha256Hex, jcsSha256 };