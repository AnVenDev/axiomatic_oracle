// scripts/publish_from_env.mjs
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createRequire } from "node:module";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

// ---- .env
const envTxt = await fs.readFile(path.join(repoRoot, ".env"), "utf8").catch(() => "");
const env = Object.fromEntries(
  (envTxt || "")
    .split(/\r?\n/)
    .filter((l) => l && !l.trim().startsWith("#") && l.includes("="))
    .map((l) => { const i = l.indexOf("="); return [l.slice(0, i).trim(), l.slice(i + 1).trim()]; })
);

const network = (env.ALGORAND_NETWORK || "testnet").trim();
const mnemonic = (env.ALGORAND_MNEMONIC || "").trim();
if (!mnemonic) { console.error("❌ Missing ALGORAND_MNEMONIC in .env"); process.exit(1); }

// ---- proofkit (solo build & bytes della nota)
const proofkitEntry = path.join(repoRoot, "packages/proofkit/dist/packages/proofkit/src/index.js");
const { buildP1, canonicalNoteBytesP1 } = await import(pathToFileURL(proofkitEntry).href);

// ---- algosdk
async function loadAlgodSdk() {
  try { const m = await import("algosdk"); return m.default ?? m; } catch { }
  const req = createRequire(path.join(repoRoot, "packages/proofkit/package.json"));
  const resolved = req.resolve("algosdk");
  const m = await import(pathToFileURL(resolved).href);
  return m.default ?? m;
}
const algosdk = await loadAlgodSdk();

function makeAlgod() {
  const url = (env.ALGOD_URL || "").trim();
  const token = (env.ALGOD_TOKEN || "").trim();
  const headers = {};
  if (env.ALGOD_API_KEY) headers[env.ALGOD_API_KEY_HEADER || "X-API-Key"] = env.ALGOD_API_KEY;
  if (url) return new algosdk.Algodv2(token, url, "", headers);
  const fallback = network === "mainnet" ? "https://mainnet-api.algonode.cloud"
    : "https://testnet-api.algonode.cloud";
  return new algosdk.Algodv2("", fallback, "");
}
const algod = makeAlgod();

// ---- address dal mnemonic
const { sk, addr } = algosdk.mnemonicToSecretKey(mnemonic);
const fromStr = addr && addr.toString ? addr.toString() : String(addr);
const valid = algosdk.isValidAddress(fromStr);
console.log("Network:", network);
console.log("Algod:", (env.ALGOD_URL || `(default ${network}-api.algonode.cloud)`));
console.log("From:", fromStr, " valid?", valid, " len:", fromStr?.length);
if (!valid) { console.error("❌ Derived address not valid — interrompo."); process.exit(1); }

// ---- build p1 & nota
const p1 = buildP1({
  modelVersion: "v2",
  modelHashHex: "",
  inputHashHex: "a".repeat(64),
  valueEUR: 550000,
  uncertaintyLowEUR: 520000,
  uncertaintyHighEUR: 580000,
});
const { bytes: noteBytes } = await canonicalNoteBytesP1(p1);

// ---- suggested params
const sp = await algod.getTransactionParams().do();

// helper: prova varianti compatibili con versioni diverse dell'SDK
async function tryCreateTxn(variant) {
  switch (variant) {
    case "fromToString":
      return algosdk.makePaymentTxnWithSuggestedParamsFromObject({
        from: fromStr,
        to: fromStr,
        amount: 0,
        note: noteBytes,
        suggestedParams: sp,
      });
    case "fromToAddress": {
      const A = algosdk.decodeAddress(fromStr);
      return algosdk.makePaymentTxnWithSuggestedParamsFromObject({
        from: A,
        to: A,
        amount: 0,
        note: noteBytes,
        suggestedParams: sp,
      });
    }
    case "senderReceiverString":
      return algosdk.makePaymentTxnWithSuggestedParamsFromObject({
        sender: fromStr,
        receiver: fromStr,
        amount: 0,
        note: noteBytes,
        suggestedParams: sp,
      });
    case "senderReceiverAddress": {
      const A = algosdk.decodeAddress(fromStr);
      return algosdk.makePaymentTxnWithSuggestedParamsFromObject({
        sender: A,
        receiver: A,
        amount: 0,
        note: noteBytes,
        suggestedParams: sp,
      });
    }
    default:
      throw new Error("unknown variant");
  }
}

const variants = [
  "fromToString",
  "fromToAddress",
  "senderReceiverString",
  "senderReceiverAddress",
];

let txn, modeUsed = null, lastErr = null;
for (const v of variants) {
  try {
    txn = await tryCreateTxn(v);
    modeUsed = v;
    break;
  } catch (e) {
    lastErr = e;
    console.warn(`⚠️  makePayment...FromObject with ${v} failed:`, e?.message ?? e);
  }
}
if (!txn) {
  console.error("❌ Nessuna variante accettata dalla tua versione di algosdk.");
  console.error("   fromStr head/tail:", fromStr?.slice(0, 6), "...", fromStr?.slice(-6));
  console.error("   decodeAddress(publicKey len):", algosdk.decodeAddress(fromStr)?.publicKey?.length);
  console.error("Ultimo errore:", lastErr?.stack || lastErr);
  process.exit(1);
}

console.log("Txn creation mode:", modeUsed);

// ---- firma & invio
const { blob, txID } = algosdk.signTransaction(txn, sk);
await algod.sendRawTransaction(blob).do();
await algosdk.waitForConfirmation(algod, txID, 4);

const explorerUrl = `https://${network === "mainnet" ? "" : "testnet."}algoexplorer.io/tx/${txID}`;
console.log("PUBLISHED", { txid: txID, explorerUrl });
