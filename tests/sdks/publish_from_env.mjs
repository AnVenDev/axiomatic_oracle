// tests/publish_from_env.mjs
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import algosdk from "algosdk";
import { buildP1, publishP1 } from "../../packages_node/proofkit/dist/index.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..");

async function loadEnv() {
  const envPath = path.join(ROOT, ".env");
  let txt;
  try {
    txt = await fs.readFile(envPath, "utf8");
  } catch {
    console.warn(`⚠️ .env not found at ${envPath}`);
    return;
  }

  for (const line of txt.split(/\r?\n/)) {
    if (!line || line.trim().startsWith("#") || !line.includes("=")) continue;

    const i = line.indexOf("=");
    const key = line.slice(0, i).trim();
    const val = line.slice(i + 1).trim();
    if (!key) continue;

    const existing = process.env[key];
    if (!existing || existing.trim() === "") {
      process.env[key] = val;
    }
  }
}

async function main() {
  await loadEnv();

  const network = (process.env.ALGORAND_NETWORK || "testnet").trim();
  const mnemonic = (process.env.ALGORAND_MNEMONIC || "").trim();

  console.log("DBG ALGORAND_MNEMONIC length:", mnemonic.length);

  if (!mnemonic) {
    console.error("❌ Missing ALGORAND_MNEMONIC in .env (or empty)");
    process.exit(1);
  }

  const { sk, addr } = algosdk.mnemonicToSecretKey(mnemonic);
  const from = addr && addr.toString ? addr.toString() : String(addr);

  console.log("Network:", network);
  console.log("From:", from);

  const p1 = buildP1({
    modelVersion: "v2",
    modelHashHex: "",
    inputHashHex: "a".repeat(64),
    valueEUR: 550000,
    uncertaintyLowEUR: 520000,
    uncertaintyHighEUR: 580000,
  });

  const sign = async (unsignedBytes) => {
    const tx = algosdk.decodeUnsignedTransaction(unsignedBytes);
    const { blob } = algosdk.signTransaction(tx, sk);
    return blob;
  };

  const res = await publishP1({
    p1,
    from,
    sign,
    network,
  });

  console.log("PUBLISHED", res);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
