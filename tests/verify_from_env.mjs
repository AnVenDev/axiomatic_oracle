import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

const envTxt = await fs.readFile(path.join(repoRoot, ".env"), "utf8").catch(() => "");
const env = Object.fromEntries(
  (envTxt || "")
    .split(/\r?\n/)
    .filter((l) => l && !l.trim().startsWith("#") && l.includes("="))
    .map((l) => { const i = l.indexOf("="); return [l.slice(0,i).trim(), l.slice(i+1).trim()]; })
);
const network = env.ALGORAND_NETWORK || "testnet";
const indexerUrl =
  env.INDEXER_URL ||
  (network === "mainnet"
    ? "https://mainnet-idx.algonode.cloud"
    : "https://testnet-idx.algonode.cloud");

const txid = process.argv[2];
if (!txid) { console.error("Usage: node scripts/verify_from_env.mjs <TXID>"); process.exit(1); }

const verifierEntry = path.join(repoRoot, "packages_node/verifier/dist/index.js");

const { verifyTx } = await import(pathToFileURL(verifierEntry).href);

const opts = { txid, network, indexerUrl };
const res = await verifyTx(opts);
console.log(res);