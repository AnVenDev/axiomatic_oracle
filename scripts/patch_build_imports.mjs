// scripts/patch_build_imports.mjs
import fs from "node:fs/promises";
import path from "node:path";

const repoRoot = process.cwd();

async function patch(relPath) {
  const p = path.join(repoRoot, relPath);
  try {
    let s = await fs.readFile(p, "utf8");
    const before = s;
    // Aggiungi l'estensione .js agli import della JCS buildata
    s = s.replace(/shared\/jcs\/index(?=['"])/g, "shared/jcs/index.js");
    if (s !== before) {
      await fs.writeFile(p, s);
      console.log("patched:", relPath);
    } else {
      console.log("no change:", relPath);
    }
  } catch {
    console.log("skip (not found):", relPath);
  }
}

// patch per entrambi i pacchetti (layout “profondo” attuale)
await patch("packages/proofkit/dist/packages/proofkit/src/index.js");
await patch("packages/verifier/dist/packages/verifier/src/index.js");
