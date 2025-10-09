function ensureJsonSafe(value: any): any {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("Non-finite float not allowed in canonical JSON (NaN/Inf).");
    return value;
  }
  if (value === null || typeof value === "string" || typeof value === "boolean") return value;

  if (Array.isArray(value)) return value.map((v) => ensureJsonSafe(v));

  if (value && typeof value === "object") {
    const out: Record<string, any> = {};
    const keys = Object.keys(value).map(String).sort();
    for (const k of keys) out[k] = ensureJsonSafe((value as any)[k]);
    return out;
  }
  return String(value);
}

export function toJcsBytes(obj: unknown): Uint8Array {
  const safe = ensureJsonSafe(obj);
  const s = JSON.stringify(safe, null, 0).replace(/\s+/g, "");
  return new TextEncoder().encode(s);
}

export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  // Usa WebCrypto se disponibile (browser o Node >= 16 con webcrypto)
  const g: any = globalThis as any;
  const subtle = g.crypto?.subtle;

  if (subtle && typeof subtle.digest === "function") {
    // ⚠️ digest richiede un BufferSource “puro” (ArrayBuffer), non una view con offset
    const view =
      bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
        ? bytes.buffer
        : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const buf = await subtle.digest("SHA-256", view);
    return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
  }

  // Fallback Node (ESM-safe, no require)
  const { createHash } = await import("crypto");
  const { Buffer } = await import("buffer");
  return createHash("sha256").update(Buffer.from(bytes)).digest("hex");
}

export async function jcsSha256(obj: unknown): Promise<string> {
  return sha256Hex(toJcsBytes(obj));
}