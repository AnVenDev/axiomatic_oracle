// packages/*/src/jcs.ts
function ensureJsonSafe(value) {
    // normalizza undefined (specialmente in array) -> null
    if (typeof value === "undefined")
        return null;
    // numeri finiti
    if (typeof value === "number") {
        if (!Number.isFinite(value))
            throw new Error("Non-finite float not allowed in canonical JSON (NaN/Inf).");
        return value;
    }
    // pass-through semplici
    if (value === null || typeof value === "string" || typeof value === "boolean")
        return value;
    // BigInt -> string deterministica (come fallback sicuro cross-lang)
    if (typeof value === "bigint")
        return value.toString(10);
    // array / tuple
    if (Array.isArray(value))
        return value.map((v) => ensureJsonSafe(v));
    // oggetti: ordina chiavi e normalizza valori
    if (value && typeof value === "object") {
        const out = {};
        for (const k of Object.keys(value).map(String).sort())
            out[k] = ensureJsonSafe(value[k]);
        return out;
    }
    // fallback: stringify
    return String(value);
}
export function toJcsBytes(obj) {
    const safe = ensureJsonSafe(obj);
    // chiavi ordinate già in ensureJsonSafe; rimuoviamo ogni whitespace
    const s = JSON.stringify(safe, null, 0).replace(/\s+/g, "");
    return new TextEncoder().encode(s);
}
export async function sha256Hex(bytes) {
    const g = globalThis;
    const subtle = g.crypto?.subtle;
    if (subtle && typeof subtle.digest === "function") {
        // usa ArrayBuffer “tight” come BufferSource
        const view = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
            ? bytes.buffer
            : bytes.buffer.slice(bytes.byteOffset, bytes.byteLength + bytes.byteOffset);
        const buf = await subtle.digest("SHA-256", view);
        return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
    }
    // fallback Node (ESM-safe)
    const { createHash } = await import("node:crypto");
    const { Buffer } = await import("node:buffer");
    return createHash("sha256").update(Buffer.from(bytes)).digest("hex");
}
export async function jcsSha256(obj) {
    return sha256Hex(toJcsBytes(obj));
}
