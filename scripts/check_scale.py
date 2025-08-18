# scripts/check_scale.py
from __future__ import annotations
from pathlib import Path
import json, os, joblib, numpy as np, pandas as pd      # type: ignore
from sklearn.linear_model import LinearRegression       # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def find_property_dir() -> Path:
    candidates = [
        PROJECT_ROOT / "notebooks" / "outputs" / "modeling" / "property",
        PROJECT_ROOT / "outputs" / "modeling" / "property",
    ]
    for p in candidates:
        if (p / "training_manifest.json").exists():
            return p
    # ultima spiaggia: se esiste la cartella, usala
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Cartella 'property' non trovata (né manifest).")

def resolve_path(s: str | None, base: Path) -> Path | None:
    if not s:
        return None
    # normalizza slash
    s_norm = s.replace("\\", "/")
    p = Path(s_norm)
    if p.is_absolute() and p.exists():
        return p
    # relative to base
    p1 = (base / s_norm).resolve()
    if p1.exists():
        return p1
    # relative to project root
    p2 = (PROJECT_ROOT / s_norm).resolve()
    if p2.exists():
        return p2
    return None

def load_manifest(prop_dir: Path) -> dict:
    mf_path = prop_dir / "training_manifest.json"
    if not mf_path.exists():
        raise FileNotFoundError(f"Manifest non trovato: {mf_path}")
    return json.loads(mf_path.read_text(encoding="utf-8"))

def pick_pipeline_path(mf: dict, prop_dir: Path) -> Path:
    paths = mf.get("paths", {}) or {}
    candidates = [
        paths.get("pipeline_path"),
        paths.get("model"),
        paths.get("rf_model"),
    ]
    for c in candidates:
        p = resolve_path(c, prop_dir)
        if p and p.exists():
            return p
    # fallback: cerca joblib noti
    globs = [
        list(prop_dir.glob("value_regressor*.joblib")),
        list((prop_dir.parent / "artifacts").glob("rf_champion_*.joblib")),
        list(prop_dir.glob("*.joblib")),
    ]
    for arr in globs:
        if arr:
            return arr[0]
    raise FileNotFoundError("Nessun file .joblib trovato (pipeline).")

def pick_dataset_path(mf: dict, prop_dir: Path) -> Path:
    paths = mf.get("paths", {}) or {}
    ds_candidates = [
        resolve_path(paths.get("dataset"), prop_dir),
        resolve_path(paths.get("dataset_path"), prop_dir),
        PROJECT_ROOT / "notebooks" / "outputs" / "dataset_generated.csv",
        PROJECT_ROOT / "outputs" / "dataset_generated.csv",
    ]
    for p in ds_candidates:
        if p and p.exists():
            return p
    raise FileNotFoundError("Dataset CSV non trovato (paths.dataset o fallback).")

def expected_features_from_manifest(mf: dict) -> tuple[list[str], list[str]]:
    feats = (mf.get("expected_features") or mf.get("feature_config") or {})
    cat = list(feats.get("categorical") or [])
    num = list(feats.get("numeric") or [])
    # dedup preservando ordine
    def _dedup(seq: list[str]) -> list[str]:
        seen, out = set(), []
        for s in seq:
            if s not in seen:
                seen.add(s); out.append(s)
        return out
    cat = _dedup(cat)
    num = _dedup([c for c in num if c not in cat])
    return cat, num

def main():
    prop_dir = find_property_dir()
    mf = load_manifest(prop_dir)
    pipe_path = pick_pipeline_path(mf, prop_dir)
    ds_path = pick_dataset_path(mf, prop_dir)

    pipe = joblib.load(pipe_path)

    cat, num = expected_features_from_manifest(mf)
    if not (cat or num):
        raise RuntimeError("expected_features/feature_config assenti nel manifest.")
    cols = cat + num

    df = pd.read_csv(ds_path)
    if "valuation_k" not in df.columns:
        raise RuntimeError("Il dataset non contiene 'valuation_k' (target).")

    # prepara X con colonne attese
    X = df.reindex(columns=cols)
    # predizione RAW (stessa scala del train → k€)
    y_pred = pipe.predict(X).astype(float)
    y_true = df["valuation_k"].astype(float).values

    # rapporti e retta di calibrazione (y_true = a*y_pred + b)
    mask = (y_true != 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]
    ratio = np.divide(y_pred_f, y_true_f, out=np.full_like(y_pred_f, np.nan), where=(y_true_f!=0))
    ratio = ratio[np.isfinite(ratio)]
    med = float(np.median(ratio)) if ratio.size else float("nan")
    p25 = float(np.percentile(ratio, 25)) if ratio.size else float("nan")
    p75 = float(np.percentile(ratio, 75)) if ratio.size else float("nan")

    lr = LinearRegression().fit(y_pred_f.reshape(-1,1), y_true_f)
    a = float(lr.coef_[0]); b = float(lr.intercept_); r2 = float(lr.score(y_pred_f.reshape(-1,1), y_true_f))

    print("Pipeline:", pipe_path)
    print(f"n={len(y_true_f)} | median(y_pred / y_true)={med:.4f} | IQR=[{p25:.4f}, {p75:.4f}]")
    print(f"Linear fit y_true = a*y_pred + b -> a={a:.2f}, b={b:.2f}, R2={r2:.3f}")

    if 0.005 <= med <= 0.02 and r2 >= 0.70:
        print(f"⇒ Probabile SCALA 1/100. Suggerito scale_factor ≈ {1/med:.1f}")
    elif 0.8 <= med <= 1.2 and r2 >= 0.70:
        print("⇒ Scala OK; il modello sembra coerente sul dataset.")
    else:
        print("⇒ Non è solo scala: possibile mismatch di feature/categorie o drift.")

if __name__ == "__main__":
    main()