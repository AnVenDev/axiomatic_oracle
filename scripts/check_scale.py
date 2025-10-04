# scripts/check_scale.py
from __future__ import annotations
"""
Sanity tool per scala/calibrazione del modello.
Vedi README in testa al file originale per le heuristic.
"""

import json, os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCALE_OK_MED_MIN   = float(os.getenv("SCALE_OK_MED_MIN", "0.8"))
SCALE_OK_MED_MAX   = float(os.getenv("SCALE_OK_MED_MAX", "1.2"))
SCALE_OK_R2_MIN    = float(os.getenv("SCALE_OK_R2_MIN", "0.70"))
SCALE_1_100_MED_MIN= float(os.getenv("SCALE_1_100_MED_MIN", "0.005"))
SCALE_1_100_MED_MAX= float(os.getenv("SCALE_1_100_MED_MAX", "0.02"))

def find_property_dir() -> Path:
    candidates = [
        PROJECT_ROOT / "notebooks" / "outputs" / "modeling" / "property",
        PROJECT_ROOT / "outputs"    / "modeling" / "property",
    ]
    for p in candidates:
        if (p / "training_manifest.json").exists():
            return p
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Cartella 'property' non trovata.")

def resolve_path(s: Optional[str], base: Path) -> Optional[Path]:
    if not s: return None
    s_norm = s.replace("\\", "/")
    p = Path(s_norm)
    if p.is_absolute() and p.exists(): return p
    p1 = (base / s_norm).resolve()
    if p1.exists(): return p1
    p2 = (PROJECT_ROOT / s_norm).resolve()
    if p2.exists(): return p2
    return None

def load_manifest(prop_dir: Path) -> Dict:
    mf_path = prop_dir / "training_manifest.json"
    if not mf_path.exists():
        raise FileNotFoundError(f"Manifest non trovato: {mf_path}")
    return json.loads(mf_path.read_text(encoding="utf-8"))

def pick_pipeline_path(mf: Dict, prop_dir: Path) -> Path:
    paths = mf.get("paths", {}) or {}
    for c in (paths.get("pipeline_path"), paths.get("model"), paths.get("rf_model")):
        p = resolve_path(c, prop_dir)
        if p and p.exists():
            return p
    for arr in (
        list(prop_dir.glob("value_regressor*.joblib")),
        list((prop_dir.parent / "artifacts").glob("*.joblib")),
        list(prop_dir.glob("*.joblib")),
    ):
        if arr: return arr[0]
    raise FileNotFoundError("Nessun file .joblib trovato.")

def pick_dataset_path(mf: Dict, prop_dir: Path) -> Path:
    paths = mf.get("paths", {}) or {}
    candidates = [
        resolve_path(paths.get("dataset"), prop_dir),
        resolve_path(paths.get("dataset_path"), prop_dir),
        PROJECT_ROOT / "notebooks" / "outputs" / "dataset_generated.csv",
        PROJECT_ROOT / "outputs"    / "dataset_generated.csv",
    ]
    for p in candidates:
        if p and p.exists():
            return p
    raise FileNotFoundError("Dataset non trovato.")

def expected_features_from_manifest(mf: Dict) -> Tuple[List[str], List[str]]:
    feats = (mf.get("expected_features") or mf.get("feature_config") or {})
    cat = list(feats.get("categorical") or [])
    num = list(feats.get("numeric") or [])
    seen=set(); cat=[c for c in cat if not (c in seen or seen.add(c))]
    seen=set(cat); num=[c for c in num if not (c in seen or seen.add(c))]
    return cat, num

def _read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet",".pq"}:
        df = pd.read_parquet(path)  # type: ignore
    else:
        df = pd.read_csv(path)
    if "valuation_k" not in df.columns:
        raise RuntimeError("Dataset senza 'valuation_k' (target in k€).")
    return df

# -------- NEW: gestione TTR/inverse + pipeline unwrap ----------
def _unwrap_final_estimator(model):
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            model = model.steps[-1][1]
    except Exception:
        pass
    ttr = None
    if getattr(model, "__class__", type(None)).__name__ == "TransformedTargetRegressor":
        ttr = model
        base = getattr(model, "regressor", None)
        if base is not None:
            model = base
    return model, ttr

def _predict_nat(pipe, X_df: pd.DataFrame) -> np.ndarray:
    """Restituisce predizioni in k€ anche se il target è stato log-trasformato."""
    # Prova a usare l’intera pipeline (con preproc) se possibile
    try:
        y = pipe.predict(X_df)
        model, ttr = _unwrap_final_estimator(pipe)
    except Exception:
        # se non è una Pipeline sklearn, trattalo come stimatore “puro”
        model, ttr = _unwrap_final_estimator(pipe)
        y = model.predict(X_df)

    y = np.asarray(y, dtype=float).ravel()
    if ttr is None:
        # se il training non usava TTR ma log1p custom, qui potresti essere ancora in log:
        # fai un check euristico: se la mediana < 25 assume log1p e fai expm1 safe
        med = float(np.nanmedian(y))
        if med < 25:  # euristico: log1p prezzi casa difficilmente > ~25
            y = np.expm1(np.clip(y, -20.0, 12.0))
        return y

    # inverse via inverse_func oppure transformer_.inverse_transform
    try:
        invf = getattr(ttr, "inverse_func", None)
        if callable(invf):
            return np.asarray(invf(y), dtype=float).ravel()
    except Exception:
        pass
    try:
        tr = getattr(ttr, "transformer_", None) or getattr(ttr, "transformer", None)
        if tr is not None and hasattr(tr, "inverse_transform"):
            y2 = y.reshape(-1, 1)
            return np.asarray(tr.inverse_transform(y2), dtype=float).ravel()
    except Exception:
        pass
    return y

def _print_scale_diagnostics(pipe_path: Path, n: int, med: float, p25: float, p75: float, a: float, b: float, r2: float) -> None:
    print("Pipeline:", pipe_path)
    print(f"n={n} | median(y_pred / y_true)={med:.4f} | IQR=[{p25:.4f}, {p75:.4f}]")
    print(f"Linear fit y_true = a*y_pred + b -> a={a:.2f}, b={b:.2f}, R2={r2:.3f}")
    if SCALE_1_100_MED_MIN <= med <= SCALE_1_100_MED_MAX and r2 >= SCALE_OK_R2_MIN:
        print(f"⇒ Probabile SCALA 1/100. Suggerito OUT_MULT ≈ {int(round(1/med))} (tipico 100).")
    elif SCALE_OK_MED_MIN <= med <= SCALE_OK_MED_MAX and r2 >= SCALE_OK_R2_MIN:
        print("⇒ Scala OK; modello coerente.")
    else:
        print("⇒ Possibile mismatch feature/categorie o drift (non solo scala).")

def main() -> None:
    prop_dir = find_property_dir()
    mf = load_manifest(prop_dir)
    pipe_path = pick_pipeline_path(mf, prop_dir)
    ds_path   = pick_dataset_path(mf, prop_dir)

    pipe = joblib.load(pipe_path)
    cat, num = expected_features_from_manifest(mf)
    if not (cat or num):
        raise RuntimeError("expected_features/feature_config assenti nel manifest.")
    cols = cat + num

    df = _read_dataset(ds_path)
    X  = df.reindex(columns=cols)
    y_true = np.asarray(df["valuation_k"], dtype=float)

    y_pred = _predict_nat(pipe, X)  # <-- sempre in k€
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)
    y_true_f, y_pred_f = y_true[mask], y_pred[mask]
    if y_true_f.size == 0:
        raise RuntimeError("Nessun dato valido dopo il filtraggio.")

    ratio = np.divide(y_pred_f, y_true_f, out=np.full_like(y_pred_f, np.nan), where=(y_true_f != 0))
    ratio = ratio[np.isfinite(ratio)]
    med  = float(np.median(ratio)) if ratio.size else float("nan")
    p25  = float(np.percentile(ratio, 25)) if ratio.size else float("nan")
    p75  = float(np.percentile(ratio, 75)) if ratio.size else float("nan")

    lr = LinearRegression().fit(y_pred_f.reshape(-1,1), y_true_f)
    a  = float(lr.coef_[0]); b = float(lr.intercept_); r2 = float(lr.score(y_pred_f.reshape(-1,1), y_true_f))

    _print_scale_diagnostics(pipe_path, len(y_true_f), med, p25, p75, a, b, r2)

if __name__ == "__main__":
    main()