# scripts/check_scale.py
from __future__ import annotations
"""
Module: check_scale.py — Quick sanity tool for model scale/calibration.

Goal
- Caricare pipeline e dataset, allineare le feature attese, e misurare:
  • Rapporto mediano y_pred / y_true (+ IQR)
  • Regressione lineare y_true = a * y_pred + b (R²)
- Riconoscere mismatch tipici di scala (es. 1/100) o problemi più profondi.

Heuristics (tunable via env):
- If 0.005 ≤ median ≤ 0.02 and R² ≥ 0.70 ⇒ probabile SCALA 1/100
- If 0.8  ≤ median ≤ 1.2 and R² ≥ 0.70 ⇒ Scala OK
- Otherwise ⇒ possibile mismatch di feature/categorie o drift

ENV (opzionali):
- SCALE_OK_MED_MIN=0.8
- SCALE_OK_MED_MAX=1.2
- SCALE_OK_R2_MIN=0.70
- SCALE_1_100_MED_MIN=0.005
- SCALE_1_100_MED_MAX=0.02
"""

# =========================
# Standard library imports
# =========================
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ===========
# Third-party
# ===========
import joblib  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore

# =============================================================================
# Constants / Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Heuristic thresholds (override via env if needed)
SCALE_OK_MED_MIN = float(os.getenv("SCALE_OK_MED_MIN", "0.8"))
SCALE_OK_MED_MAX = float(os.getenv("SCALE_OK_MED_MAX", "1.2"))
SCALE_OK_R2_MIN = float(os.getenv("SCALE_OK_R2_MIN", "0.70"))

SCALE_1_100_MED_MIN = float(os.getenv("SCALE_1_100_MED_MIN", "0.005"))
SCALE_1_100_MED_MAX = float(os.getenv("SCALE_1_100_MED_MAX", "0.02"))

# =============================================================================
# Path helpers
# =============================================================================
def find_property_dir() -> Path:
    """
    Individua la cartella 'modeling/property' contenente il manifest.
    Priority:
      1) notebooks/outputs/modeling/property
      2) outputs/modeling/property
      3) Se esiste 'property' senza manifest, usa comunque quella cartella.
    """
    candidates = [
        PROJECT_ROOT / "notebooks" / "outputs" / "modeling" / "property",
        PROJECT_ROOT / "outputs" / "modeling" / "property",
    ]
    for p in candidates:
        if (p / "training_manifest.json").exists():
            return p
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Cartella 'property' non trovata o priva di 'training_manifest.json'.")

def resolve_path(s: Optional[str], base: Path) -> Optional[Path]:
    """
    Risolve un percorso assoluto o relativo a:
      - base (cartella property)
      - PROJECT_ROOT
    Ritorna None se il file non esiste.
    """
    if not s:
        return None
    s_norm = s.replace("\\", "/")
    p = Path(s_norm)
    if p.is_absolute() and p.exists():
        return p
    p1 = (base / s_norm).resolve()
    if p1.exists():
        return p1
    p2 = (PROJECT_ROOT / s_norm).resolve()
    if p2.exists():
        return p2
    return None

# =============================================================================
# Manifest & artifact selection
# =============================================================================
def load_manifest(prop_dir: Path) -> Dict:
    mf_path = prop_dir / "training_manifest.json"
    if not mf_path.exists():
        raise FileNotFoundError(f"Manifest non trovato: {mf_path}")
    return json.loads(mf_path.read_text(encoding="utf-8"))

def pick_pipeline_path(mf: Dict, prop_dir: Path) -> Path:
    """
    Seleziona il file .joblib della pipeline/regressore:
    - Priorità: paths.pipeline_path -> paths.model -> paths.rf_model
    - Fallback: pattern comuni in property/ e artifacts/
    """
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

    globs = [
        list(prop_dir.glob("value_regressor*.joblib")),
        list((prop_dir.parent / "artifacts").glob("rf_champion_*.joblib")),
        list(prop_dir.glob("*.joblib")),
    ]
    for arr in globs:
        if arr:
            return arr[0]

    raise FileNotFoundError("Nessun file .joblib trovato per la pipeline/regressore.")

def pick_dataset_path(mf: Dict, prop_dir: Path) -> Path:
    """
    Seleziona dataset di valutazione:
    - Priorità: paths.dataset -> paths.dataset_path
    - Fallback: notebooks/outputs/dataset_generated.csv -> outputs/dataset_generated.csv
    """
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
    raise FileNotFoundError("Dataset non trovato (paths.dataset / paths.dataset_path o fallback).")

def expected_features_from_manifest(mf: Dict) -> Tuple[List[str], List[str]]:
    """
    Estrae (categorical, numeric) dal manifest in modo robusto, deduplicando con preservazione ordine.
    """
    feats = (mf.get("expected_features") or mf.get("feature_config") or {})
    cat = list(feats.get("categorical") or [])
    num = list(feats.get("numeric") or [])

    def _dedup(seq: List[str]) -> List[str]:
        seen, out = set(), []
        for s in seq:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    cat = _dedup(cat)
    num = _dedup([c for c in num if c not in cat])
    return cat, num

# =============================================================================
# Core logic
# =============================================================================
def _read_dataset(path: Path) -> pd.DataFrame:
    """
    Legge CSV o Parquet (auto). Richiede colonna 'valuation_k'.
    """
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)  # type: ignore
    else:
        df = pd.read_csv(path)
    if "valuation_k" not in df.columns:
        raise RuntimeError("Il dataset non contiene 'valuation_k' (target in k€).")
    return df

def _print_scale_diagnostics(
    pipe_path: Path,
    n: int,
    med: float,
    p25: float,
    p75: float,
    a: float,
    b: float,
    r2: float,
) -> None:
    print("Pipeline:", pipe_path)
    print(f"n={n} | median(y_pred / y_true)={med:.4f} | IQR=[{p25:.4f}, {p75:.4f}]")
    print(f"Linear fit y_true = a*y_pred + b -> a={a:.2f}, b={b:.2f}, R2={r2:.3f}")

    if SCALE_1_100_MED_MIN <= med <= SCALE_1_100_MED_MAX and r2 >= SCALE_OK_R2_MIN:
        print(f"⇒ Probabile SCALA 1/100. Suggerito scale_factor ≈ {1/med:.1f}")
    elif SCALE_OK_MED_MIN <= med <= SCALE_OK_MED_MAX and r2 >= SCALE_OK_R2_MIN:
        print("⇒ Scala OK; il modello sembra coerente sul dataset.")
    else:
        print("⇒ Non è solo scala: possibile mismatch di feature/categorie o drift.")

def main() -> None:
    # Locate artifacts
    prop_dir = find_property_dir()
    mf = load_manifest(prop_dir)
    pipe_path = pick_pipeline_path(mf, prop_dir)
    ds_path = pick_dataset_path(mf, prop_dir)

    # Load pipeline and dataset
    pipe = joblib.load(pipe_path)
    cat, num = expected_features_from_manifest(mf)
    if not (cat or num):
        raise RuntimeError("expected_features/feature_config assenti nel manifest.")
    cols = cat + num

    df = _read_dataset(ds_path)

    # Align features
    X = df.reindex(columns=cols)

    # Predict and compare against true valuation_k (both in k€)
    y_pred = np.asarray(pipe.predict(X), dtype=float)
    y_true = np.asarray(df["valuation_k"], dtype=float)

    # Filter valid pairs
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)
    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]
    if y_true_f.size == 0:
        raise RuntimeError("Nessun dato valido dopo il filtraggio (controllare dataset e features).")

    # Ratios and calibration
    ratio = np.divide(y_pred_f, y_true_f, out=np.full_like(y_pred_f, np.nan), where=(y_true_f != 0))
    ratio = ratio[np.isfinite(ratio)]
    med = float(np.median(ratio)) if ratio.size else float("nan")
    p25 = float(np.percentile(ratio, 25)) if ratio.size else float("nan")
    p75 = float(np.percentile(ratio, 75)) if ratio.size else float("nan")

    lr = LinearRegression().fit(y_pred_f.reshape(-1, 1), y_true_f)
    a = float(lr.coef_[0])
    b = float(lr.intercept_)
    r2 = float(lr.score(y_pred_f.reshape(-1, 1), y_true_f))

    _print_scale_diagnostics(pipe_path, len(y_true_f), med, p25, p75, a, b, r2)

if __name__ == "__main__":
    main()
