from __future__ import annotations

"""
Dataset builder:
- Generazione batched e deterministica (rng iniettato)
- Normalizzazione robusta dei pesi location
- Error budget configurabile sui record falliti
- Progress bar opzionale (tqdm se disponibile)
- Drift & quality report invariati (compute_location_drift, enrich_quality_report)
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np                                  # type: ignore
import pandas as pd                                 # type: ignore
from numpy.random import Generator as NPGenerator   # type: ignore

from notebooks.shared.n01_generate_dataset.asset_factory import generate_asset
from notebooks.shared.common.quality import enrich_quality_report, generate_base_quality_report
from notebooks.shared.n03_train_model.metrics import compute_location_drift

logger = logging.getLogger(__name__)

__all__ = [
    "generate_dataset_df",
]

def _normalize_weights(raw_weights: Mapping[str, float], locations: List[str]) -> Dict[str, float]:
    """Normalizza pesi per location; fallback uniforme se somma non positiva o lista vuota."""
    if not isinstance(raw_weights, Mapping) or not locations:
        return {loc: 1.0 / max(len(locations), 1) for loc in locations}

    probs = np.array([float(raw_weights.get(loc, 0.0)) for loc in locations], dtype=float)
    s = probs.sum()
    if s <= 0.0 or not np.isfinite(s):
        return {loc: 1.0 / len(locations) for loc in locations}

    probs = probs / s
    return {loc: float(p) for loc, p in zip(locations, probs)}

def _maybe_tqdm(total: int, enabled: bool):
    """Ritorna un contesto progress (no-op se tqdm non disponibile o disabled)."""
    if not enabled:
        class Dummy:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def update(self, *a, **k): pass
        return Dummy()
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        return _maybe_tqdm(total, enabled=False)
    return tqdm(total=total)

def generate_dataset_df(
    config: Mapping[str, Any],
    locations: List[str],
    urban_map: Mapping[str, str],
    region_map: Mapping[str, str],
    seasonality: Mapping[int, float],
    city_base_prices: Mapping[str, Mapping[str, float]],
    rng: NPGenerator,
    reference_time: datetime,
    *,
    batch_size: int = 1000,
    show_progress: bool = True,
    validate_each: bool = True,
    error_budget_pct: float = 0.01,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Genera un dataset di asset sintetici in batch.

    Args:
        config: configurazione end-to-end (contiene location_weights, expected_profile, ecc.)
        locations: lista di location valide
        urban_map, region_map, seasonality, city_base_prices: mapping di supporto
        rng: numpy.random.Generator (determinismo)
        reference_time: timestamp di riferimento per le feature temporali
        batch_size: numero di record per batch
        show_progress: se True, usa tqdm se disponibile
        validate_each: se True, valida ogni asset (via factory)
        error_budget_pct: frazione massima di record falliti ammessa prima del fail

    Returns:
        (DataFrame, quality_report_dict)
    """
    t0 = time.time()
    asset_type = str(config.get("asset_type", "property"))
    n_rows = int(config.get("n_rows", 0))
    if n_rows <= 0:
        raise ValueError("n_rows must be > 0")

    logger.info("[DATASET] Generating %d assets of type '%s'", n_rows, asset_type)

    # Normalizza location_weights e prepara array per sampling
    raw_location_weights = config.get("location_weights", {}) or {}
    normalized_location_weights = _normalize_weights(raw_location_weights, locations)
    weight_array = np.array([normalized_location_weights.get(loc, 0.0) for loc in locations], dtype=float)
    if weight_array.sum() > 0:
        weight_array = weight_array / weight_array.sum()
    else:
        weight_array = np.ones(len(locations), dtype=float) / max(len(locations), 1)

    # Sampling policy
    sampling_cfg = config.get("sampling", {}) or {}
    allow_replacement = bool(sampling_cfg.get("allow_replacement", True))

    # Costruzione sequenza location
    if len(locations) >= n_rows:
        # Permutazione senza replacement per i primi n_rows
        permuted = rng.permutation(locations)
        location_sequence = list(permuted[:n_rows])
    else:
        if not allow_replacement:
            raise ValueError(
                "La sequenza delle location è più corta di n_rows e 'allow_replacement' è disabilitato."
            )
        logger.info(
            "Numero di location (%d) < n_rows (%d): campiono con replacement secondo i pesi.",
            len(locations), n_rows,
        )
        location_sequence = list(rng.choice(locations, size=n_rows, replace=True, p=weight_array))

    dfs: List[pd.DataFrame] = []
    failures: List[int] = []

    n_batches = (n_rows + batch_size - 1) // batch_size
    with _maybe_tqdm(total=n_rows, enabled=show_progress) as pbar:
        for b in range(n_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, n_rows)
            batch_rows: List[Dict[str, Any]] = []

            for i in range(start, end):
                loc = location_sequence[i]
                try:
                    asset = generate_asset(
                        asset_type=asset_type,
                        index=i,
                        validate=validate_each,
                        rng=rng,
                        config=config,
                        locations=locations,
                        urban_map=urban_map,
                        region_map=region_map,
                        seasonality=seasonality,
                        city_base_prices=city_base_prices,
                        reference_time=reference_time,
                        location=loc,
                    )
                    batch_rows.append(asset)
                except Exception as e:
                    failures.append(i)
                    logger.exception("[ERROR] Asset %d generation/validation failed: %s", i, e)
                finally:
                    if pbar:
                        pbar.update(1)

            if batch_rows:
                dfs.append(pd.DataFrame(batch_rows))

    if not dfs:
        raise RuntimeError("No assets generated.")

    df = pd.concat(dfs, ignore_index=True)
    for col in ("location", "zone", "region", "urban_type", "energy_class",
            "orientation", "view", "condition", "heating"):
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Budget error: fail se superiamo la soglia
    fail_rate = len(failures) / float(n_rows)
    if fail_rate > error_budget_pct:
        raise RuntimeError(
            f"Generation failures {len(failures)}/{n_rows} ({fail_rate:.2%}) exceed budget {error_budget_pct:.2%}"
        )
    if failures:
        logger.warning("[DATASET] %d/%d records failed (%.2f%%).", len(failures), n_rows, 100.0 * fail_rate)

    # Location drift calculation (usa i pesi normalizzati)
    drift_info = compute_location_drift(
        df,
        target_weights=normalized_location_weights,
        tolerance=float(config.get("expected_profile", {}).get("location_distribution_tolerance", 0.05)),
    )
    df.attrs["location_drift_report"] = drift_info
    drifted = [loc for loc, info in drift_info.items() if info.get("drifted")]
    if drifted:
        logger.warning("[DRIFT] Locations drifted: %s", drifted)
    else:
        logger.info("[DRIFT] Location distribution within tolerance.")

    # Quality report (base + enriched)
    try:
        base_report = generate_base_quality_report(df)
        full_report = enrich_quality_report(df, base_report, config)
    except Exception as e:
        logger.warning("Quality/Drift enrichment skipped: %s", e)
        full_report = {"status": "degraded", "error": str(e)}

    full_report.setdefault("generation", {})

    # 2) prendi il seed preferendo quello passato via config (se presente)
    seed_val = None
    try:
        if isinstance(config, dict) and "seed" in config:
            seed_val = int(config["seed"])
    except Exception:
        seed_val = None
    
    full_report["generation"]["seed"] = seed_val
    
    # (opzionale ma utile per telemetria)
    if reference_time is not None:
        try:
            rt_iso = reference_time.replace(microsecond=0).isoformat().replace("+00:00", "Z")
            full_report["generation"]["reference_time"] = rt_iso
        except Exception:
            pass

    # Telemetria nel report
    elapsed = time.time() - t0
    full_report.setdefault("generation", {})
    full_report["generation"].update({
        "asset_type": asset_type,
        "n_rows_requested": n_rows,
        "n_rows_generated": int(len(df)),
        "n_failures": int(len(failures)),
        "fail_rate": float(fail_rate),
        "n_batches": int(n_batches),
        "batch_size": int(batch_size),
        "elapsed_sec": float(elapsed),
    })

    return df, full_report