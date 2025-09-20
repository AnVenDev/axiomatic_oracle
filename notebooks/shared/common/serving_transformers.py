# notebooks/shared/common/serving_transformers.py
from __future__ import annotations
import numpy as np, pandas as pd                            # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin    # type: ignore

class GeoCanonizer(BaseEstimator, TransformerMixin):
    """Canonizza city/zone/region con fallback e lowercase."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        out = X.copy()
        if "city" not in out.columns and "location" in out.columns:
            out["city"] = out["location"]
        if "zone" not in out.columns:
            out["zone"] = "semi_center"
        if "region" not in out.columns:
            # lascia NaN: sarà riempito da region_index_prior o da mapping a valle
            out["region"] = pd.Series(index=out.index, dtype="object")
        for c in ("city","zone","region"):
            if c in out.columns:
                out[c] = out[c].astype(str).str.strip().str.lower()
        return out

class PriorsGuard(BaseEstimator, TransformerMixin):
    """
    Riempie/crea city_zone_prior e region_index_prior con fallback robusti.
    N.B. __init__ non modifica i parametri → clone-safe.
    """
    def __init__(self, city_base=None, region_index=None, zone_medians=None, global_cityzone_median=0.0):
        self.city_base = city_base if city_base is not None else {}
        self.region_index = region_index if region_index is not None else {"north":1.05,"center":1.00,"south":0.92}
        self.zone_medians = zone_medians if zone_medians is not None else {}
        self.global_cityzone_median = float(global_cityzone_median)

    def fit(self, X, y=None): return self

    def transform(self, X):
        out = X.copy()

        # --- city_zone_prior
        cz = pd.to_numeric(out.get("city_zone_prior"), errors="coerce") if "city_zone_prior" in out.columns else None
        if cz is None or cz.isna().all():
            ci = out.get("city", pd.Series(index=out.index, dtype=str)).astype(str).str.lower()
            zo = out.get("zone", pd.Series(index=out.index, dtype=str)).astype(str).str.lower()
            out["city_zone_prior"] = np.array([
                (self.city_base.get(c, {}).get(z)
                 if not pd.isna(self.city_base.get(c, {}).get(z, np.nan))
                 else self.zone_medians.get(z, self.global_cityzone_median))
                for c, z in zip(ci, zo)
            ], dtype="float64")
        else:
            mask = cz.isna()
            if mask.any():
                ci = out["city"].astype(str).str.lower()
                zo = out["zone"].astype(str).str.lower()
                out.loc[mask, "city_zone_prior"] = np.array([
                    (self.city_base.get(c, {}).get(z)
                     if not pd.isna(self.city_base.get(c, {}).get(z, np.nan))
                     else self.zone_medians.get(z, self.global_cityzone_median))
                    for c, z in zip(ci[mask], zo[mask])
                ], dtype="float64")

        # --- region_index_prior
        rip = pd.to_numeric(out.get("region_index_prior"), errors="coerce") if "region_index_prior" in out.columns else None
        if rip is None or rip.isna().all():
            out["region_index_prior"] = out["region"].astype(str).str.lower().map(self.region_index).astype("float64")
        else:
            mask = rip.isna()
            if mask.any():
                out.loc[mask, "region_index_prior"] = (
                    out.loc[mask, "region"].astype(str).str.lower().map(self.region_index).astype("float64")
                )
        return out