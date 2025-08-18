from __future__ import annotations

"""
Performance & Memory Utilities:
- Profilazione di memoria, performance e suggerimenti ottimizzazione dtypes
- Benchmark operazioni comuni (groupby, sort)
- Analisi selectivity colonne per suggerimento indici
"""

import logging
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np                          # type: ignore
import pandas as pd                         # type: ignore
from pandas.api import types as ptypes      # type: ignore

from notebooks.shared.common.constants import (
    ASSET_ID, ENERGY_CLASS, LOCATION, REGION, ZONE, VALUATION_K
)

logger = logging.getLogger(__name__)

__all__ = [
    "DatasetProfiler",
    "DtypeOptimizer",
]

# Configurazioni default (eventualmente spostabili in config.py)
FLOAT_DOWNCAST_ATOL = 1e-6
FLOAT_DOWNCAST_RTOL = 1e-3
HIGH_SELECTIVITY_MIN = 0.1
HIGH_SELECTIVITY_MAX = 0.95

INT_RANGES = {
    "uint8": (0, 2**8 - 1),
    "uint16": (0, 2**16 - 1),
    "uint32": (0, 2**32 - 1),
    "int8": (-2**7, 2**7 - 1),
    "int16": (-2**15, 2**15 - 1),
    "int32": (-2**31, 2**31 - 1),
}


class DatasetProfiler:
    """
    Profiling di memoria e performance, con suggerimenti di ottimizzazione.
    """

    def __init__(
        self,
        float_downcast_atol: float = FLOAT_DOWNCAST_ATOL,
        float_downcast_rtol: float = FLOAT_DOWNCAST_RTOL
    ) -> None:
        self.float_downcast_atol = float_downcast_atol
        self.float_downcast_rtol = float_downcast_rtol

    def profile(
        self,
        df: pd.DataFrame,
        groupby_observed: bool = True
    ) -> Dict[str, Any]:
        """
        Esegue profiling completo di un DataFrame.

        Args:
            df: DataFrame da analizzare.
            groupby_observed: Flag per ottimizzare GroupBy.

        Returns:
            Dizionario con statistiche di memoria, performance, dtype e indici.
        """
        if df.empty:
            logger.warning("[PERF] DataFrame vuoto in profile()")
            return {}

        return {
            "memory": self._profile_memory(df),
            "performance": self._profile_performance(df, groupby_observed),
            "dtype_optimization": self._suggest_dtype_ops(df),
            "index_suggestions": self._suggest_indexes(df),
            "summary": {},
        }

    def _profile_memory(self, df: pd.DataFrame) -> Dict[str, Any]:
        total_bytes = df.memory_usage(deep=True).sum()
        total_mb = total_bytes / 1024**2
        per_row_kb = (total_bytes / 1024) / max(len(df), 1)

        by_dtype: Dict[str, Any] = {}
        for dt in df.dtypes.unique():
            cols = df.select_dtypes(include=[dt]).columns
            size = df[cols].memory_usage(deep=True).sum()
            by_dtype[str(dt)] = {
                "bytes": int(size),
                "mb": round(size / 1024**2, 2),
                "pct": round(size / total_bytes * 100, 1),
                "n_columns": len(cols),
            }

        logger.info(
            "[PERF] Memory usage",
            extra={"total_mb": round(total_mb, 2), "per_row_kb": round(per_row_kb, 3)}
        )
        return {
            "total_bytes": int(total_bytes),
            "total_mb": round(total_mb, 2),
            "per_row_kb": round(per_row_kb, 3),
            "by_dtype": by_dtype,
        }

    def _profile_performance(
        self,
        df: pd.DataFrame,
        groupby_observed: bool
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        # GroupBy benchmark
        gb_cols = [c for c in (LOCATION, ENERGY_CLASS) if c in df]
        if gb_cols and VALUATION_K in df:
            start = perf_counter()
            _ = df.groupby(gb_cols, observed=groupby_observed)[VALUATION_K].mean()
            elapsed = (perf_counter() - start) * 1e3
            results["groupby"] = {
                "columns": gb_cols,
                "time_ms": round(elapsed, 2),
                "rows_per_sec": int(len(df) / (elapsed / 1e3)) if elapsed > 0 else None,
            }

        # Sort benchmark
        if VALUATION_K in df:
            start = perf_counter()
            _ = df.sort_values(VALUATION_K)
            elapsed = (perf_counter() - start) * 1e3
            results["sort"] = {
                "column": VALUATION_K,
                "time_ms": round(elapsed, 2),
                "rows_per_sec": int(len(df) / (elapsed / 1e3)) if elapsed > 0 else None,
            }

        return results

    def _suggest_dtype_ops(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        suggestions: Dict[str, Dict[str, Any]] = {}

        for col in df:
            series = df[col]
            dt = series.dtype

            # integer downcast
            if ptypes.is_integer_dtype(dt):
                opt = self._optimize_int(series)
                if opt:
                    suggestions[col] = opt

            # float64 -> float32
            elif ptypes.is_float_dtype(dt) and dt == np.float64:
                if self._can_downcast_float(series):
                    suggestions[col] = {
                        "current": "float64",
                        "target": "float32",
                        "reason": "precision within tolerance",
                        "memory_reduction_pct": 50,
                    }

            # object -> category
            elif dt == object:
                uniq_ratio = series.nunique(dropna=False) / len(series) if len(series) else 1
                if uniq_ratio < 0.5:
                    suggestions[col] = {
                        "current": "object",
                        "target": "category",
                        "reason": f"low cardinality ({uniq_ratio:.1%})",
                    }

        return suggestions

    def _optimize_int(self, s: pd.Series) -> Optional[Dict[str, Any]]:
        if s.dropna().empty:
            return None
        mn, mx = s.min(), s.max()
        for tgt, (lo, hi) in INT_RANGES.items():
            if lo <= mn <= mx <= hi and str(s.dtype) != tgt:
                red = (1 - np.dtype(tgt).itemsize / s.dtype.itemsize) * 100
                return {
                    "current": str(s.dtype),
                    "target": tgt,
                    "reason": f"values in [{mn}, {mx}]",
                    "memory_reduction_pct": int(red),
                }
        return None

    def _can_downcast_float(self, s: pd.Series) -> bool:
        try:
            s32 = s.astype("float32")
            return np.allclose(
                s, s32, equal_nan=True,
                atol=self.float_downcast_atol,
                rtol=self.float_downcast_rtol
            )
        except Exception:
            return False

    def _suggest_indexes(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        n = len(df)
        for col in (ASSET_ID, LOCATION, ENERGY_CLASS, ZONE, REGION):
            if col not in df:
                continue
            card = df[col].nunique(dropna=False)
            sel = card / n if n else 0
            if col == ASSET_ID and card == n:
                util, reason = "excellent", "unique identifier"
            elif HIGH_SELECTIVITY_MIN < sel < HIGH_SELECTIVITY_MAX:
                util, reason = "good", f"selectivity={sel:.1%}"
            elif sel >= HIGH_SELECTIVITY_MAX:
                util, reason = "moderate", "nearly unique"
            else:
                util, reason = "poor", f"selectivity={sel:.1%}"

            out.append({
                "column": col,
                "cardinality": int(card),
                "selectivity": round(sel, 3),
                "utility": util,
                "reason": reason,
            })

        order = {"excellent": 0, "good": 1, "moderate": 2, "poor": 3}
        return sorted(out, key=lambda x: order[x["utility"]])


class DtypeOptimizer:
    """
    Applica ottimizzazioni dtype in-place o su copia DataFrame.
    """

    def __init__(
        self,
        validate_float: bool = True,
        atol: float = FLOAT_DOWNCAST_ATOL,
        rtol: float = FLOAT_DOWNCAST_RTOL
    ) -> None:
        self.validate_float = validate_float
        self.atol = atol
        self.rtol = rtol

    def apply(
        self,
        df: pd.DataFrame,
        suggestions: Dict[str, Dict[str, Any]],
        inplace: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Applica le ottimizzazioni dtype suggerite.

        Args:
            df: DataFrame di input.
            suggestions: Mapping colonna → ottimizzazione.
            inplace: Se True modifica df originale.

        Returns:
            (DataFrame ottimizzato, report applicazioni)
        """
        result = df if inplace else df.copy()
        report = {"applied": [], "skipped": [], "failed": []}

        for col, opt in suggestions.items():
            if col not in result.columns:
                report["skipped"].append({"column": col, "reason": "missing"})
                continue
            tgt = opt.get("target")
            try:
                converted = result[col].astype(tgt)
                if self.validate_float and tgt == "float32":
                    if not np.allclose(
                        result[col], converted, equal_nan=True,
                        atol=self.atol, rtol=self.rtol
                    ):
                        raise ValueError("precision loss")
                result[col] = converted
                report["applied"].append({"column": col, "to": tgt})
                logger.info("[PERF] Ottimizzato dtype", extra={"column": col, "target": tgt})
            except Exception as e:
                report["failed"].append({"column": col, "error": str(e)})
                logger.warning("[PERF] Fallita ottimizzazione", extra={"column": col, "error": str(e)})

        report["summary"] = {
            "total": len(suggestions),
            "applied": len(report["applied"]),
            "skipped": len(report["skipped"]),
            "failed": len(report["failed"]),
        }
        return result, report