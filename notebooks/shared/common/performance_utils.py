from __future__ import annotations

"""
Performance & Memory Utilities (notebooks/shared)
-------------------------------------------------
Capabilities
- DataFrame memory profiling (total, per-row, by dtype).
- Micro-benchmarks for common ops (groupby, sort).
- dtype optimization suggestions (ints downcast, float64→float32, object→category).
- Simple index/selectivity heuristics for analytics workloads.

Non-goals
- No mutation during profiling; optimizations are suggestions only (see DtypeOptimizer).
- No external telemetry; pure in-process measurements.
"""

import logging
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from pandas.api import types as ptypes  # type: ignore

from notebooks.shared.common.constants import (
    ASSET_ID,
    ENERGY_CLASS,
    LOCATION,
    REGION,
    ZONE,
    VALUATION_K,
)

__all__ = ["DatasetProfiler", "DtypeOptimizer"]

logger = logging.getLogger(__name__)

# =============================================================================
# Tunables (can be lifted to a shared config if needed)
# =============================================================================

FLOAT_DOWNCAST_ATOL: float = 1e-6
FLOAT_DOWNCAST_RTOL: float = 1e-3

# Consider a column index-worthy when selectivity is in (10%, 95%) for analytics.
HIGH_SELECTIVITY_MIN: float = 0.10
HIGH_SELECTIVITY_MAX: float = 0.95

# Integer target ranges (inclusive), used for safe downcast checks.
INT_RANGES: Dict[str, Tuple[int, int]] = {
    "uint8": (0, 2**8 - 1),
    "uint16": (0, 2**16 - 1),
    "uint32": (0, 2**32 - 1),
    "int8": (-(2**7), 2**7 - 1),
    "int16": (-(2**15), 2**15 - 1),
    "int32": (-(2**31), 2**31 - 1),
}

# =============================================================================
# Helpers
# =============================================================================


def _dtype_itemsize_bytes(dtype: Any) -> int:
    """Best-effort itemsize in bytes for both NumPy and pandas extension dtypes."""
    try:
        # NumPy dtype or pandas nullable integer has .itemsize
        return int(getattr(dtype, "itemsize"))
    except Exception:
        pass
    # Heuristic fallback: common pandas extension types
    s = str(dtype).lower()
    if any(tok in s for tok in ("int64", "float64")):
        return 8
    if any(tok in s for tok in ("int32", "float32")):
        return 4
    if "int16" in s:
        return 2
    if "int8" in s:
        return 1
    # Default conservative
    return 8


def _safe_nunique_ratio(series: pd.Series) -> float:
    n = len(series)
    if n == 0:
        return 1.0
    try:
        return float(series.nunique(dropna=False)) / float(n)
    except Exception:
        return 1.0


# =============================================================================
# DatasetProfiler
# =============================================================================


class DatasetProfiler:
    """
    End-to-end DataFrame profiling and optimization suggestions.

    Methods:
        - profile(df): returns memory stats, perf micro-benchmarks, dtype suggestions, index hints.
    """

    def __init__(
        self,
        float_downcast_atol: float = FLOAT_DOWNCAST_ATOL,
        float_downcast_rtol: float = FLOAT_DOWNCAST_RTOL,
    ) -> None:
        self.float_downcast_atol = float_downcast_atol
        self.float_downcast_rtol = float_downcast_rtol

    # ---------- Public API ----------

    def profile(self, df: pd.DataFrame, groupby_observed: bool = True) -> Dict[str, Any]:
        """
        Execute a full profiling pass.

        Args:
            df: Input DataFrame.
            groupby_observed: Pass `observed=True` for categorical groupers (perf).

        Returns:
            Dict with 'memory', 'performance', 'dtype_optimization', 'index_suggestions', 'summary'.
        """
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            logger.warning("[PERF] Empty or invalid DataFrame in profile()")
            return {}

        mem = self._profile_memory(df)
        perf = self._profile_performance(df, groupby_observed)
        dtype_ops = self._suggest_dtype_ops(df)
        index_sugg = self._suggest_indexes(df)

        return {
            "memory": mem,
            "performance": perf,
            "dtype_optimization": dtype_ops,
            "index_suggestions": index_sugg,
            "summary": {},
        }

    # ---------- Memory ----------

    def _profile_memory(self, df: pd.DataFrame) -> Dict[str, Any]:
        total_bytes = int(df.memory_usage(deep=True).sum())
        total_mb = round(total_bytes / (1024**2), 2)
        per_row_kb = round((total_bytes / 1024) / max(len(df), 1), 3)

        by_dtype: Dict[str, Any] = {}
        for dt in df.dtypes.unique():
            cols = df.select_dtypes(include=[dt]).columns
            size = int(df[cols].memory_usage(deep=True).sum())
            by_dtype[str(dt)] = {
                "bytes": size,
                "mb": round(size / (1024**2), 2),
                "pct": round((size / total_bytes) * 100, 1) if total_bytes else 0.0,
                "n_columns": int(len(cols)),
            }

        logger.info(
            "[PERF] Memory usage snapshot",
            extra={"total_mb": total_mb, "per_row_kb": per_row_kb},
        )
        return {
            "total_bytes": total_bytes,
            "total_mb": total_mb,
            "per_row_kb": per_row_kb,
            "by_dtype": by_dtype,
        }

    # ---------- Micro-benchmarks ----------

    def _profile_performance(self, df: pd.DataFrame, groupby_observed: bool) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        # GroupBy mean on common dimensions if present
        gb_cols = [c for c in (LOCATION, ENERGY_CLASS) if c in df.columns]
        if gb_cols and VALUATION_K in df.columns:
            start = perf_counter()
            try:
                _ = df.groupby(gb_cols, observed=groupby_observed)[VALUATION_K].mean()
            finally:
                elapsed = (perf_counter() - start) * 1e3
            results["groupby"] = {
                "columns": gb_cols,
                "time_ms": round(elapsed, 2),
                "rows_per_sec": int(len(df) / (elapsed / 1e3)) if elapsed > 0 else None,
            }

        # Sort benchmark on valuation
        if VALUATION_K in df.columns:
            start = perf_counter()
            try:
                _ = df.sort_values(VALUATION_K)
            finally:
                elapsed = (perf_counter() - start) * 1e3
            results["sort"] = {
                "column": VALUATION_K,
                "time_ms": round(elapsed, 2),
                "rows_per_sec": int(len(df) / (elapsed / 1e3)) if elapsed > 0 else None,
            }

        return results

    # ---------- Optimization hints ----------

    def _suggest_dtype_ops(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        suggestions: Dict[str, Dict[str, Any]] = {}

        for col in df.columns:
            s = df[col]
            dt = s.dtype

            # Integer downcast (supports NumPy ints and pandas nullable Int*Dtype)
            if ptypes.is_integer_dtype(dt):
                opt = self._optimize_int(s)
                if opt:
                    suggestions[col] = opt

            # float64 → float32 (approximate equality under tolerances)
            elif ptypes.is_float_dtype(dt) and str(dt) == "float64":
                if self._can_downcast_float(s):
                    suggestions[col] = {
                        "current": "float64",
                        "target": "float32",
                        "reason": "precision within tolerance",
                        "memory_reduction_pct": 50,
                    }

            # object → category if low cardinality
            elif str(dt) == "object":
                ratio = _safe_nunique_ratio(s)
                if ratio < 0.50:
                    suggestions[col] = {
                        "current": "object",
                        "target": "category",
                        "reason": f"low cardinality ({ratio:.1%})",
                    }

        return suggestions

    def _optimize_int(self, s: pd.Series) -> Optional[Dict[str, Any]]:
        ser = s.dropna()
        if ser.empty:
            return None

        # Obtain min/max safely (works for pandas nullable ints)
        try:
            mn, mx = int(ser.min()), int(ser.max())
        except Exception:
            return None

        current_dtype = str(s.dtype)
        current_size = _dtype_itemsize_bytes(s.dtype)

        for target, (lo, hi) in INT_RANGES.items():
            if lo <= mn <= mx <= hi and current_dtype != target:
                target_size = _dtype_itemsize_bytes(np.dtype(target))
                red = int(round((1 - (target_size / max(current_size, 1))) * 100))
                return {
                    "current": current_dtype,
                    "target": target,
                    "reason": f"values in [{mn}, {mx}]",
                    "memory_reduction_pct": max(0, min(100, red)),
                }
        return None

    def _can_downcast_float(self, s: pd.Series) -> bool:
        try:
            s32 = s.astype("float32")
            return np.allclose(
                s.values, s32.values, equal_nan=True, atol=self.float_downcast_atol, rtol=self.float_downcast_rtol
            )
        except Exception:
            return False

    # ---------- Index/selectivity ----------

    def _suggest_indexes(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        n = len(df)

        for col in (ASSET_ID, LOCATION, ENERGY_CLASS, ZONE, REGION):
            if col not in df.columns:
                continue

            card = int(df[col].nunique(dropna=False)) if n else 0
            sel = float(card / n) if n else 0.0

            if col == ASSET_ID and card == n and n > 0:
                utility, reason = "excellent", "unique identifier"
            elif HIGH_SELECTIVITY_MIN < sel < HIGH_SELECTIVITY_MAX:
                utility, reason = "good", f"selectivity={sel:.1%}"
            elif sel >= HIGH_SELECTIVITY_MAX:
                utility, reason = "moderate", "nearly unique"
            else:
                utility, reason = "poor", f"selectivity={sel:.1%}"

            out.append(
                {
                    "column": col,
                    "cardinality": card,
                    "selectivity": round(sel, 3),
                    "utility": utility,
                    "reason": reason,
                }
            )

        order = {"excellent": 0, "good": 1, "moderate": 2, "poor": 3}
        return sorted(out, key=lambda x: order.get(x["utility"], 4))


# =============================================================================
# DtypeOptimizer
# =============================================================================


class DtypeOptimizer:
    """
    Apply dtype optimizations in-place or on a copy.

    Usage:
        profiler = DatasetProfiler()
        suggestions = profiler.profile(df)["dtype_optimization"]
        df_opt, report = DtypeOptimizer().apply(df, suggestions, inplace=False)
    """

    def __init__(self, validate_float: bool = True, atol: float = FLOAT_DOWNCAST_ATOL, rtol: float = FLOAT_DOWNCAST_RTOL) -> None:
        self.validate_float = validate_float
        self.atol = atol
        self.rtol = rtol

    def apply(
        self,
        df: pd.DataFrame,
        suggestions: Dict[str, Dict[str, Any]],
        inplace: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply dtype conversions based on `suggestions`.

        Args:
            df: Input DataFrame.
            suggestions: Mapping column -> optimization spec with 'target'.
            inplace: If True, mutate the original df; otherwise returns a copy.

        Returns:
            (optimized DataFrame, report dict with applied/skipped/failed/summary)
        """
        result = df if inplace else df.copy()
        report: Dict[str, List[Dict[str, Any]]] = {"applied": [], "skipped": [], "failed": []}

        for col, opt in suggestions.items():
            if col not in result.columns:
                report["skipped"].append({"column": col, "reason": "missing"})
                continue

            target = opt.get("target")
            if not target:
                report["skipped"].append({"column": col, "reason": "no target"})
                continue

            try:
                converted = result[col].astype(target)
                # Optional precision validation for float downcasts
                if self.validate_float and target == "float32":
                    if not np.allclose(
                        result[col].values, converted.values, equal_nan=True, atol=self.atol, rtol=self.rtol
                    ):
                        raise ValueError("precision loss beyond tolerance")
                result[col] = converted
                report["applied"].append({"column": col, "to": target})
                logger.info("[PERF] dtype optimized", extra={"column": col, "target": target})
            except Exception as e:
                report["failed"].append({"column": col, "error": str(e)})
                logger.warning("[PERF] dtype optimization failed", extra={"column": col, "error": str(e)})

        report["summary"] = {
            "total": len(suggestions),
            "applied": len(report["applied"]),
            "skipped": len(report["skipped"]),
            "failed": len(report["failed"]),
        }
        return result, report
