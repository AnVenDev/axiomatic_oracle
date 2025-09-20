from __future__ import annotations

"""
ML Preparation:
- Analisi robusta delle correlazioni col target (escludendo feature leaky)
- Strategie di encoding categoriale basate su soglie configurabili
- Validazioni input e gestione DataFrame vuoti
- Logging strutturato con dettagli sulle feature analizzate
"""

import logging
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np                                    # type: ignore
import pandas as pd                                   # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from notebooks.shared.common.constants import (
    VALUATION_K, PRICE_PER_SQM, PRICE_PER_SQM_VS_REGION_AVG,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MLPreparationAnalyzer",
]

ML_LEAKY_FEATURES: Set[str] = {
    PRICE_PER_SQM,
    PRICE_PER_SQM_VS_REGION_AVG,
    "_viz_price_per_sqm"
}


ENCODING_STRATEGY_THRESHOLDS = {
    "one_hot_max": 10,
    "target_encoding_min": 50
}

from sklearn.model_selection import GroupKFold # type: ignore

def train_test_split_grouped(X, y, groups, test_size=0.2, random_state=42):
    gkf = GroupKFold(n_splits=int(1/test_size))
    # prima split (deterministico con ordinamento gruppi)
    groups = pd.Series(groups).astype(str)
    first_group = sorted(groups.unique())[0]
    for train_idx, test_idx in gkf.split(X, y, groups):
        return train_idx, test_idx

def stable_feature_order(df: pd.DataFrame) -> list[str]:
    # ordina con numeriche prima (ord. alfabetico), poi categoriali/one-hot
    num = sorted(df.select_dtypes(include=[np.number]).columns)
    oth = sorted([c for c in df.columns if c not in num])
    return num + oth

def stratify_by_quantiles(y: pd.Series, q: int = 10) -> pd.Series:
    """Ritorna etichette di quantile per stratificare una regressione."""
    y_numeric = pd.to_numeric(y, errors="coerce")
    ranks = y_numeric.rank(method="first").to_numpy()
    bins = np.ceil(ranks / (len(y_numeric) / q)).astype(int)
    bins[~np.isfinite(y_numeric)] = -1  # NaN/inf vanno fuori strat
    return pd.Series(bins, index=y.index)

def train_val_test_split_regression(
    X: pd.DataFrame, y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    """Split con stratificazione per decili del target (regressione)."""
    strat = stratify_by_quantiles(y, q=10)
    X_tmp, X_test, y_tmp, y_test, s_tmp, _ = train_test_split(
        X, y, strat, test_size=test_size, random_state=random_state, stratify=strat
    )
    # Val dalla porzione temporanea
    strat_tmp = s_tmp.replace(-1, s_tmp.mode().iat[0])  # fallback se serve
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_size, random_state=random_state, stratify=strat_tmp
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

class MLPreparationAnalyzer:
    """
    Classe per l'analisi e preparazione del dataset per ML.

    Attributes:
        target_column: Nome della colonna target.
        leaky_features: Set di feature da escludere per evitare leakage.
    """

    def __init__(
        self,
        target_column: str = VALUATION_K,
        leaky_features: Optional[Set[str]] = None
    ) -> None:
        self.target_column = target_column
        self.leaky_features = leaky_features or ML_LEAKY_FEATURES

    def analyze_target_correlations(
        self,
        df: pd.DataFrame,
        top_n: int = 10
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Analizza le correlazioni tra feature numeriche e target, escludendo quelle leaky.

        Args:
            df: DataFrame di input.
            top_n: Numero di feature più correlate da restituire.

        Returns:
            correlations_df: DataFrame con feature e correlazioni.
            analysis_meta: Dizionario con dettagli analisi.
        """
        if df.empty:
            logger.warning("[MLPrep] DataFrame vuoto")
            return pd.DataFrame(), {"status": "empty_dataframe"}

        if self.target_column not in df.columns:
            logger.warning("[MLPrep] Target '%s' non presente", self.target_column)
            return pd.DataFrame(), {"status": "target_missing"}

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        clean_cols = [c for c in numeric_cols if c not in self.leaky_features]

        if self.target_column not in clean_cols:
            logger.warning("[MLPrep] Target '%s' non è numerico o è stato escluso", self.target_column)
            return pd.DataFrame(), {"status": "target_not_numeric"}

        correlations = df[clean_cols].corr()[self.target_column].drop(self.target_column)
        top_corr = correlations.abs().sort_values(ascending=False).head(top_n)

        result_df = pd.DataFrame({
            "feature": top_corr.index,
            "correlation": correlations[top_corr.index]
        })

        logger.info(
            "[MLPrep] Analizzate correlazioni per '%s' - Top %d feature: %s",
            self.target_column, top_n, list(top_corr.index)
        )

        return result_df, {
            "status": "success",
            "top_n": top_n,
            "excluded_features": list(self.leaky_features),
            "analyzed_features": clean_cols
        }

    def apply_encoding_strategies(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Applica encoding categoriale in base a soglie configurabili.

        Args:
            df: DataFrame di input.

        Returns:
            DataFrame con encoding applicato.
        """
        if df.empty:
            logger.warning("[MLPrep] DataFrame vuoto in apply_encoding_strategies")
            return df

        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            unique_count = df[col].nunique()
            if unique_count <= ENCODING_STRATEGY_THRESHOLDS["one_hot_max"]:
                df = pd.get_dummies(df, columns=[col], drop_first=True)
                logger.debug("[MLPrep] One-hot encoding applicato a %s", col)
            elif unique_count >= ENCODING_STRATEGY_THRESHOLDS["target_encoding_min"]:
                target_means = df.groupby(col)[self.target_column].mean()
                df[col] = df[col].map(target_means)
                logger.debug("[MLPrep] Target encoding applicato a %s", col)
            else:
                logger.debug("[MLPrep] Colonna %s lasciata invariata (unique_count=%d)", col, unique_count)

        return df