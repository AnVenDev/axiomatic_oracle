from __future__ import annotations
import numpy as np                                          # type: ignore
import pandas as pd                                         # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin    # type: ignore

class PropertyDerivedFeatures(BaseEstimator, TransformerMixin):
    """
    Crea feature derivate + priors stabili:
      - log_size_m2, sqm_per_room, baths_per_100sqm
      - elev_x_floor, no_elev_high_floor
      - rooms_per_100sqm
      - city_zone_prior  (da city_base_prices)
      - region_index_prior (da region_index)
    """
    def __init__(
        self,
        city_base: dict[str, dict[str, float]] | None = None,
        region_index: dict[str, float] | None = None,
    ):
        self.city_base = {str(k).lower(): {str(zk).lower(): float(v) for zk, v in zv.items()}
                          for k, zv in (city_base or {}).items()}
        self.region_index = {str(k).lower(): float(v) for k, v in (region_index or {}).items()}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()

        # ----- basic numeriche -----
        size = pd.to_numeric(df.get("size_m2"), errors="coerce")
        rooms = pd.to_numeric(df.get("rooms"), errors="coerce").replace(0, np.nan)
        baths = pd.to_numeric(df.get("bathrooms"), errors="coerce")
        floor = pd.to_numeric(df.get("floor"), errors="coerce").fillna(0)
        elev  = pd.to_numeric(df.get("has_elevator"), errors="coerce").fillna(0)

        df["log_size_m2"] = np.log1p(size)
        df["sqm_per_room"] = size / rooms
        df["baths_per_100sqm"] = 100.0 * baths / size.replace(0, np.nan)
        df["elev_x_floor"] = elev * np.maximum(floor - 1, 0)
        df["no_elev_high_floor"] = (1 - elev) * (floor >= 3).astype(int)
        df["rooms_per_100sqm"] = 100.0 * rooms / size.replace(0, np.nan)

        # ----- priors da config -----
        # city/zone
        city = df.get("city")
        zone = df.get("zone")
        if city is not None and zone is not None:
            city_zone_vals = []
            for c, z in zip(city.astype(str), zone.astype(str)):
                v = self.city_base.get(c.lower(), {}).get(z.lower(), np.nan)
                city_zone_vals.append(v)
            df["city_zone_prior"] = np.array(city_zone_vals, dtype="float64")
        else:
            df["city_zone_prior"] = np.nan

        # region macro index
        reg = df.get("region")
        if reg is not None:
            df["region_index_prior"] = reg.astype(str).str.lower().map(self.region_index).astype("float64")
        else:
            df["region_index_prior"] = np.nan

        return df