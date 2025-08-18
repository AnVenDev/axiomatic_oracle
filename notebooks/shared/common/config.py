# shared/config.py
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union, Literal

try:
    import yaml     # type: ignore
except Exception:   # pragma: no cover
    yaml = None

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator     # type: ignore   

from notebooks.shared.common.constants import (
    ASSET_ID, ASSET_TYPE, CONDITION_SCORE, LOCATION, ENERGY_CLASS, OWNER_OCCUPIED, PUBLIC_TRANSPORT_NEARBY, RISK_SCORE,
    SIZE_M2, ROOMS, BATHROOMS, YEAR_BUILT, FLOOR, BUILDING_FLOORS,
    HUMIDITY_LEVEL, TEMPERATURE_AVG, NOISE_LEVEL, AIR_QUALITY_INDEX,
    LUXURY_SCORE, ENV_SCORE, DISTANCE_TO_CENTER_KM,
    VALUATION_K, PREDICTION_TS, AGE_YEARS, AGE_CATEGORY, LUXURY_CATEGORY, VALUE_SEGMENT, TIMESTAMP
)

from notebooks.shared.common.constants import (
    DEFAULT_REGION_BY_CITY,
    DEFAULT_URBAN_TYPE_BY_CITY,
)

__all__ = [
    "ConfigurationError",
    "ViewMultipliers", "FloorModifiers", "BuildAgeModifiers", "EnergyClassMultipliers",
    "StateModifiers", "ExtrasModifiers", "PricingConfigModel",
    "ASSET_CONFIG",
    "DEFAULT_PRICING_MODEL", "DEFAULT_PRICING",
    "PathsConfig", "ExpectedProfile", "IncoherenceWeights", "IncoherenceConfig",
    "PriceCaps", "GenerationConfig", "PipelineConfig",
    "DEFAULT_CONFIG",
    "configure_logger", "load_config", "load_settings",
]

logger = logging.getLogger(__name__)


# ----------------------------- Errors --------------------------------------

class ConfigurationError(RuntimeError):
    """Raised when configuration is invalid or cannot be loaded."""


# ------------------------- Pricing submodels --------------------------------

class ViewMultipliers(BaseModel):
    sea: float = 1.25
    landmarks: float = 1.10


class FloorModifiers(BaseModel):
    is_top_floor: float = 0.08
    is_ground_floor: float = -0.08


class BuildAgeModifiers(BaseModel):
    new: float = 0.20
    recent: float = 0.05
    old: float = -0.15


class EnergyClassMultipliers(BaseModel):
    A: float = 1.15
    B: float = 1.08
    C: float = 1.00
    D: float = 0.95
    E: float = 0.90
    F: float = 0.85
    G: float = 0.80


class StateModifiers(BaseModel):
    new: float = 1.15
    renovated: float = 1.08
    good: float = 1.00
    needs_renovation: float = 0.85


class ExtrasModifiers(BaseModel):
    has_balcony: float = 0.04
    has_garage: float = 0.06
    has_garden: float = 0.05


class PricingConfigModel(BaseModel):
    view_multipliers: ViewMultipliers = Field(default_factory=ViewMultipliers)
    floor_modifiers: FloorModifiers = Field(default_factory=FloorModifiers)
    build_age: BuildAgeModifiers = Field(default_factory=BuildAgeModifiers)
    energy_class_multipliers: EnergyClassMultipliers = Field(default_factory=EnergyClassMultipliers)
    state_modifiers: StateModifiers = Field(default_factory=StateModifiers)
    extras: ExtrasModifiers = Field(default_factory=ExtrasModifiers)

    model_config = {"extra": "forbid"}


# default instance & dict
DEFAULT_PRICING_MODEL = PricingConfigModel()
DEFAULT_PRICING = DEFAULT_PRICING_MODEL.model_dump()


# ----------------------- Asset config for modeling --------------------------

ASSET_CONFIG = {
    "property": {
        "categorical": [
            "city",                   # <-- importante
            "region", "zone",
            "energy_class", "condition", "heating", "view",
            "public_transport_nearby"
        ],
        "numeric": [
            "size_m2","rooms","bathrooms","floor","building_floors",
            "has_elevator","has_garden","has_balcony","has_garage",
            "year_built","listing_month"
        ],
        "exclude": ["location"],     # eviti doppioni city/region/zone

        # ---- mapping e sinonimi (unificazione) ----
        # usa come base i DEFAULT_* del modulo constants; override vuoti di default
        "region_by_city": {**DEFAULT_REGION_BY_CITY},
        "urban_type_by_city": {**DEFAULT_URBAN_TYPE_BY_CITY},

        # sinonimi (lowercase -> CanonicalCase); estendi se ti serve
        "city_synonyms": {
            "milano":"Milan", "firenze":"Florence", "roma":"Rome",
            "torino":"Turin", "napoli":"Naples", "genova":"Genoa",
            "venezia":"Venice", "cagliari":"Cagliari", "verona":"Verona",
            "trieste":"Trieste", "padova":"Padua", "bari":"Bari",
            "catania":"Catania", "palermo":"Palermo"
        },
    }
}

# ------------------------- Higher-level config ------------------------------

class PathsConfig(BaseModel):
    output_path: Path = Path("../data/property_dataset_v2.csv")
    snapshot_dir: Path = Path("../data/snapshots")
    log_dir: Path = Path("../logs")

    @model_validator(mode="after")
    def normalize(self) -> "PathsConfig":
        # Normalize to resolve .. and user home if present
        self.output_path = self.output_path.expanduser().resolve()
        self.snapshot_dir = self.snapshot_dir.expanduser().resolve()
        self.log_dir = self.log_dir.expanduser().resolve()
        return self


class ExpectedProfile(BaseModel):
    location_distribution_tolerance: float = 0.05

    @field_validator("location_distribution_tolerance")
    @classmethod
    def tol_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("location_distribution_tolerance must be in [0,1]")
        return v


class IncoherenceWeights(BaseModel):
    condition: float = 0.5
    luxury: float = 0.3
    env: float = 0.2

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "IncoherenceWeights":
        s = self.condition + self.luxury + self.env
        if not math.isclose(s, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"Incoherence weights must sum to 1.0 (got {s:.6f})")
        return self


class IncoherenceConfig(BaseModel):
    val_threshold_quantile: float = 0.95
    confidence_thresh: float = 0.6
    weights: IncoherenceWeights = Field(default_factory=IncoherenceWeights)

    @field_validator("val_threshold_quantile", "confidence_thresh")
    @classmethod
    def prob_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("quantiles and thresholds must be in [0,1]")
        return v


class PriceCaps(BaseModel):
    max_multiplier: float = 3.0

    @field_validator("max_multiplier")
    @classmethod
    def positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("max_multiplier must be > 0")
        return v


class GenerationConfig(BaseModel):
    # data sizing & type
    n_rows: int = 15000
    asset_type: Literal["property"] = "property"
    generation_version: str = "v1.0"
    source_tag: str = "synthetic_with_priors"

    # location & pricing
    location_weights: Dict[str, float] = Field(default_factory=lambda: {
        "Milan": 0.20, "Rome": 0.18, "Turin": 0.08, "Naples": 0.08,
        "Bologna": 0.06, "Florence": 0.05, "Genoa": 0.05, "Palermo": 0.05,
        "Venice": 0.04, "Verona": 0.04, "Bari": 0.04, "Padua": 0.04,
        "Catania": 0.03, "Trieste": 0.03, "Cagliari": 0.03,
    })
    city_base_prices: Dict[str, Dict[str, float]] = Field(default_factory=lambda: {
    "Milan": {"center": 7500, "semi_center": 5000, "periphery": 3500},
    "Rome": {"center": 6000, "semi_center": 4000, "periphery": 2800},
    "Florence": {"center": 5500, "semi_center": 3800, "periphery": 2700},
    "Turin": {"center": 4500, "semi_center": 3200, "periphery": 2100},
    "Naples": {"center": 4000, "semi_center": 2800, "periphery": 1900},
    "Bologna": {"center": 4700, "semi_center": 3400, "periphery": 2300},
    "Genoa": {"center": 4200, "semi_center": 3000, "periphery": 2000},
    "Palermo": {"center": 3500, "semi_center": 2500, "periphery": 1700},
    "Venice": {"center": 6000, "semi_center": 4200, "periphery": 3000},
    "Verona": {"center": 4600, "semi_center": 3200, "periphery": 2200},
    "Bari": {"center": 3800, "semi_center": 2700, "periphery": 1800},
    "Padua": {"center": 4400, "semi_center": 3100, "periphery": 2100},
    "Catania": {"center": 3300, "semi_center": 2400, "periphery": 1600},
    "Trieste": {"center": 4100, "semi_center": 2900, "periphery": 1900},
    "Cagliari": {"center": 3700, "semi_center": 2600, "periphery": 1700},
    })
    zone_thresholds_km: Dict[str, float] = Field(default_factory=lambda: {"center": 1.5, "semi_center": 5.0})
    pricing: PricingConfigModel = Field(default_factory=PricingConfigModel)
    seasonality: Dict[int, float] = Field(default_factory=lambda: {
        1: 0.98, 2: 0.98, 3: 1.02, 4: 1.05, 5: 1.05, 6: 1.03,
        7: 1.00, 8: 0.97, 9: 1.02, 10: 1.01, 11: 0.99, 12: 0.97
    })

    # semantic domains
    urban_type_by_city: Dict[str, str] = Field(default_factory=lambda: {
        "Milan": "urban", "Rome": "urban", "Naples": "urban", "Florence": "urban",
        "Genoa": "urban", "Palermo": "urban", "Venice": "urban", "Verona": "urban", 
        "Bari": "urban", "Padua": "urban", "Catania": "urban", "Trieste": "urban", 
        "Cagliari": "urban",
    })

    # pipeline behavior
    groupby_observed: bool = True
    expected_profile: ExpectedProfile = Field(default_factory=ExpectedProfile)
    incoherence: IncoherenceConfig = Field(default_factory=IncoherenceConfig)
    price_caps: PriceCaps = Field(default_factory=PriceCaps)

    # paths
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @model_validator(mode="after")
    def validate_domains(self) -> "GenerationConfig":
        # location_weights sums to ~1
        total = sum(self.location_weights.values())
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"location_weights must sum to 1.0 (got {total:.6f})")

        # seasonality keys are 1..12
        if set(self.seasonality.keys()) != set(range(1, 13)):
            raise ValueError("seasonality must have integer keys 1..12")

        # city keys alignment (best-effort, not mandatory but recommended)
        missing_in_prices = set(self.location_weights) - set(self.city_base_prices)
        if missing_in_prices:
            logger.warning("Cities in location_weights missing from city_base_prices: %s", sorted(missing_in_prices))

        return self


class PipelineConfig(BaseModel):
    """Top-level container (ready to be extended in futuro)."""
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


# --------------- Defaults preserved for backward compatibility --------------

DEFAULT_CONFIG: Dict[str, Any] = PipelineConfig().to_dict()


# --------------------------- Logger configuration ---------------------------

_LEVEL_MAP: Mapping[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def _level_from_any(level: Union[int, str]) -> int:
    if isinstance(level, int):
        return level
    return _LEVEL_MAP.get(level.upper(), logging.INFO)


def configure_logger(
    level: Union[int, str] = logging.INFO,
    name: Optional[str] = None,
    json_format: Optional[bool] = None,
) -> logging.Logger:
    """Create or return a configured logger.

    Args:
        level: int or name (e.g., "INFO").
        name: logger name; None → root logger.
        json_format: force JSON formatting; if None, read env LOG_FORMAT=json|text.
    """
    lvl = _level_from_any(level)
    lg = logging.getLogger(name) if name else logging.getLogger()
    lg.setLevel(lvl)

    # avoid duplicate handlers of same type
    have_stream = any(isinstance(h, logging.StreamHandler) for h in lg.handlers)
    if not have_stream:
        handler = logging.StreamHandler()
        fmt = os.getenv("LOG_FORMAT", "").lower() if json_format is None else ("json" if json_format else "text")
        if fmt == "json":
            formatter = logging.Formatter(
                '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s",'
                '"msg":"%(message)s","module":"%(module)s","line":%(lineno)d}'
            )
        else:
            formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        lg.addHandler(handler)

    # If not root, prevent double emission up the chain
    if name:
        lg.propagate = False

    return lg


# -------------------------- Config loading helpers --------------------------

def _merge_deep(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep-merged copy of base ⟵ override (override wins)."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge_deep(out[k], v)
        else:
            out[k] = v
    return out


def _read_file_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    # Detect by extension first
    if path.suffix.lower() in {".json"}:
        return json.loads(text)
    if path.suffix.lower() in {".yml", ".yaml"}:
        if not yaml:
            raise ConfigurationError("PyYAML is not installed but a YAML config was provided.")
        return yaml.safe_load(text) or {}
    # Fallback: try JSON then YAML
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if not yaml:
            raise ConfigurationError("Unknown config format and PyYAML not available.")
        return yaml.safe_load(text) or {}


def _normalize_and_validate(raw_cfg: Dict[str, Any]) -> PipelineConfig:
    """Build a PipelineConfig from dict, normalizing nested structures."""
    # Normalize pricing section via model; then cast entire tree
    gen = raw_cfg.get("generation") or raw_cfg  # support legacy flat dict
    if "pricing" in gen and not isinstance(gen["pricing"], dict):
        raise ConfigurationError("generation.pricing must be an object")

    try:
        cfg = PipelineConfig.model_validate(raw_cfg if "generation" in raw_cfg else {"generation": gen})
    except ValidationError as e:
        raise ConfigurationError(f"Invalid configuration: {e}") from e
    return cfg


def load_config(path: Optional[Union[str, Path]] = "./dataset_config.yaml") -> Dict[str, Any]:
    """Backward-compatible loader that returns a dict (legacy API)."""
    base = DEFAULT_CONFIG
    if path is not None:
        p = Path(path)
        if p.exists():
            override = _read_file_config(p)
            merged = _merge_deep(base, override)
        else:
            merged = base
    else:
        merged = base
    cfg = _normalize_and_validate(merged)
    return cfg.to_dict()


def load_settings(
    path: Optional[Union[str, Path]] = "./dataset_config.yaml",
    ensure_dirs: bool = False,
) -> PipelineConfig:
    """Preferred loader: returns a strongly-typed PipelineConfig.

    Args:
        path: JSON/YAML file or None to use defaults.
        ensure_dirs: if True, create paths (snapshot_dir, log_dir, parent of output_path).
    """
    cfg_dict = load_config(path)
    settings = PipelineConfig.model_validate(cfg_dict)

    if ensure_dirs:
        settings.generation.paths.snapshot_dir.mkdir(parents=True, exist_ok=True)
        settings.generation.paths.log_dir.mkdir(parents=True, exist_ok=True)
        settings.generation.paths.output_path.parent.mkdir(parents=True, exist_ok=True)

    return settings