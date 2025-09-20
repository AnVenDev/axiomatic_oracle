# shared/asset_factory.py
from __future__ import annotations

"""
Asset factory & registry:
- Registry per generatori e validatori per asset_type
- Normalizzazione pricing tramite PricingConfigModel (shared.config)
- Nessuna mutazione di kwargs; supporto a rng injection
"""

from typing import Dict, Any, Optional, Protocol, Mapping, runtime_checkable, Literal
import logging
from copy import deepcopy

from notebooks.shared.common.schema import get_all_fields, get_required_fields
from notebooks.shared.common.config import PricingConfigModel
from notebooks.shared.common.constants import AssetType
from notebooks.shared.n01_generate_dataset.asset_builder import generate_property

logger = logging.getLogger(__name__)

__all__ = [
    "ASSET_GENERATORS",
    "ASSET_VALIDATORS",
    "register_asset_generator",
    "register_asset_validator",
    "generate_asset",
    "validate_asset",
    "normalize_pricing_input",
]

@runtime_checkable
class AssetGenerator(Protocol):
    def __call__(self, *, index: int, **kwargs: Any) -> Dict[str, Any]: ...

@runtime_checkable
class AssetValidator(Protocol):
    def __call__(self, asset: Dict[str, Any]) -> Dict[str, Any]: ...

ASSET_GENERATORS: Dict[str, AssetGenerator] = {}
ASSET_VALIDATORS: Dict[str, AssetValidator] = {}

def register_asset_generator(name: str):
    def decorator(fn: AssetGenerator) -> AssetGenerator:
        if name in ASSET_GENERATORS:
            logger.warning("Sovrascrivo asset generator esistente per '%s'", name)
        ASSET_GENERATORS[name] = fn
        return fn
    return decorator

def register_asset_validator(name: str):
    def decorator(fn: AssetValidator) -> AssetValidator:
        if name in ASSET_VALIDATORS:
            logger.warning("Sovrascrivo asset validator esistente per '%s'", name)
        ASSET_VALIDATORS[name] = fn
        return fn
    return decorator

def normalize_pricing_input(config: Mapping[str, Any]) -> Dict[str, Any]:
    raw_pricing = dict(config.get("pricing", {})) if isinstance(config, Mapping) else {}
    build_age = raw_pricing.get("build_age", {}) or {}
    normalized = {
        "view_multipliers": raw_pricing.get("view_multipliers", {"sea": 1.0, "landmarks": 1.0}),
        "floor_modifiers": raw_pricing.get("floor_modifiers", {"is_top_floor": 0.0, "is_ground_floor": 0.0}),
        "build_age": {
            "new": build_age.get("new", build_age.get("new_build", 0.0)),
            "recent": build_age.get("recent", build_age.get("recent_build", 0.0)),
            "old": build_age.get("old", build_age.get("old_build", 0.0)),
        },
        "energy_class_multipliers": raw_pricing.get("energy_class_multipliers", {}),
        "state_modifiers": raw_pricing.get("state_modifiers", {}),
        "extras": raw_pricing.get("extras", {"has_balcony": 0.0, "has_garage": 0.0, "has_garden": 0.0}),
    }
    try:
        model = PricingConfigModel(**normalized)
    except Exception as e:
        logger.warning("Validazione pricing fallita, uso default. Dettagli: %s", e)
        model = PricingConfigModel()
    return model.model_dump()

def generate_asset(
    asset_type: str,
    *,
    index: int,
    validate: bool = True,
    rng: Optional[Any] = None,
    error_policy: Literal["raise", "warn", "skip"] = "raise",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Genera un asset del tipo richiesto usando il generatore registrato.
    error_policy:
      - 'raise' (default): solleva se validator fallisce
      - 'warn'  : log warning e ritorna asset non validato
      - 'skip'  : ritorna dict vuoto in caso di errore
    """
    if asset_type not in ASSET_GENERATORS:
        raise ValueError(f"[ASSET_FACTORY] Unsupported asset_type: '{asset_type}'")

    generator = ASSET_GENERATORS[asset_type]
    call_kwargs = deepcopy(kwargs)

    cfg = call_kwargs.get("config")
    if cfg is not None:
        pricing_norm = normalize_pricing_input(cfg)
        call_kwargs["pricing"] = pricing_norm
        call_kwargs["pricing_input"] = pricing_norm

    if rng is not None:
        call_kwargs["rng"] = rng

    asset = generator(index=index, **call_kwargs)

    if validate:
        try:
            asset = validate_asset(asset_type, asset)
        except Exception as e:
            if error_policy == "raise":
                raise
            if error_policy == "warn":
                logger.warning("Validation failed for asset #%d (%s): %s", index, asset_type, e)
                return asset
            logger.warning("Validation failed for asset #%d (%s), skipping: %s", index, asset_type, e)
            return {}

    return asset

def validate_asset(asset_type: str, asset: Dict[str, Any]) -> Dict[str, Any]:
    validator = ASSET_VALIDATORS.get(asset_type)
    if not validator:
        logger.debug("Nessun validator registrato per '%s' — skip.", asset_type)
        return asset
    return validator(asset)

@register_asset_generator(AssetType.PROPERTY.value)
def generate_property_asset(*, index: int, **kwargs: Any) -> Dict[str, Any]:
    return generate_property(index=index, **kwargs)

@register_asset_validator(AssetType.PROPERTY.value)
def validate_property_asset(asset: Dict[str, Any]) -> Dict[str, Any]:
    missing_required = set(get_required_fields("property")) - set(asset.keys())
    all_expected = set(get_all_fields("property"))
    extra_fields = set(asset.keys()) - all_expected

    if missing_required:
        raise ValueError(f"[VALIDATOR][property] missing required fields: {sorted(missing_required)}")
    if extra_fields:
        logger.info("[VALIDATOR][property] extra fields present: %s", sorted(extra_fields))

    asset = {k: asset[k] for k in sorted(asset.keys())}

    return asset