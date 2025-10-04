# notebooks/shared/common/asset_factory.py
from __future__ import annotations

"""
Asset factory & registry

Responsibilities
- Maintain a registry of asset generators and validators by asset_type.
- Normalize pricing config into the canonical structure (PricingConfigModel).
- Provide a single entry point to generate and (optionally) validate assets.
- Avoid mutating caller-provided kwargs and support rng injection for determinism.
"""

from typing import Any, Dict, Mapping, Optional, Protocol, runtime_checkable, Literal
import logging
from copy import deepcopy

from shared.common.schema import (
    get_all_fields,
    get_required_fields,
    normalize_column_order,  # to keep schema-first ordering
)
from shared.common.config import PricingConfigModel
from shared.common.constants import AssetType
from shared.n01_generate_dataset.asset_builder import generate_property

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

# --------------------------------------------------------------------------- #
# Protocols & registries
# --------------------------------------------------------------------------- #

@runtime_checkable
class AssetGenerator(Protocol):
    def __call__(self, *, index: int, **kwargs: Any) -> Dict[str, Any]: ...


@runtime_checkable
class AssetValidator(Protocol):
    def __call__(self, asset: Dict[str, Any]) -> Dict[str, Any]: ...


ASSET_GENERATORS: Dict[str, AssetGenerator] = {}
ASSET_VALIDATORS: Dict[str, AssetValidator] = {}


def register_asset_generator(name: str):
    """Decorator to register a new asset generator."""
    def decorator(fn: AssetGenerator) -> AssetGenerator:
        if name in ASSET_GENERATORS:
            logger.warning("Overwriting existing asset generator for '%s'", name)
        ASSET_GENERATORS[name] = fn
        return fn
    return decorator


def register_asset_validator(name: str):
    """Decorator to register a new asset validator."""
    def decorator(fn: AssetValidator) -> AssetValidator:
        if name in ASSET_VALIDATORS:
            logger.warning("Overwriting existing asset validator for '%s'", name)
        ASSET_VALIDATORS[name] = fn
        return fn
    return decorator


# --------------------------------------------------------------------------- #
# Pricing normalization
# --------------------------------------------------------------------------- #

def normalize_pricing_input(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    """
    Normalize the (possibly legacy) pricing section to the canonical schema
    enforced by PricingConfigModel.

    Accepts:
      - config["pricing"] (preferred)
      - legacy flat dicts with sub-keys like build_age.new_build, etc.

    Returns:
      A dict that matches PricingConfigModel().model_dump().
    """
    raw_pricing = {}
    if isinstance(config, Mapping):
        raw_pricing = dict(config.get("pricing", {}) or {})

    build_age = (raw_pricing.get("build_age", {}) or {})
    normalized = {
        "view_multipliers": raw_pricing.get("view_multipliers", {"sea": 1.0, "landmarks": 1.0}),
        "floor_modifiers": raw_pricing.get("floor_modifiers", {"is_top_floor": 0.0, "is_ground_floor": 0.0}),
        "build_age": {
            # Support both canonical and legacy keys
            "new": float(build_age.get("new", build_age.get("new_build", 0.0) or 0.0)),
            "recent": float(build_age.get("recent", build_age.get("recent_build", 0.0) or 0.0)),
            "old": float(build_age.get("old", build_age.get("old_build", 0.0) or 0.0)),
        },
        "energy_class_multipliers": dict(raw_pricing.get("energy_class_multipliers", {})),
        "state_modifiers": dict(raw_pricing.get("state_modifiers", {})),
        "extras": raw_pricing.get("extras", {"has_balcony": 0.0, "has_garage": 0.0, "has_garden": 0.0}),
    }

    try:
        model = PricingConfigModel(**normalized)
        return model.model_dump()
    except Exception as e:
        # Fall back to safe defaults if validation fails
        logger.warning("Pricing validation failed, using defaults. Details: %s", e)
        return PricingConfigModel().model_dump()


# --------------------------------------------------------------------------- #
# Public factory API
# --------------------------------------------------------------------------- #

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
    Generate an asset of the requested type using the registered generator.

    Args:
        asset_type: e.g., "property".
        index: progressive index used inside the generator.
        validate: whether to apply a registered validator after generation.
        rng: optional RNG (e.g., numpy Generator) for determinism.
        error_policy:
            - 'raise' (default): raise if validation fails.
            - 'warn' : log a warning and return the (unvalidated) asset.
            - 'skip' : log a warning and return {}.
        **kwargs: forwarded to the concrete generator.

    Returns:
        The generated asset dict (possibly validated), or {} if skipped.
    """
    if asset_type not in ASSET_GENERATORS:
        raise ValueError(f"[ASSET_FACTORY] Unsupported asset_type: '{asset_type}'")

    generator = ASSET_GENERATORS[asset_type]
    call_kwargs = deepcopy(kwargs)

    # Normalize pricing if a config is provided (do not override explicit pricing/pricing_input).
    cfg = call_kwargs.get("config")
    if cfg is not None:
        pricing_norm = normalize_pricing_input(cfg)
        call_kwargs.setdefault("pricing", pricing_norm)
        call_kwargs.setdefault("pricing_input", pricing_norm)

    # Inject rng if provided
    if rng is not None:
        call_kwargs["rng"] = rng

    asset = generator(index=index, **call_kwargs)

    if not validate:
        return asset

    try:
        asset = validate_asset(asset_type, asset)
        return asset
    except Exception as e:
        if error_policy == "raise":
            raise
        if error_policy == "warn":
            logger.warning("Validation failed for asset #%d (%s): %s", index, asset_type, e)
            return asset
        logger.warning("Validation failed for asset #%d (%s), skipping: %s", index, asset_type, e)
        return {}


def validate_asset(asset_type: str, asset: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply a registered validator for the asset_type, if available.
    Keeps the asset’s key order aligned with schema (required-first).
    """
    validator = ASSET_VALIDATORS.get(asset_type)
    if not validator:
        logger.debug("No validator registered for '%s' — returning as-is.", asset_type)
        return asset

    validated = validator(asset)

    # Reorder keys so required fields come first (stable, schema-first order).
    try:
        ordered = normalize_column_order(
            # Build a single-row DataFrame to reuse the ordering helper, then back to dict
            __import__("pandas").DataFrame([validated]),  # lazy import to avoid hard dep here
            asset_type=asset_type,
        )
        return dict(ordered.iloc[0])
    except Exception:
        # Fallback: return validated as-is if pandas/ordering is not available
        return validated


# --------------------------------------------------------------------------- #
# Default registrations for 'property'
# --------------------------------------------------------------------------- #

@register_asset_generator(AssetType.PROPERTY.value)
def generate_property_asset(*, index: int, **kwargs: Any) -> Dict[str, Any]:
    return generate_property(index=index, **kwargs)


@register_asset_validator(AssetType.PROPERTY.value)
def validate_property_asset(asset: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal contract validation:
    - Ensure required fields are present.
    - Log any extra fields (non-fatal).
    - Return the asset (no mutation other than potential ordering done upstream).
    """
    missing_required = set(get_required_fields("property")) - set(asset.keys())
    all_expected = set(get_all_fields("property"))
    extra_fields = set(asset.keys()) - all_expected

    if missing_required:
        raise ValueError(f"[VALIDATOR][property] missing required fields: {sorted(missing_required)}")
    if extra_fields:
        logger.info("[VALIDATOR][property] extra fields present: %s", sorted(extra_fields))

    return asset
