"""
Lightweight common imports for notebooks / EDA (not for production).

Scope
- Convenience aliases to keep notebooks concise:
  - Third-party: `np` (NumPy), `pd` (Pandas), `plt` (Matplotlib Pyplot)
  - Stdlib: `Path` and a curated set of typing aliases
- Zero heavy/optional dependencies re-exported (to avoid coupling).
- No side effects (no global rcparams tweaks, no backend forcing).

Usage
- Import this module ONLY in notebooks or ad-hoc analysis scripts.
- Do NOT import from reusable libraries (/shared) or production code.

Design
- Dependency surface is explicit and minimal.
- Optional libs (seaborn, yaml) are imported lazily (if present) but NOT exported.
"""

from __future__ import annotations

# --- Stdlib (curated subset) ---
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypedDict,
    Union,
)

# --- Third-party (curated, no side effects) ---
import numpy as np            # type: ignore
import pandas as pd           # type: ignore
import matplotlib.pyplot as plt  # type: ignore

# Optional helpers for notebooks only (NOT exported):
# Kept available for occasional ad-hoc use without creating hard deps.
try:  # pragma: no cover
    import seaborn as sns  # noqa: F401  # type: ignore
except Exception:  # pragma: no cover
    sns = None  # type: ignore

try:  # pragma: no cover
    import yaml  # noqa: F401  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

# Explicit export list (keeps surface small and stable)
__all__ = [
    # typing
    "Any",
    "Callable",
    "Dict",
    "Iterable",
    "List",
    "Mapping",
    "Optional",
    "Sequence",
    "Set",
    "Tuple",
    "TypedDict",
    "Union",
    # stdlib
    "Path",
    # third-party
    "np",
    "pd",
    "plt",
]