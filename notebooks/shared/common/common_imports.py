# shared/common_imports.py
"""
Common imports for notebooks/EDA ONLY.
Do NOT import this inside reusable /shared modules.

Provides: pd, np, plt and a curated subset of stdlib/typing names.
"""

# --- Stdlib (subset) ---
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set, Union, Callable, TypedDict

# --- Third-party (curated) ---
import numpy as np                  # type: ignore
import pandas as pd                 # type: ignore
import matplotlib.pyplot as plt     # type: ignore

# Keep heavy libs available but DO NOT execute side-effects.
# (Not re-exported by default to avoid accidental dependencies.)
try:
    import seaborn as sns  # noqa: F401 # type: ignore
except Exception:
    pass

try:
    import yaml  # noqa: F401 # type: ignore
except Exception:
    pass

__all__ = [
    "Any", "Dict", "List", "Tuple", "Optional", "Set", "Union", "Callable", "TypedDict", "Path",
    "np", "pd", "plt",
]
