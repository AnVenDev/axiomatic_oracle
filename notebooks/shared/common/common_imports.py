"""
Common imports for **notebooks / EDA only**.
Do **NOT** import this module inside reusable `/shared` libraries or production code.

Exposes:
- Third-party: `np` (NumPy), `pd` (Pandas), `plt` (Matplotlib pyplot)
- Stdlib subset frequently used in notebooks: `Path` and selected typing aliases

Rationale:
- Keep notebooks concise while avoiding accidental heavy side-effects.
- Do not re-export optional/heavy libs (e.g., seaborn, yaml) as API to reduce coupling.
"""

# --- Stdlib (curated subset) ---
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set, Union, Callable, TypedDict

# --- Third-party (curated, no side-effects) ---
import numpy as np                  # type: ignore
import pandas as pd                 # type: ignore
import matplotlib.pyplot as plt     # type: ignore

# Optional: keep available for ad-hoc use in notebooks,
# but do **not** re-export to avoid accidental dependencies downstream.
try:
    import seaborn as sns  # noqa: F401  # type: ignore
except Exception:
    sns = None  # type: ignore

try:
    import yaml  # noqa: F401  # type: ignore
except Exception:
    yaml = None  # type: ignore

__all__ = [
    # typing
    "Any", "Dict", "List", "Tuple", "Optional", "Set", "Union", "Callable", "TypedDict",
    # stdlib
    "Path",
    # third-party
    "np", "pd", "plt",
]
