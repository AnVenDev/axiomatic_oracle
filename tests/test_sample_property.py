# tests/test_sample_property.py
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def test_sample_property_v2_exists_and_shape():
    import scripts.sample_property as sp
    importlib.reload(sp)

    assert isinstance(sp.sample_response, dict)
    d = sp.sample_response

    # chiavi principali v2
    for k in ("schema_version", "asset_id", "asset_type", "timestamp", "metrics", "flags", "model_meta", "model_health"):
        assert k in d

    # non vincoliamo l'esistenza fisica del file: basta sia una stringa
    assert isinstance(d["model_health"].get("model_path"), str)

def test_multiple_samples_v2_list():
    import scripts.sample_property as sp
    importlib.reload(sp)

    assert isinstance(sp.multiple_samples, list)
    assert len(sp.multiple_samples) >= 1
    assert isinstance(sp.multiple_samples[0], dict)
