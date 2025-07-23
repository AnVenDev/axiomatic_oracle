import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Define log path and file
LOG_PATH = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_PATH / "published_assets.json"

def log_asset_publication(data: dict):
    """
    Append a single asset publication to the JSON log file.
    Adds a UTC timestamp.
    """
    LOG_PATH.mkdir(exist_ok=True)

    enriched_data = {
        **data,
        "logged_at": datetime.utcnow().isoformat()
    }

    if LOG_FILE.exists():
        with open(LOG_FILE, "r+", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
            existing.append(enriched_data)
            f.seek(0)
            json.dump(existing, f, indent=2, ensure_ascii=False)
    else:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([enriched_data], f, indent=2, ensure_ascii=False)

def save_publications_to_json(results: List[Dict], filename: Path = LOG_FILE):
    """
    Overwrite the JSON file with a list of publication results.
    Useful for saving batch results at once.
    """
    LOG_PATH.mkdir(exist_ok=True)
    
    enriched_results = [
        {**result, "logged_at": datetime.utcnow().isoformat()}
        for result in results
    ]

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(enriched_results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved publication results to {filename}")