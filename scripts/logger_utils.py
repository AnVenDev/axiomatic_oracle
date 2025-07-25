import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from hashlib import sha256

# Define log path and file
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs"
LOG_FILE = LOG_PATH / "published_assets.json"

DETAIL_DIR = LOG_PATH / "detail_reports"
DETAIL_DIR.mkdir(parents=True, exist_ok=True)


def log_asset_publication(data: dict):
    """
    Append a single asset publication to the JSON log file.
    Adds a UTC timestamp.
    """
    LOG_PATH.mkdir(parents=True, exist_ok=True)

    enriched_data = {**data, "logged_at": datetime.utcnow().isoformat()}

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
    LOG_PATH.mkdir(parents=True, exist_ok=True)

    enriched_results = [
        {**result, "logged_at": datetime.utcnow().isoformat()} for result in results
    ]

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(enriched_results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved publication results to {filename}")


def compute_file_hash(path: Path) -> str:
    """
    Compute SHA256 hash of a file's content.
    """
    with path.open("rb") as f:
        return sha256(f.read()).hexdigest()


def save_prediction_detail(prediction: dict) -> str:
    """
    Saves the full prediction response to a detail file and returns its SHA256 hash.
    """
    asset_id = prediction["asset_id"]
    detail_path = DETAIL_DIR / f"{asset_id}.json"
    with detail_path.open("w", encoding="utf-8") as f:
        json.dump(prediction, f, indent=2)

    return compute_file_hash(detail_path)
