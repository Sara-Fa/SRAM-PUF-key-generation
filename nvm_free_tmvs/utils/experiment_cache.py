"""Shared caching utilities for experiment processors.

Provides deterministic cache paths based on experiment configuration
and pickle-based save/load functions. All cache files are stored under
``experiments_cache/<subdir>/``.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

CACHE_BASE = Path(__file__).resolve().parent.parent / "experiments_cache"


def get_cache_path(subdir: str, config: dict, base: Path | None = None) -> Path:
    """Return a deterministic cache file path for *config*.

    The filename is derived from an MD5 hash of the JSON-serialised
    config dict so that identical configurations always map to the same
    file.  Pass *base* to override the default ``CACHE_BASE``.
    """
    d = (base or CACHE_BASE) / subdir
    d.mkdir(parents=True, exist_ok=True)
    config_str = json.dumps(config, sort_keys=True)
    h = hashlib.md5(config_str.encode()).hexdigest()[:10]
    return d / f"results_{h}.pkl"


def save_cache(path: Path, results: dict) -> None:
    """Pickle *results* to *path*, adding a UTC timestamp."""
    results["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(path, "wb") as f:
        pickle.dump(results, f)
    print(f"Cached to {path}")


def load_cache(path: Path) -> dict:
    """Load and return a previously cached result dict."""
    with open(path, "rb") as f:
        return pickle.load(f)
