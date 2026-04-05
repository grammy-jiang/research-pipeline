"""Venue reputation scoring.

Maps venue names to quality tiers using bundled CORE rankings data.
Falls back to a configurable penalty score for unknown venues.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Tier → numeric score mapping
TIER_SCORES: dict[str, float] = {
    "A*": 1.0,
    "A": 0.8,
    "B": 0.5,
    "C": 0.3,
}
UNKNOWN_SCORE = 0.1
PREPRINT_SCORE = 0.1

# Lazy-loaded venue data
_venue_data: dict[str, str] | None = None


def _load_venue_data(data_path: str = "") -> dict[str, str]:
    """Load venue-to-tier mapping from JSON.

    Args:
        data_path: Path to JSON file. Empty string uses bundled data.

    Returns:
        Dict mapping venue name (case-insensitive key) to tier string.
    """
    global _venue_data  # noqa: PLW0603
    if _venue_data is not None:
        return _venue_data

    if data_path:
        path = Path(data_path)
    else:
        path = Path(__file__).parent / "data" / "core_rankings.json"

    if not path.exists():
        logger.warning("Venue data file not found: %s", path)
        _venue_data = {}
        return _venue_data

    with open(path) as fh:
        raw = json.load(fh)

    # Build case-insensitive lookup
    _venue_data = {k.lower(): v for k, v in raw.items()}
    logger.info("Loaded %d venue entries from %s", len(_venue_data), path)
    return _venue_data


def reset_venue_cache() -> None:
    """Reset the cached venue data (for testing)."""
    global _venue_data  # noqa: PLW0603
    _venue_data = None


def get_venue_tier(venue: str, data_path: str = "") -> str | None:
    """Look up the CORE tier for a venue name.

    Args:
        venue: Venue name (case-insensitive matching).
        data_path: Path to custom venue data JSON.

    Returns:
        Tier string ("A*", "A", "B", "C") or None if unknown.
    """
    if not venue:
        return None
    data = _load_venue_data(data_path)
    return data.get(venue.lower())


def venue_score(venue: str | None, data_path: str = "") -> float:
    """Compute venue reputation score.

    Args:
        venue: Venue name. None or empty treated as preprint.
        data_path: Path to custom venue data JSON.

    Returns:
        Score in [0, 1].
    """
    if not venue:
        return PREPRINT_SCORE

    tier = get_venue_tier(venue, data_path)
    if tier is None:
        return UNKNOWN_SCORE

    return TIER_SCORES.get(tier, UNKNOWN_SCORE)
