"""Stage registry: maps stage names to execution functions."""

import logging

logger = logging.getLogger(__name__)

STAGE_ORDER = [
    "plan",
    "search",
    "screen",
    "download",
    "convert",
    "extract",
    "summarize",
]


def validate_stage_name(stage: str) -> bool:
    """Check if a stage name is valid.

    Args:
        stage: Stage name to validate.

    Returns:
        True if valid.
    """
    return stage in STAGE_ORDER
