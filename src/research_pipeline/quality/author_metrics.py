"""Author credibility metrics.

Computes author-based quality signals from h-index and citation
metadata.  Uses max author h-index as the paper-level signal
(senior author heuristic).
"""

import logging
import math

logger = logging.getLogger(__name__)


def author_credibility(
    max_h_index: int | None,
    scale: int = 100,
) -> float:
    """Compute log-normalized author credibility score.

    Args:
        max_h_index: Maximum h-index among paper's authors (None = 0).
        scale: Reference h-index for normalization.

    Returns:
        Score in [0, 1].
    """
    h = max_h_index or 0
    return min(1.0, math.log(1 + h) / math.log(1 + scale))
