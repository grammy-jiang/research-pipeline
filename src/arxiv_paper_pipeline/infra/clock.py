"""UTC time utilities and date window calculations."""

import logging
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(UTC)


def format_arxiv_date(dt: datetime) -> str:
    """Format a datetime for arXiv submittedDate filter.

    Args:
        dt: Datetime to format.

    Returns:
        String in arXiv format ``YYYYMMDDTTTT`` (e.g. ``202401010000``).
    """
    return dt.strftime("%Y%m%d%H%M")


def date_window(
    months_back: int,
    reference: datetime | None = None,
) -> tuple[str, str]:
    """Calculate an arXiv date window ending at *reference*.

    Args:
        months_back: Number of months to look back.
        reference: End of the window (default: now UTC).

    Returns:
        Tuple of ``(from_date, to_date)`` in arXiv format.
    """
    ref = reference or utc_now()
    start = ref - timedelta(days=months_back * 30)
    from_date = format_arxiv_date(start)
    to_date = format_arxiv_date(ref)
    logger.debug(
        "Date window: %d months back → %s to %s", months_back, from_date, to_date
    )
    return from_date, to_date
