"""HTTP session management with shared session and User-Agent."""

import logging

import requests

from arxiv_paper_pipeline import __version__

logger = logging.getLogger(__name__)

_USER_AGENT = f"arxiv-paper-pipeline/{__version__}"


def create_session(contact_email: str = "") -> requests.Session:
    """Create a configured requests session for arXiv access.

    Args:
        contact_email: Contact email for the User-Agent header.

    Returns:
        A ``requests.Session`` with proper headers set.
    """
    session = requests.Session()
    ua = _USER_AGENT
    if contact_email:
        ua = f"{ua} (contact: {contact_email})"
    session.headers.update({"User-Agent": ua})
    logger.info("Created HTTP session with User-Agent: %s", ua)
    return session
