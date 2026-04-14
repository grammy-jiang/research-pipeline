"""HuggingFace daily papers search source.

Uses the HuggingFace Papers API to discover recent high-impact papers
featured on the HuggingFace daily papers page.  This is an optional
source that improves recency coverage for trending AI/ML papers.

API endpoint: ``https://huggingface.co/api/daily_papers``
Rate limit: conservative 2 req/s (no official limit published).
"""

import logging
import re
from datetime import UTC, datetime

import requests

from research_pipeline.infra.rate_limit import RateLimiter
from research_pipeline.infra.retry import retry
from research_pipeline.models.candidate import CandidateRecord

logger = logging.getLogger(__name__)

_HF_API_BASE = "https://huggingface.co/api"
_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})")


class HuggingFaceSource:
    """Search source backed by the HuggingFace daily papers API.

    Implements the ``SearchSource`` protocol.  Papers are matched against
    the query terms using simple keyword filtering on title and abstract.
    """

    def __init__(
        self,
        min_interval: float = 0.5,
        limit: int = 100,
        session: requests.Session | None = None,
    ) -> None:
        self._rate_limiter = RateLimiter(min_interval=min_interval, name="huggingface")
        self._limit = limit
        self._session = session or requests.Session()

    @property
    def name(self) -> str:
        return "huggingface"

    @retry(
        max_attempts=3,
        backoff_base=2.0,
        retryable_exceptions=(requests.RequestException,),
    )
    def _api_get(
        self, url: str, params: dict[str, str | int]
    ) -> list[dict[str, object]]:
        """Execute a rate-limited, retried GET request.

        Args:
            url: API endpoint URL.
            params: Query parameters.

        Returns:
            Parsed JSON response (list of paper entries).
        """
        self._rate_limiter.wait()
        response = self._session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    def search(
        self,
        topic: str,
        must_terms: list[str],
        nice_terms: list[str],
        max_results: int = 100,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[CandidateRecord]:
        """Search HuggingFace daily papers and filter by topic relevance.

        The HF daily papers API does not support query-based search, so
        we fetch recent papers and filter locally using must_terms and
        nice_terms against title and abstract.

        Args:
            topic: Raw topic string.
            must_terms: Terms that must appear (at least one must match).
            nice_terms: Boost terms (not required for matching).
            max_results: Maximum number of results to return.
            date_from: Date filter start (YYYY-MM-DD).
            date_to: Date filter end (YYYY-MM-DD).

        Returns:
            List of CandidateRecords with ``source="huggingface"``.
        """
        logger.info(
            "HuggingFace search: topic=%s, must_terms=%s (limit=%d)",
            topic,
            must_terms,
            self._limit,
        )

        url = f"{_HF_API_BASE}/daily_papers"
        params: dict[str, str | int] = {"limit": self._limit}

        try:
            data = self._api_get(url, params)
        except requests.RequestException as exc:
            logger.error("HuggingFace daily papers API failed: %s", exc)
            return []

        if not isinstance(data, list):
            logger.warning("Unexpected HuggingFace API response type: %s", type(data))
            return []

        # Parse date filters
        date_from_dt = _parse_date(date_from) if date_from else None
        date_to_dt = _parse_date(date_to) if date_to else None

        candidates: list[CandidateRecord] = []
        for entry in data:
            try:
                candidate = self._parse_entry(entry)
            except Exception as exc:
                paper_id = ""
                if isinstance(entry, dict):
                    paper = entry.get("paper", {})
                    if isinstance(paper, dict):
                        paper_id = str(paper.get("id", "?"))
                logger.warning(
                    "Failed to parse HuggingFace entry %s: %s", paper_id, exc
                )
                continue

            # Date filtering
            if date_from_dt and candidate.published < date_from_dt:
                continue
            if date_to_dt and candidate.published > date_to_dt:
                continue

            # Keyword filtering: at least one must_term must appear in
            # title or abstract (case-insensitive)
            if must_terms and not _matches_terms(candidate, must_terms):
                continue

            candidates.append(candidate)
            if len(candidates) >= max_results:
                break

        logger.info(
            "HuggingFace: %d candidates after filtering (from %d daily papers)",
            len(candidates),
            len(data),
        )
        return candidates

    def _parse_entry(self, entry: dict[str, object]) -> CandidateRecord:
        """Parse a HuggingFace daily paper entry into a CandidateRecord.

        Args:
            entry: Raw entry dict from the HF daily papers API.

        Returns:
            A CandidateRecord populated with HuggingFace data.
        """
        paper: dict[str, object] = entry.get("paper", {})  # type: ignore[assignment]
        if not isinstance(paper, dict):
            paper = {}

        paper_id = str(paper.get("id", ""))
        title = str(paper.get("title", ""))
        abstract = str(paper.get("summary", ""))

        # Extract authors
        authors_raw = paper.get("authors", [])
        authors: list[str] = []
        if isinstance(authors_raw, list):
            for author in authors_raw:
                if isinstance(author, dict):
                    name = author.get("name", "")
                    if name:
                        authors.append(str(name))
                elif isinstance(author, str):
                    authors.append(author)

        # Publication date
        pub_date_str = str(paper.get("publishedAt", ""))
        published = _parse_date(pub_date_str) if pub_date_str else datetime.now(UTC)

        # Determine arXiv ID from the paper ID field
        arxiv_id = ""
        match = _ARXIV_ID_RE.search(paper_id)
        if match:
            arxiv_id = match.group(1)
        if not arxiv_id:
            arxiv_id = (
                f"hf-{paper_id}"
                if paper_id
                else f"hf-{abs(hash(title.lower())) % 10**10}"
            )

        # Upvotes/comments available but not yet used in scoring
        # upvotes = entry.get("paper", {})
        # num_comments = entry.get("numComments", 0)

        pdf_url = ""
        abs_url = ""
        if arxiv_id and not arxiv_id.startswith("hf-"):
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
            abs_url = f"https://arxiv.org/abs/{arxiv_id}"

        return CandidateRecord(
            arxiv_id=arxiv_id,
            version="v1",
            title=title,
            authors=authors,
            published=published,
            updated=published,
            categories=[],
            primary_category="",
            abstract=abstract,
            abs_url=abs_url,
            pdf_url=pdf_url,
            source="huggingface",
        )


def _matches_terms(candidate: CandidateRecord, terms: list[str]) -> bool:
    """Check if at least one term appears in the candidate's title or abstract.

    Args:
        candidate: The candidate record to check.
        terms: Terms to match against (case-insensitive).

    Returns:
        True if at least one term matches.
    """
    text = (candidate.title + " " + candidate.abstract).lower()
    return any(term.lower() in text for term in terms)


def _parse_date(date_str: str) -> datetime:
    """Parse a date string into a timezone-aware datetime.

    Args:
        date_str: Date string (ISO 8601 or YYYY-MM-DD).

    Returns:
        Parsed datetime with UTC timezone.
    """
    # Handle ISO 8601 with timezone
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except ValueError:
        pass
    # Handle YYYY-MM-DD
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        return datetime.now(UTC)
