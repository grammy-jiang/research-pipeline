"""Citation graph client using the Semantic Scholar API.

Provides functions to discover related papers via citation and reference
graphs.  The agent decides WHICH papers to expand; the package fetches
and returns results without filtering.
"""

import logging
from datetime import UTC, datetime

import requests

from research_pipeline.infra.rate_limit import RateLimiter
from research_pipeline.infra.retry import retry
from research_pipeline.models.candidate import CandidateRecord

logger = logging.getLogger(__name__)

_S2_API_BASE = "https://api.semanticscholar.org/graph/v1"

_CITATION_FIELDS = (
    "paperId,externalIds,title,abstract,year,authors,"
    "citationCount,influentialCitationCount,venue,publicationDate,openAccessPdf"
)


def _bm25_score_text(text: str, query_terms: list[str]) -> float:
    """Simple BM25-like term frequency scoring for hop pruning."""
    text_lower = text.lower()
    tokens = text_lower.split()
    if not tokens:
        return 0.0
    score = 0.0
    for term in query_terms:
        tf = tokens.count(term.lower())
        score += tf / (tf + 1.5)  # BM25-like saturation
    return score


class CitationGraphClient:
    """Client for Semantic Scholar citation/reference graph APIs.

    Args:
        api_key: Optional S2 API key for higher rate limits.
        rate_limiter: Rate limiter (shared with other S2 calls).
        session: HTTP session for API calls.
    """

    def __init__(
        self,
        api_key: str = "",
        rate_limiter: RateLimiter | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self._rate_limiter = rate_limiter or RateLimiter(
            min_interval=1.0, name="s2_citations"
        )
        self._session = session or requests.Session()
        if api_key:
            self._session.headers["x-api-key"] = api_key

    @retry(
        max_attempts=3,
        backoff_base=2.0,
        retryable_exceptions=(requests.RequestException,),
    )
    def _api_get(self, url: str, params: dict[str, str | int]) -> dict:  # type: ignore[type-arg]
        """Execute a rate-limited, retried GET request.

        Args:
            url: API endpoint URL.
            params: Query parameters.

        Returns:
            Parsed JSON response.
        """
        self._rate_limiter.wait()
        response = self._session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    def get_citations(self, paper_id: str, limit: int = 50) -> list[CandidateRecord]:
        """Fetch papers that cite the given paper.

        Args:
            paper_id: arXiv ID or Semantic Scholar paper ID.
            limit: Maximum number of citing papers to return.

        Returns:
            List of CandidateRecords for citing papers.
        """
        return self._fetch_graph(paper_id, "citations", limit)

    def get_references(self, paper_id: str, limit: int = 50) -> list[CandidateRecord]:
        """Fetch papers referenced by the given paper.

        Args:
            paper_id: arXiv ID or Semantic Scholar paper ID.
            limit: Maximum number of referenced papers to return.

        Returns:
            List of CandidateRecords for referenced papers.
        """
        return self._fetch_graph(paper_id, "references", limit)

    def fetch_related(
        self,
        paper_ids: list[str],
        direction: str = "both",
        limit_per_paper: int = 50,
        reference_boost: float = 1.0,
    ) -> list[CandidateRecord]:
        """Batch-fetch related papers for multiple seed papers.

        When ``direction="both"`` and ``reference_boost > 1.0``, references
        (backward links) receive a higher per-paper limit than citations
        (forward links).  Deep research shows backward references yield
        +10-20 pp higher recall for discovery because foundational papers
        curate high-quality reference lists.

        Args:
            paper_ids: List of paper IDs to expand.
            direction: One of "citations", "references", or "both".
            limit_per_paper: Max results per paper per direction (base).
            reference_boost: Multiplier for the reference limit when
                ``direction="both"``.  E.g. 2.0 fetches twice as many
                references as citations.  Ignored when direction is not
                "both".  Default 1.0 (equal treatment).

        Returns:
            Deduplicated list of related CandidateRecords.
        """
        seen_ids: set[str] = set()
        results: list[CandidateRecord] = []

        citation_limit = limit_per_paper
        reference_limit = limit_per_paper
        if direction == "both" and reference_boost > 1.0:
            reference_limit = int(limit_per_paper * reference_boost)
            logger.info(
                "Backward-preference: citations=%d, references=%d",
                citation_limit,
                reference_limit,
            )

        for pid in paper_ids:
            if direction in ("citations", "both"):
                for c in self.get_citations(pid, citation_limit):
                    key = c.arxiv_id
                    if key not in seen_ids:
                        seen_ids.add(key)
                        results.append(c)

            if direction in ("references", "both"):
                for c in self.get_references(pid, reference_limit):
                    key = c.arxiv_id
                    if key not in seen_ids:
                        seen_ids.add(key)
                        results.append(c)

        logger.info(
            "Fetched %d unique related papers for %d seed papers",
            len(results),
            len(paper_ids),
        )
        return results

    def bfs_expand(
        self,
        seed_ids: list[str],
        query_terms: list[str],
        max_depth: int = 2,
        limit_per_paper: int = 20,
        top_k_per_hop: int = 10,
        direction: str = "both",
    ) -> list[CandidateRecord]:
        """Breadth-first expansion of citation graph with BM25 pruning.

        At each hop, fetches citations/references and re-ranks using BM25
        to prune low-relevance branches. Deep research shows +24pp recall
        improvement over single-hop expansion.

        Args:
            seed_ids: Starting paper IDs for BFS.
            query_terms: Terms for BM25 relevance scoring at each hop.
            max_depth: Maximum hops from seed papers.
            limit_per_paper: Max results per paper per direction per hop.
            top_k_per_hop: Keep only top-k most relevant papers per hop.
            direction: "citations", "references", or "both".

        Returns:
            Deduplicated list of discovered CandidateRecords across all hops.
        """
        if not seed_ids:
            return []

        frontier: list[str] = list(seed_ids)
        seen_ids: set[str] = set(seed_ids)
        all_results: list[CandidateRecord] = []

        for depth in range(1, max_depth + 1):
            hop_candidates: list[CandidateRecord] = []

            for paper_id in frontier:
                if direction in ("citations", "both"):
                    for c in self.get_citations(paper_id, limit_per_paper):
                        if c.arxiv_id not in seen_ids:
                            hop_candidates.append(c)

                if direction in ("references", "both"):
                    for c in self.get_references(paper_id, limit_per_paper):
                        if c.arxiv_id not in seen_ids:
                            hop_candidates.append(c)

            # Deduplicate within this hop
            unique: dict[str, CandidateRecord] = {}
            for c in hop_candidates:
                if c.arxiv_id not in unique:
                    unique[c.arxiv_id] = c
            hop_candidates = list(unique.values())

            # Score and rank by BM25 relevance
            scored = [
                (c, _bm25_score_text(f"{c.title} {c.abstract}", query_terms))
                for c in hop_candidates
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            top_candidates = [c for c, _ in scored[:top_k_per_hop]]

            for c in top_candidates:
                seen_ids.add(c.arxiv_id)
                all_results.append(c)

            frontier = [c.arxiv_id for c in top_candidates]

            logger.info(
                "BFS depth %d: %d candidates found, %d kept after pruning",
                depth,
                len(hop_candidates),
                len(top_candidates),
            )

            if not frontier:
                logger.info("BFS terminated at depth %d: no frontier papers", depth)
                break

        logger.info(
            "BFS expansion complete: %d total papers across %d hops",
            len(all_results),
            min(max_depth, len(all_results) > 0 and max_depth or 0),
        )
        return all_results

    def _fetch_graph(
        self, paper_id: str, direction: str, limit: int
    ) -> list[CandidateRecord]:
        """Fetch citation or reference graph for a paper.

        Args:
            paper_id: Paper identifier.
            direction: "citations" or "references".
            limit: Maximum results.

        Returns:
            List of CandidateRecords.
        """
        # Resolve paper ID format for S2 API
        if paper_id and not paper_id.startswith(("s2-", "ARXIV:")):
            # Assume arXiv ID
            s2_id = f"ARXIV:{paper_id}"
        else:
            s2_id = paper_id

        url = f"{_S2_API_BASE}/paper/{s2_id}/{direction}"
        params: dict[str, str | int] = {
            "fields": (
                f"citingPaper.{_CITATION_FIELDS}"
                if direction == "citations"
                else f"citedPaper.{_CITATION_FIELDS}"
            ),
            "limit": min(limit, 1000),
        }

        try:
            data = self._api_get(url, params)
        except requests.RequestException as exc:
            logger.error(
                "Citation graph fetch failed for %s (%s): %s",
                paper_id,
                direction,
                exc,
            )
            return []

        paper_key = "citingPaper" if direction == "citations" else "citedPaper"
        items = data.get("data", [])

        candidates: list[CandidateRecord] = []
        for item in items[:limit]:
            paper = item.get(paper_key, {})
            if not paper or not paper.get("title"):
                continue
            try:
                candidate = self._parse_paper(paper)
                candidates.append(candidate)
            except Exception as exc:
                logger.warning("Failed to parse %s paper: %s", direction, exc)

        logger.info("Fetched %d %s for %s", len(candidates), direction, paper_id)
        return candidates

    def _parse_paper(self, paper: dict) -> CandidateRecord:  # type: ignore[type-arg]
        """Parse a Semantic Scholar paper dict into a CandidateRecord.

        Args:
            paper: Raw paper dict from S2 API.

        Returns:
            A CandidateRecord populated with S2 data.
        """
        external_ids = paper.get("externalIds") or {}
        arxiv_id = external_ids.get("ArXiv", "")
        doi = external_ids.get("DOI")

        if not arxiv_id:
            arxiv_id = f"s2-{paper.get('paperId', '')[:10]}"

        title = paper.get("title", "")
        abstract = paper.get("abstract") or ""
        year = paper.get("year")

        authors_raw = paper.get("authors") or []
        authors = [a.get("name", "") for a in authors_raw if a.get("name")]

        pub_date_str = paper.get("publicationDate")
        if pub_date_str:
            try:
                published = datetime.fromisoformat(pub_date_str).replace(tzinfo=UTC)
            except ValueError:
                published = datetime(year or 2024, 1, 1, tzinfo=UTC)
        elif year:
            published = datetime(year, 1, 1, tzinfo=UTC)
        else:
            published = datetime.now(UTC)

        open_access = paper.get("openAccessPdf") or {}
        pdf_url = open_access.get("url", "")
        if not pdf_url and arxiv_id and not arxiv_id.startswith("s2-"):
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        abs_url = ""
        if arxiv_id and not arxiv_id.startswith("s2-"):
            abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        elif doi:
            abs_url = f"https://doi.org/{doi}"

        venue = paper.get("venue") or ""

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
            source="semantic_scholar",
            doi=doi,
            semantic_scholar_id=paper.get("paperId"),
            citation_count=paper.get("citationCount"),
            influential_citation_count=paper.get("influentialCitationCount"),
            venue=venue if venue else None,
            year=year,
        )
