"""Rate-limited, idempotent PDF downloader for arXiv papers."""

import logging
import tempfile
from pathlib import Path

import requests

from research_pipeline.arxiv.rate_limit import ArxivRateLimiter
from research_pipeline.infra.clock import utc_now
from research_pipeline.infra.hashing import sha256_file
from research_pipeline.infra.retry import retry
from research_pipeline.models.download import DownloadManifestEntry

logger = logging.getLogger(__name__)


@retry(
    max_attempts=3, backoff_base=2.0, retryable_exceptions=(requests.RequestException,)
)
def _fetch_pdf_bytes(
    session: requests.Session,
    pdf_url: str,
    timeout: int = 60,
) -> requests.Response:
    """Fetch PDF content with retry on transient failures.

    Args:
        session: HTTP session.
        pdf_url: URL to download.
        timeout: Request timeout in seconds.

    Returns:
        The HTTP response (streamed).
    """
    response = session.get(pdf_url, timeout=timeout, stream=True)
    response.raise_for_status()
    return response


def download_pdf(
    arxiv_id: str,
    version: str,
    pdf_url: str,
    output_dir: Path,
    session: requests.Session,
    rate_limiter: ArxivRateLimiter,
) -> DownloadManifestEntry:
    """Download a single PDF, with idempotency and atomic write.

    If the file already exists at the expected path, it is skipped.
    Downloads to a temp file first, then renames atomically.

    Args:
        arxiv_id: Base arXiv ID.
        version: Version string.
        pdf_url: URL to download.
        output_dir: Directory to save PDFs.
        session: HTTP session for the request.
        rate_limiter: Rate limiter for arXiv compliance.

    Returns:
        Download manifest entry recording the outcome.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{arxiv_id}{version}.pdf"
    target_path = output_dir / filename

    # Skip if already exists
    if target_path.exists():
        file_hash = sha256_file(target_path)
        size = target_path.stat().st_size
        logger.info("PDF already exists, skipping: %s", target_path)
        return DownloadManifestEntry(
            arxiv_id=arxiv_id,
            version=version,
            pdf_url=pdf_url,
            local_path=str(target_path),
            sha256=file_hash,
            size_bytes=size,
            downloaded_at=utc_now(),
            status="skipped_exists",
        )

    # Rate-limited download
    rate_limiter.wait()
    logger.info("Downloading PDF: %s → %s", pdf_url, target_path)

    try:
        response = _fetch_pdf_bytes(session, pdf_url)

        # Atomic write via temp file
        with tempfile.NamedTemporaryFile(
            dir=output_dir, suffix=".tmp", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)

        tmp_path.rename(target_path)
        file_hash = sha256_file(target_path)
        size = target_path.stat().st_size

        logger.info(
            "Downloaded %s (%d bytes, sha256=%s)",
            filename,
            size,
            file_hash[:16],
        )
        return DownloadManifestEntry(
            arxiv_id=arxiv_id,
            version=version,
            pdf_url=pdf_url,
            local_path=str(target_path),
            sha256=file_hash,
            size_bytes=size,
            downloaded_at=utc_now(),
            status="downloaded",
        )

    except Exception as exc:
        logger.error("Failed to download %s: %s", pdf_url, exc)
        # Clean up any partial temp file
        if "tmp_path" in locals():
            tmp_path.unlink(missing_ok=True)
        return DownloadManifestEntry(
            arxiv_id=arxiv_id,
            version=version,
            pdf_url=pdf_url,
            local_path=str(target_path),
            sha256="",
            size_bytes=0,
            downloaded_at=utc_now(),
            status="failed",
            error=str(exc),
        )


def download_batch(
    papers: list[dict[str, str]],
    output_dir: Path,
    session: requests.Session,
    rate_limiter: ArxivRateLimiter,
    max_downloads: int = 20,
) -> list[DownloadManifestEntry]:
    """Download a batch of PDFs with a capped limit.

    Args:
        papers: List of dicts with keys ``arxiv_id``, ``version``, ``pdf_url``.
        output_dir: Directory to save PDFs.
        session: HTTP session.
        rate_limiter: Rate limiter.
        max_downloads: Maximum number of new downloads (skips don't count).

    Returns:
        List of download manifest entries.
    """
    entries: list[DownloadManifestEntry] = []
    new_downloads = 0

    for paper in papers:
        if new_downloads >= max_downloads:
            logger.warning("Reached max download limit (%d), stopping", max_downloads)
            break

        entry = download_pdf(
            arxiv_id=paper["arxiv_id"],
            version=paper["version"],
            pdf_url=paper["pdf_url"],
            output_dir=output_dir,
            session=session,
            rate_limiter=rate_limiter,
        )
        entries.append(entry)

        if entry.status == "downloaded":
            new_downloads += 1

    logger.info(
        "Download batch complete: %d entries (%d new downloads)",
        len(entries),
        new_downloads,
    )
    return entries
