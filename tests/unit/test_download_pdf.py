"""Tests for the download SSRF guard (issue #104).

A candidate-supplied ``pdf_url`` reaches ``session.get`` in
``download/pdf.py``; without a guard a poisoned candidate could point the
fetcher at an internal address (SSRF). ``validate_download_url`` rejects
non-http(s) schemes and private/internal IP-literal / ``localhost`` hosts
before any fetch. The guard is intentionally DNS-free so it stays offline and
does not break the "unit tests, no network" contract.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from research_pipeline.download.pdf import (
    UnsafeURLError,
    download_pdf,
    validate_download_url,
)


@pytest.mark.parametrize(
    "url",
    [
        "https://arxiv.org/pdf/2401.12345",
        "http://export.arxiv.org/pdf/2401.12345v1",
        "https://example.com/paper.pdf",
    ],
)
def test_validate_allows_public_http_urls(url: str) -> None:
    validate_download_url(url)  # must not raise


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x",
        "http://127.0.0.1:8080/x",
        "http://localhost/x",
        "http://[::1]/x",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://10.0.0.5/internal",
        "http://192.168.1.1/x",
        "http://172.16.0.1/x",
        "http://0.0.0.0/x",
    ],
)
def test_validate_blocks_private_and_internal_hosts(url: str) -> None:
    with pytest.raises(UnsafeURLError):
        validate_download_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "gopher://example.com/x",
        "data:text/plain,hi",
        "https:///no-host",
    ],
)
def test_validate_blocks_non_http_schemes_and_missing_host(url: str) -> None:
    with pytest.raises(UnsafeURLError):
        validate_download_url(url)


def test_download_pdf_blocks_unsafe_url_without_fetching(tmp_path: Path) -> None:
    """An unsafe URL is blocked before the session/rate-limiter are ever used."""
    session = Mock()
    rate_limiter = Mock()
    entry = download_pdf(
        arxiv_id="2401.99999",
        version="v1",
        pdf_url="http://169.254.169.254/latest/meta-data/",
        output_dir=tmp_path,
        session=session,
        rate_limiter=rate_limiter,
    )
    assert entry.status == "failed"
    assert "blocked" in (entry.error or "")
    # Guard short-circuits before any network / rate-limit work.
    session.get.assert_not_called()
    rate_limiter.wait.assert_not_called()
