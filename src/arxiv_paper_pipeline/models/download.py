"""Download manifest entry model."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DownloadManifestEntry(BaseModel):
    """Record of a single PDF download attempt."""

    arxiv_id: str = Field(description="Base arXiv ID.")
    version: str = Field(description="Version string.")
    pdf_url: str = Field(description="Source PDF URL.")
    local_path: str = Field(description="Local path where PDF was saved.")
    sha256: str = Field(description="SHA-256 hash of the downloaded file.")
    size_bytes: int = Field(description="File size in bytes.")
    downloaded_at: datetime = Field(description="Download timestamp (UTC).")
    status: Literal["downloaded", "skipped_exists", "failed"] = Field(
        description="Download outcome."
    )
    error: str | None = Field(default=None, description="Error message if failed.")
