"""Conversion manifest entry model."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ConvertManifestEntry(BaseModel):
    """Record of a single PDF-to-Markdown conversion attempt."""

    arxiv_id: str = Field(description="Base arXiv ID.")
    version: str = Field(description="Version string.")
    pdf_path: str = Field(description="Path to the source PDF.")
    pdf_sha256: str = Field(description="SHA-256 of the source PDF.")
    markdown_path: str = Field(description="Path to the output Markdown.")
    converter_name: str = Field(description="Converter backend name.")
    converter_version: str = Field(description="Converter backend version.")
    converter_config_hash: str = Field(description="Hash of converter configuration.")
    converted_at: datetime = Field(description="Conversion timestamp (UTC).")
    warnings: list[str] = Field(
        default_factory=list,
        description="Conversion warnings.",
    )
    status: Literal["converted", "skipped_exists", "failed"] = Field(
        description="Conversion outcome."
    )
    tier: Literal["rough", "fine"] = Field(
        default="rough",
        description="Conversion tier: rough (pymupdf4llm) or fine (high-quality).",
    )
    error: str | None = Field(default=None, description="Error message if failed.")
    retry_count: int = Field(default=0, description="Number of retry attempts made.")
    last_error: str | None = Field(
        default=None, description="Error from the last retry attempt."
    )
