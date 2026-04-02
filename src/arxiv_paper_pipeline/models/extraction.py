"""Extraction models for structured content extraction from Markdown."""

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata for a single Markdown chunk."""

    paper_id: str = Field(description="arXiv paper ID.")
    section_path: str = Field(description="Hierarchical section path.")
    chunk_id: str = Field(description="Unique chunk identifier.")
    source_span: str = Field(description="Line range in the source Markdown.")
    token_count: int = Field(description="Approximate token count.")


class ExtractedClaim(BaseModel):
    """A claim extracted from a paper with evidence tracing."""

    claim: str = Field(description="The extracted claim text.")
    chunk_ids: list[str] = Field(description="Source chunk IDs supporting this claim.")
    confidence: float = Field(description="Extraction confidence (0-1).")


class MarkdownExtraction(BaseModel):
    """Structured extraction result from a converted Markdown paper."""

    arxiv_id: str = Field(description="Base arXiv ID.")
    version: str = Field(description="Version string.")
    chunks: list[ChunkMetadata] = Field(
        default_factory=list,
        description="Chunk index for the paper.",
    )
    claims: list[ExtractedClaim] = Field(
        default_factory=list,
        description="Extracted claims with evidence.",
    )
    sections: list[str] = Field(
        default_factory=list,
        description="Top-level section headings found.",
    )
