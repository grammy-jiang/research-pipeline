"""Structure-first Markdown chunking by headings and token limits."""

import logging
import re

from research_pipeline.models.extraction import ChunkMetadata

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_APPROX_TOKENS_PER_CHAR = 0.25  # rough approximation


def _estimate_tokens(text: str) -> int:
    """Rough token count estimation from character count."""
    return max(1, int(len(text) * _APPROX_TOKENS_PER_CHAR))


def chunk_markdown(
    markdown_text: str,
    paper_id: str,
    max_tokens: int = 1500,
) -> list[tuple[ChunkMetadata, str]]:
    """Split Markdown into chunks by heading structure, then by token limit.

    First splits on headings to maintain section boundaries, then splits
    oversized sections into sub-chunks.

    Args:
        markdown_text: Full Markdown text.
        paper_id: arXiv paper ID for metadata.
        max_tokens: Maximum approximate tokens per chunk.

    Returns:
        List of (metadata, chunk_text) tuples.
    """
    lines = markdown_text.split("\n")
    sections: list[tuple[str, int, int]] = []  # (heading, start_line, end_line)

    heading_positions: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if match:
            heading_positions.append((i, match.group(2).strip()))

    if not heading_positions:
        # No headings: treat entire document as one section
        heading_positions = [(0, "document")]

    for idx, (pos, heading) in enumerate(heading_positions):
        end = (
            heading_positions[idx + 1][0]
            if idx + 1 < len(heading_positions)
            else len(lines)
        )
        sections.append((heading, pos, end))

    chunks: list[tuple[ChunkMetadata, str]] = []
    chunk_counter = 0

    for section_heading, start, end in sections:
        section_text = "\n".join(lines[start:end])
        section_tokens = _estimate_tokens(section_text)

        if section_tokens <= max_tokens:
            chunk_counter += 1
            meta = ChunkMetadata(
                paper_id=paper_id,
                section_path=section_heading,
                chunk_id=f"{paper_id}_chunk_{chunk_counter:03d}",
                source_span=f"L{start + 1}-L{end}",
                token_count=section_tokens,
            )
            chunks.append((meta, section_text))
        else:
            # Split large sections by paragraphs
            paragraphs: list[str] = []
            current: list[str] = []
            for line in lines[start:end]:
                if line.strip() == "" and current:
                    paragraphs.append("\n".join(current))
                    current = []
                else:
                    current.append(line)
            if current:
                paragraphs.append("\n".join(current))

            buffer: list[str] = []
            buffer_tokens = 0
            buffer_start = start

            for para in paragraphs:
                para_tokens = _estimate_tokens(para)
                if buffer_tokens + para_tokens > max_tokens and buffer:
                    chunk_counter += 1
                    chunk_text = "\n\n".join(buffer)
                    meta = ChunkMetadata(
                        paper_id=paper_id,
                        section_path=section_heading,
                        chunk_id=f"{paper_id}_chunk_{chunk_counter:03d}",
                        source_span=(
                            f"L{buffer_start + 1}"
                            f"-L{buffer_start + len(chunk_text.split(chr(10)))}"
                        ),
                        token_count=_estimate_tokens(chunk_text),
                    )
                    chunks.append((meta, chunk_text))
                    buffer = []
                    buffer_tokens = 0
                    buffer_start += len(chunk_text.split("\n"))
                buffer.append(para)
                buffer_tokens += para_tokens

            if buffer:
                chunk_counter += 1
                chunk_text = "\n\n".join(buffer)
                meta = ChunkMetadata(
                    paper_id=paper_id,
                    section_path=section_heading,
                    chunk_id=f"{paper_id}_chunk_{chunk_counter:03d}",
                    source_span=f"L{buffer_start + 1}-L{end}",
                    token_count=_estimate_tokens(chunk_text),
                )
                chunks.append((meta, chunk_text))

    logger.info(
        "Chunked paper %s into %d chunks (max_tokens=%d)",
        paper_id,
        len(chunks),
        max_tokens,
    )
    return chunks
