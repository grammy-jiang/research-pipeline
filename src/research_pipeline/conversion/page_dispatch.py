"""Tiered page dispatch: classify PDF pages by difficulty for conversion routing.

Inspects PDF page content to determine complexity:
- SIMPLE: text-only pages → fast backend (pymupdf4llm)
- MODERATE: pages with images or simple formatting → standard backend
- COMPLEX: pages with tables, math equations, or dense figures → high-quality backend

This enables per-page routing where simple pages are converted quickly
and complex pages get higher-quality (but slower) conversion.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)


class PageDifficulty(StrEnum):
    """Difficulty classification for a PDF page."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class PageClassification:
    """Classification result for a single PDF page."""

    page_number: int
    difficulty: PageDifficulty
    has_tables: bool = False
    has_math: bool = False
    has_images: bool = False
    text_density: float = 0.0
    reason: str = ""


@dataclass
class DocumentDispatchPlan:
    """Conversion routing plan for an entire document."""

    pdf_path: str
    total_pages: int
    pages: list[PageClassification] = field(default_factory=list)
    simple_count: int = 0
    moderate_count: int = 0
    complex_count: int = 0
    recommended_backend: str = "pymupdf4llm"

    @property
    def complexity_ratio(self) -> float:
        """Fraction of pages classified as COMPLEX."""
        if self.total_pages == 0:
            return 0.0
        return self.complex_count / self.total_pages

    @property
    def summary(self) -> dict[str, int]:
        """Page count by difficulty tier."""
        return {
            "simple": self.simple_count,
            "moderate": self.moderate_count,
            "complex": self.complex_count,
        }


# Math indicator patterns (LaTeX-like symbols in extracted text)
_MATH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\\(frac|sum|int|prod|lim|infty|partial)"),
    re.compile(r"\\(alpha|beta|gamma|delta|epsilon|theta|lambda|sigma)"),
    re.compile(r"\\(mathbb|mathcal|mathbf|mathrm)"),
    re.compile(r"\$[^$]+\$"),
    re.compile(r"\\begin\{(equation|align|gather|array)\}"),
    re.compile(r"[∑∏∫∂∞αβγδεθλσ≤≥≠≈∈∉⊂⊃∪∩]"),
]

# Table indicator patterns
_TABLE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\|[\s-]+\|"),
    re.compile(r"Table\s+\d+[.:]"),
    re.compile(r"\\begin\{tabular\}"),
    re.compile(r"\S+\s*&\s*\S+\s*&\s*\S+"),
]


def _has_math_content(text: str) -> bool:
    """Check if text contains mathematical notation."""
    return any(p.search(text) for p in _MATH_PATTERNS)


def _has_table_content(text: str) -> bool:
    """Check if text contains table-like structures."""
    return any(p.search(text) for p in _TABLE_PATTERNS)


def _compute_text_density(text: str, page_area: float) -> float:
    """Compute text density as characters per unit area.

    Args:
        text: Extracted text content.
        page_area: Page area in points squared.

    Returns:
        Text density ratio (0.0 to 1.0 scale, capped).
    """
    if page_area <= 0:
        return 0.0
    # Normalize: typical dense text page ~3000 chars on A4 (595x842 pts)
    typical_max = 4000.0
    density = len(text) / typical_max
    return min(density, 1.0)


def classify_page(
    page_number: int,
    text: str,
    image_count: int = 0,
    page_area: float = 501_490.0,  # A4 default
) -> PageClassification:
    """Classify a single page by difficulty.

    Args:
        page_number: Zero-based page index.
        text: Extracted text content from the page.
        image_count: Number of images on the page.
        page_area: Page area in points squared.

    Returns:
        PageClassification with difficulty and indicators.
    """
    has_math = _has_math_content(text)
    has_tables = _has_table_content(text)
    has_images = image_count > 0
    text_density = _compute_text_density(text, page_area)

    reasons: list[str] = []

    if has_math and has_tables:
        difficulty = PageDifficulty.COMPLEX
        reasons.append("math+tables")
    elif has_math:
        difficulty = PageDifficulty.COMPLEX
        reasons.append("math content")
    elif has_tables:
        difficulty = PageDifficulty.COMPLEX
        reasons.append("table content")
    elif has_images and image_count >= 3:
        difficulty = PageDifficulty.COMPLEX
        reasons.append(f"{image_count} images")
    elif has_images:
        difficulty = PageDifficulty.MODERATE
        reasons.append("images present")
    elif text_density < 0.1:
        difficulty = PageDifficulty.MODERATE
        reasons.append("low text density")
    else:
        difficulty = PageDifficulty.SIMPLE
        reasons.append("text-only")

    return PageClassification(
        page_number=page_number,
        difficulty=difficulty,
        has_tables=has_tables,
        has_math=has_math,
        has_images=has_images,
        text_density=text_density,
        reason=", ".join(reasons),
    )


def _recommend_backend(plan: DocumentDispatchPlan) -> str:
    """Recommend a conversion backend based on page complexity.

    Args:
        plan: The dispatch plan with page classifications.

    Returns:
        Backend name string.
    """
    if plan.total_pages == 0:
        return "pymupdf4llm"

    ratio = plan.complexity_ratio
    if ratio > 0.3:
        return "docling"
    elif ratio > 0.1 or plan.moderate_count > plan.simple_count:
        return "marker"
    return "pymupdf4llm"


def classify_document_from_text(
    pdf_path: str | Path,
    pages_text: list[str],
    pages_image_counts: list[int] | None = None,
) -> DocumentDispatchPlan:
    """Classify all pages of a document from pre-extracted text.

    This is the primary entry point when you already have page text
    (e.g., from a fast pymupdf text extraction pass).

    Args:
        pdf_path: Path to the PDF file.
        pages_text: List of text strings, one per page.
        pages_image_counts: Optional list of image counts per page.

    Returns:
        DocumentDispatchPlan with per-page classifications.
    """
    total = len(pages_text)
    if pages_image_counts is None:
        pages_image_counts = [0] * total

    classifications: list[PageClassification] = []
    simple = moderate = complex_ = 0

    for i, text in enumerate(pages_text):
        img_count = pages_image_counts[i] if i < len(pages_image_counts) else 0
        cls = classify_page(i, text, img_count)
        classifications.append(cls)

        if cls.difficulty == PageDifficulty.SIMPLE:
            simple += 1
        elif cls.difficulty == PageDifficulty.MODERATE:
            moderate += 1
        else:
            complex_ += 1

    plan = DocumentDispatchPlan(
        pdf_path=str(pdf_path),
        total_pages=total,
        pages=classifications,
        simple_count=simple,
        moderate_count=moderate,
        complex_count=complex_,
    )
    plan.recommended_backend = _recommend_backend(plan)

    logger.info(
        "Page dispatch for %s: %d pages — %d simple, %d moderate, %d complex → %s",
        Path(pdf_path).name,
        total,
        simple,
        moderate,
        complex_,
        plan.recommended_backend,
    )

    return plan


def classify_document(pdf_path: str | Path) -> DocumentDispatchPlan:
    """Classify a PDF document by extracting text with pymupdf.

    Falls back to classify_document_from_text with empty pages
    if pymupdf is not available.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        DocumentDispatchPlan with per-page classifications.
    """
    try:
        import pymupdf
    except ImportError:
        logger.warning("pymupdf not available, returning empty classification")
        return DocumentDispatchPlan(pdf_path=str(pdf_path), total_pages=0)

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.warning("PDF not found: %s", pdf_path)
        return DocumentDispatchPlan(pdf_path=str(pdf_path), total_pages=0)

    doc = pymupdf.open(str(pdf_path))
    pages_text: list[str] = []
    pages_images: list[int] = []

    for page in doc:
        pages_text.append(page.get_text())
        pages_images.append(len(page.get_images()))

    doc.close()

    return classify_document_from_text(pdf_path, pages_text, pages_images)
