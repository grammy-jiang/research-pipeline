"""Tests for tiered page dispatch module."""

from __future__ import annotations

import pytest

from research_pipeline.conversion.page_dispatch import (
    DocumentDispatchPlan,
    PageClassification,
    PageDifficulty,
    _compute_text_density,
    _has_math_content,
    _has_table_content,
    _recommend_backend,
    classify_document,
    classify_document_from_text,
    classify_page,
)


# ---------------------------------------------------------------------------
# PageDifficulty enum tests
# ---------------------------------------------------------------------------
class TestPageDifficulty:
    def test_values(self) -> None:
        assert PageDifficulty.SIMPLE == "simple"
        assert PageDifficulty.MODERATE == "moderate"
        assert PageDifficulty.COMPLEX == "complex"

    def test_from_string(self) -> None:
        assert PageDifficulty("simple") == PageDifficulty.SIMPLE
        assert PageDifficulty("complex") == PageDifficulty.COMPLEX


# ---------------------------------------------------------------------------
# PageClassification tests
# ---------------------------------------------------------------------------
class TestPageClassification:
    def test_creation(self) -> None:
        pc = PageClassification(
            page_number=0,
            difficulty=PageDifficulty.SIMPLE,
            text_density=0.5,
            reason="text-only",
        )
        assert pc.page_number == 0
        assert pc.difficulty == PageDifficulty.SIMPLE
        assert pc.has_tables is False
        assert pc.has_math is False
        assert pc.has_images is False

    def test_complex_page(self) -> None:
        pc = PageClassification(
            page_number=3,
            difficulty=PageDifficulty.COMPLEX,
            has_tables=True,
            has_math=True,
            text_density=0.3,
            reason="math+tables",
        )
        assert pc.has_tables is True
        assert pc.has_math is True


# ---------------------------------------------------------------------------
# DocumentDispatchPlan tests
# ---------------------------------------------------------------------------
class TestDocumentDispatchPlan:
    def test_empty(self) -> None:
        plan = DocumentDispatchPlan(pdf_path="test.pdf", total_pages=0)
        assert plan.complexity_ratio == 0.0
        assert plan.summary == {
            "simple": 0,
            "moderate": 0,
            "complex": 0,
        }

    def test_complexity_ratio(self) -> None:
        plan = DocumentDispatchPlan(
            pdf_path="test.pdf",
            total_pages=10,
            simple_count=5,
            moderate_count=3,
            complex_count=2,
        )
        assert abs(plan.complexity_ratio - 0.2) < 1e-9

    def test_all_complex(self) -> None:
        plan = DocumentDispatchPlan(
            pdf_path="test.pdf",
            total_pages=5,
            complex_count=5,
        )
        assert plan.complexity_ratio == 1.0

    def test_summary(self) -> None:
        plan = DocumentDispatchPlan(
            pdf_path="test.pdf",
            total_pages=10,
            simple_count=6,
            moderate_count=3,
            complex_count=1,
        )
        assert plan.summary == {
            "simple": 6,
            "moderate": 3,
            "complex": 1,
        }


# ---------------------------------------------------------------------------
# Math detection tests
# ---------------------------------------------------------------------------
class TestHasMathContent:
    def test_no_math(self) -> None:
        assert _has_math_content("This is a normal text paragraph.") is False

    def test_latex_frac(self) -> None:
        assert _has_math_content(r"The formula \frac{a}{b} gives...") is True

    def test_latex_sum(self) -> None:
        assert _has_math_content(r"\sum_{i=1}^{n} x_i") is True

    def test_latex_integral(self) -> None:
        assert _has_math_content(r"\int_0^1 f(x) dx") is True

    def test_inline_dollar(self) -> None:
        assert _has_math_content("We have $x^2 + y^2 = r^2$.") is True

    def test_greek_letters_latex(self) -> None:
        assert _has_math_content(r"Let \alpha = 0.5") is True

    def test_unicode_math(self) -> None:
        assert _has_math_content("∑ f(x) = ∫ g(x)dx") is True

    def test_mathbb(self) -> None:
        assert _has_math_content(r"\mathbb{R}^n") is True

    def test_equation_env(self) -> None:
        assert _has_math_content(r"\begin{equation} E = mc^2 \end{equation}") is True

    def test_partial(self) -> None:
        assert _has_math_content(r"\partial f / \partial x") is True


# ---------------------------------------------------------------------------
# Table detection tests
# ---------------------------------------------------------------------------
class TestHasTableContent:
    def test_no_table(self) -> None:
        assert _has_table_content("Normal text paragraph.") is False

    def test_pipe_table(self) -> None:
        assert _has_table_content("| --- | --- |") is True

    def test_table_reference(self) -> None:
        assert _has_table_content("Table 1: Results summary") is True

    def test_table_colon(self) -> None:
        assert _has_table_content("Table 3. Performance metrics") is True

    def test_latex_tabular(self) -> None:
        assert _has_table_content(r"\begin{tabular}{|c|c|}") is True

    def test_ampersand_columns(self) -> None:
        assert _has_table_content("0.95 & 0.87 & 0.91") is True


# ---------------------------------------------------------------------------
# Text density tests
# ---------------------------------------------------------------------------
class TestComputeTextDensity:
    def test_zero_area(self) -> None:
        assert _compute_text_density("some text", 0.0) == 0.0

    def test_negative_area(self) -> None:
        assert _compute_text_density("text", -100.0) == 0.0

    def test_empty_text(self) -> None:
        assert _compute_text_density("", 501_490.0) == 0.0

    def test_normal_text(self) -> None:
        text = "x" * 2000
        density = _compute_text_density(text, 501_490.0)
        assert 0.0 < density < 1.0

    def test_dense_text_capped(self) -> None:
        text = "x" * 10000
        density = _compute_text_density(text, 501_490.0)
        assert density == 1.0


# ---------------------------------------------------------------------------
# Page classification tests
# ---------------------------------------------------------------------------
class TestClassifyPage:
    def test_simple_text(self) -> None:
        text = "This is a normal paragraph. " * 50
        cls = classify_page(0, text)
        assert cls.difficulty == PageDifficulty.SIMPLE
        assert cls.has_math is False
        assert cls.has_tables is False

    def test_math_page(self) -> None:
        text = r"The equation \frac{a}{b} and \sum_{i} x_i"
        cls = classify_page(0, text)
        assert cls.difficulty == PageDifficulty.COMPLEX
        assert cls.has_math is True

    def test_table_page(self) -> None:
        text = "Table 1: Results\n| --- | --- |\n| 0.9 | 0.8 |"
        cls = classify_page(0, text)
        assert cls.difficulty == PageDifficulty.COMPLEX
        assert cls.has_tables is True

    def test_math_and_tables(self) -> None:
        text = r"Table 1: \frac{a}{b} | --- | --- |"
        cls = classify_page(0, text)
        assert cls.difficulty == PageDifficulty.COMPLEX
        assert cls.has_math is True
        assert cls.has_tables is True
        assert "math+tables" in cls.reason

    def test_few_images(self) -> None:
        text = "Normal text. " * 30
        cls = classify_page(0, text, image_count=1)
        assert cls.difficulty == PageDifficulty.MODERATE
        assert cls.has_images is True

    def test_many_images(self) -> None:
        text = "Figure descriptions. " * 10
        cls = classify_page(0, text, image_count=5)
        assert cls.difficulty == PageDifficulty.COMPLEX
        assert "5 images" in cls.reason

    def test_low_density(self) -> None:
        cls = classify_page(0, "Hi")
        assert cls.difficulty == PageDifficulty.MODERATE
        assert "low text density" in cls.reason

    def test_empty_page(self) -> None:
        cls = classify_page(0, "")
        assert cls.difficulty == PageDifficulty.MODERATE

    def test_page_number_preserved(self) -> None:
        cls = classify_page(42, "x" * 500)
        assert cls.page_number == 42


# ---------------------------------------------------------------------------
# Backend recommendation tests
# ---------------------------------------------------------------------------
class TestRecommendBackend:
    def test_empty(self) -> None:
        plan = DocumentDispatchPlan(pdf_path="t.pdf", total_pages=0)
        assert _recommend_backend(plan) == "pymupdf4llm"

    def test_all_simple(self) -> None:
        plan = DocumentDispatchPlan(
            pdf_path="t.pdf",
            total_pages=10,
            simple_count=10,
        )
        assert _recommend_backend(plan) == "pymupdf4llm"

    def test_high_complexity(self) -> None:
        plan = DocumentDispatchPlan(
            pdf_path="t.pdf",
            total_pages=10,
            simple_count=3,
            moderate_count=2,
            complex_count=5,
        )
        assert _recommend_backend(plan) == "docling"

    def test_moderate_complexity(self) -> None:
        plan = DocumentDispatchPlan(
            pdf_path="t.pdf",
            total_pages=10,
            simple_count=5,
            moderate_count=3,
            complex_count=2,
        )
        assert _recommend_backend(plan) == "marker"

    def test_mostly_moderate(self) -> None:
        plan = DocumentDispatchPlan(
            pdf_path="t.pdf",
            total_pages=10,
            simple_count=2,
            moderate_count=7,
            complex_count=1,
        )
        assert _recommend_backend(plan) == "marker"

    def test_borderline(self) -> None:
        plan = DocumentDispatchPlan(
            pdf_path="t.pdf",
            total_pages=10,
            simple_count=8,
            moderate_count=1,
            complex_count=1,
        )
        assert _recommend_backend(plan) == "pymupdf4llm"


# ---------------------------------------------------------------------------
# Document classification from text tests
# ---------------------------------------------------------------------------
class TestClassifyDocumentFromText:
    def test_empty(self) -> None:
        plan = classify_document_from_text("test.pdf", [])
        assert plan.total_pages == 0
        assert plan.recommended_backend == "pymupdf4llm"

    def test_simple_doc(self) -> None:
        pages = ["Normal text content. " * 50] * 5
        plan = classify_document_from_text("test.pdf", pages)
        assert plan.total_pages == 5
        assert plan.simple_count == 5
        assert plan.recommended_backend == "pymupdf4llm"

    def test_complex_doc(self) -> None:
        pages = [r"\frac{a}{b} Table 1: | --- |" for _ in range(8)]
        plan = classify_document_from_text("math.pdf", pages)
        assert plan.total_pages == 8
        assert plan.complex_count == 8
        assert plan.recommended_backend == "docling"

    def test_mixed_doc(self) -> None:
        pages = [
            "Normal text. " * 50,
            r"Equation \sum_{i=1}^{n} x_i",
            "More normal text. " * 50,
            "Table 1: Results\n| --- | --- |",
            "Final section. " * 50,
        ]
        plan = classify_document_from_text("mixed.pdf", pages)
        assert plan.total_pages == 5
        assert plan.complex_count == 2
        assert plan.simple_count >= 2

    def test_with_image_counts(self) -> None:
        pages = ["Normal text. " * 50, "Figure page. " * 10]
        images = [0, 4]
        plan = classify_document_from_text("img.pdf", pages, images)
        assert plan.total_pages == 2
        assert plan.pages[1].has_images is True

    def test_none_image_counts(self) -> None:
        pages = ["Text. " * 50] * 3
        plan = classify_document_from_text("t.pdf", pages, None)
        assert plan.total_pages == 3

    def test_mismatched_image_counts(self) -> None:
        pages = ["Text. " * 50] * 5
        images = [0, 1]  # shorter than pages
        plan = classify_document_from_text("t.pdf", pages, images)
        assert plan.total_pages == 5

    def test_path_object(self) -> None:
        from pathlib import Path

        plan = classify_document_from_text(
            Path("/tmp/test.pdf"), ["Hello world. " * 50]
        )
        assert plan.pdf_path == "/tmp/test.pdf"


# ---------------------------------------------------------------------------
# classify_document (pymupdf-dependent) tests
# ---------------------------------------------------------------------------
class TestClassifyDocument:
    def test_missing_pymupdf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        original_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "pymupdf":
                raise ImportError("no pymupdf")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        plan = classify_document("/nonexistent.pdf")
        assert plan.total_pages == 0

    def test_nonexistent_file(self) -> None:
        try:
            import pymupdf  # noqa: F401
        except ImportError:
            pytest.skip("pymupdf not installed")
        plan = classify_document("/tmp/__nonexistent_pdf__.pdf")
        assert plan.total_pages == 0
