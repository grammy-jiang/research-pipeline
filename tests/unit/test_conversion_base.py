"""Tests for the shared converter-backend base helpers."""

from __future__ import annotations

import pytest

from research_pipeline.conversion.base import parse_arxiv_stem


class TestParseArxivStem:
    """The stem->(*arxiv_id*, *version*) split shared by all 9 backends (#124)."""

    def test_versioned_stem(self) -> None:
        assert parse_arxiv_stem("2401.00001v2") == ("2401.00001", "v2")

    def test_v1_stem(self) -> None:
        assert parse_arxiv_stem("2401.00001v1") == ("2401.00001", "v1")

    def test_unversioned_stem_defaults_to_v1(self) -> None:
        assert parse_arxiv_stem("2401.00001") == ("2401.00001", "v1")

    def test_trailing_v_without_digit_is_not_a_version(self) -> None:
        # ``stem[-1]`` is not a digit, so nothing is peeled off.
        assert parse_arxiv_stem("draftv") == ("draftv", "v1")

    def test_multi_digit_version_is_not_recognised(self) -> None:
        # Documents the pre-existing single-digit limitation (behaviour
        # preserved by the extraction): ``stem[-2]`` is "1", not "v".
        assert parse_arxiv_stem("2401.00001v10") == ("2401.00001v10", "v1")

    @pytest.mark.parametrize("stem", ["", "a", "7"])
    def test_short_stem_does_not_raise(self, stem: str) -> None:
        # The inlined ``stem[-2]`` indexing raised IndexError on <2-char
        # stems; the extracted helper guards the length instead.
        assert parse_arxiv_stem(stem) == (stem, "v1")

    def test_two_char_version_only_stem(self) -> None:
        # Preserves the original behaviour for a bare ``v<N>`` stem.
        assert parse_arxiv_stem("v3") == ("", "v3")
