"""Unit tests for Phase C C05 wiki-link / backlink / idempotent helpers."""

from __future__ import annotations

import pytest

from research_pipeline.briefing.obsidian_links import (
    find_wikilinks,
    inject_backlinks,
    is_idempotent_update,
    make_wikilink,
    slugify,
)


def test_slugify_basic() -> None:
    assert slugify("Topic Alpha") == "Topic-Alpha"
    assert slugify("  spaces   here ") == "spaces-here"
    assert slugify("AI/ML & LLM") == "AI-ML-LLM"
    assert slugify("alpha_beta.gamma-1") == "alpha_beta.gamma-1"


def test_slugify_rejects_empty() -> None:
    with pytest.raises(ValueError):
        slugify("   ")
    with pytest.raises(ValueError):
        slugify("***")


def test_make_wikilink_plain_and_aliased() -> None:
    assert make_wikilink("Topic Alpha") == "[[Topic Alpha]]"
    assert make_wikilink("Topic Alpha", alias="alpha") == "[[Topic Alpha|alpha]]"


def test_make_wikilink_rejects_bad_chars() -> None:
    with pytest.raises(ValueError):
        make_wikilink("")
    with pytest.raises(ValueError):
        make_wikilink("a[b")
    with pytest.raises(ValueError):
        make_wikilink("foo", alias="")
    with pytest.raises(ValueError):
        make_wikilink("foo", alias="bar|baz")


def test_find_wikilinks_extracts_targets_and_aliases() -> None:
    text = "see [[Topic Alpha]] and [[Source X|sx]] but not [link](http://x)"
    assert find_wikilinks(text) == [("Topic Alpha", None), ("Source X", "sx")]


def test_inject_backlinks_appends_section() -> None:
    body = "# Doc\n\nbody text [[Topic Alpha]]\n"
    out = inject_backlinks(body, ["Topic Alpha", "Source X"])
    assert "[[Topic Alpha]]" in out  # original preserved
    assert "## Backlinks" in out
    assert "- [[Source X]]" in out
    # Deterministic alphabetical order:
    backlinks_section = out.split("## Backlinks", 1)[1]
    assert backlinks_section.index("Source X") < backlinks_section.index("Topic Alpha")


def test_inject_backlinks_is_idempotent() -> None:
    body = "# Doc\n\nbody [[A]]\n"
    once = inject_backlinks(body, ["A", "B"])
    twice = inject_backlinks(once, ["A", "B"])
    assert once == twice


def test_inject_backlinks_dedupes_and_accepts_existing_wiki_syntax() -> None:
    body = "# Doc\n\nbody\n"
    out = inject_backlinks(body, ["[[A]]", "A", "B"])
    # Only one A entry.
    assert out.count("- [[A]]") == 1
    assert "- [[B]]" in out


def test_inject_backlinks_with_empty_links_strips_existing_section() -> None:
    body = "# Doc\n\nbody [[A]]\n\n## Backlinks\n\n- [[stale]]\n"
    out = inject_backlinks(body, [])
    assert "## Backlinks" not in out
    assert "[[A]]" in out  # in-body wiki-link untouched


def test_is_idempotent_update() -> None:
    assert is_idempotent_update("abc", "abc")
    assert not is_idempotent_update("abc", "abd")


def test_inject_backlinks_does_not_rewrite_wikilinks() -> None:
    body = "Body has [[X|x]] and [[Y]] inside.\n"
    out = inject_backlinks(body, ["Z"])
    # Critical: existing wiki-links must not be converted to markdown links.
    assert "[[X|x]]" in out
    assert "[[Y]]" in out
    assert "](" not in out  # no markdown link characters appeared
