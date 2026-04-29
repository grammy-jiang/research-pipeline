"""Phase E E06 — dossier validation."""

from __future__ import annotations

from research_pipeline.briefing.dossier import build_dossier, render_dossier
from research_pipeline.briefing.validate_dossier import (
    DEFAULT_MAX_LINKS,
    DEFAULT_MAX_WORDS,
    validate_dossier_markdown,
)
from tests.unit._dossier_fixtures import make_cluster


def _good_markdown() -> str:
    cluster = make_cluster()
    dossier = build_dossier(cluster, run_date="2026-04-29")
    md = render_dossier(dossier, run_date="2026-04-29")
    # Pad with words to clear the min-word floor.
    return md + "\n\n" + (" ".join(["context"] * 80))


def test_validator_passes_well_formed_dossier() -> None:
    md = _good_markdown()
    result = validate_dossier_markdown(md)
    assert result.passed, result.errors
    assert result.metrics["link_count"] >= 1


def test_validator_rejects_missing_section() -> None:
    md = _good_markdown().replace("## Open Questions", "## Other")
    result = validate_dossier_markdown(md)
    assert not result.passed
    assert any("Open Questions" in e for e in result.errors)


def test_validator_rejects_no_evidence_url() -> None:
    cluster = make_cluster()
    dossier = build_dossier(cluster, run_date="2026-04-29")
    md = render_dossier(dossier, run_date="2026-04-29")
    # Strip http(s) URLs.
    import re

    stripped = re.sub(r"https?://\S+", "x", md)
    stripped += "\n\n" + (" ".join(["pad"] * 80))
    result = validate_dossier_markdown(stripped)
    assert not result.passed
    assert any("evidence URL" in e for e in result.errors)


def test_validator_rejects_link_spam() -> None:
    md = _good_markdown()
    extra_links = "\n".join(
        f"- https://example.com/extra-{i}" for i in range(DEFAULT_MAX_LINKS + 5)
    )
    md_overlong = md + "\n\n" + extra_links
    result = validate_dossier_markdown(md_overlong)
    assert not result.passed
    assert any("link count" in e for e in result.errors)


def test_validator_rejects_overlong_dossier() -> None:
    md = _good_markdown() + ("\n" + " ".join(["pad"] * (DEFAULT_MAX_WORDS + 200)))
    result = validate_dossier_markdown(md)
    assert not result.passed
    assert any("word count" in e and "exceeds" in e for e in result.errors)


def test_validator_rejects_too_short_dossier() -> None:
    md = (
        "---\ntype: topic-dossier\ntopic_id: t\n---\n"
        + "\n".join(
            f"## {s.removeprefix('## ')}\nbody"
            for s in [
                "## Agent Read Map",
                "## One-paragraph Summary",
                "## What Changed",
                "## Why It Matters Technically",
                "## Evidence Timeline",
                "## Artifacts To Open",
                "## Open Questions",
                "## Agent Notes",
            ]
        )
        + "\nfactuality_label=supported_fact https://example.com/x"
    )
    result = validate_dossier_markdown(md)
    assert not result.passed
    assert any("below minimum" in e for e in result.errors)


def test_validator_rejects_missing_factuality_label() -> None:
    md = _good_markdown().replace("factuality_label=supported_fact", "label=other")
    result = validate_dossier_markdown(md)
    assert not result.passed
    assert any("factuality" in e for e in result.errors)


def test_validator_rejects_multi_topic_frontmatter() -> None:
    md = _good_markdown().replace(
        "topic_id: topic_acme",
        "topic_id: topic_acme\ntopic_id: topic_other",
        1,
    )
    result = validate_dossier_markdown(md)
    assert not result.passed
    assert any("single topic" in e for e in result.errors)
