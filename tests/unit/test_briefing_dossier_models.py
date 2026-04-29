"""Phase E E01 — dossier models and factuality labels."""

from __future__ import annotations

import pytest

from research_pipeline.briefing.dossier import (
    DossierClaim,
    EvidenceTimelineEntry,
    FactualityLabel,
)


class TestFactualityLabel:
    def test_three_labels_only(self) -> None:
        labels = {label.value for label in FactualityLabel}
        assert labels == {
            "supported_fact",
            "inference",
            "speculation_or_watch_item",
        }

    def test_str_enum(self) -> None:
        assert FactualityLabel.SUPPORTED_FACT == "supported_fact"
        assert FactualityLabel("inference") is FactualityLabel.INFERENCE


class TestEvidenceTimelineEntry:
    def test_valid_cluster_event(self) -> None:
        entry = EvidenceTimelineEntry(
            date="2026-04-29",
            evidence_url="https://example.com/release",
            source_class="primary_artifact",
            note="Release note",
        )
        assert entry.origin == "cluster_event"

    def test_valid_topic_memory(self) -> None:
        entry = EvidenceTimelineEntry(
            date="2026-04-15",
            evidence_url="https://example.com/prior",
            source_class="primary_artifact",
            note="Prior reference",
            origin="topic_memory",
        )
        assert entry.origin == "topic_memory"

    def test_empty_date_rejected(self) -> None:
        with pytest.raises(ValueError, match="date"):
            EvidenceTimelineEntry(
                date="",
                evidence_url="https://example.com/x",
                source_class="primary_artifact",
                note="n",
            )

    def test_empty_url_rejected(self) -> None:
        with pytest.raises(ValueError, match="evidence_url"):
            EvidenceTimelineEntry(
                date="2026-04-29",
                evidence_url="",
                source_class="primary_artifact",
                note="n",
            )

    def test_non_url_rejected(self) -> None:
        with pytest.raises(ValueError, match="http"):
            EvidenceTimelineEntry(
                date="2026-04-29",
                evidence_url="not-a-url",
                source_class="primary_artifact",
                note="n",
            )

    def test_obsidian_url_accepted(self) -> None:
        entry = EvidenceTimelineEntry(
            date="2026-04-29",
            evidence_url="obsidian://open?vault=Research&file=topic_x",
            source_class="primary_artifact",
            note="vault link",
            origin="topic_memory",
        )
        assert entry.evidence_url.startswith("obsidian://")

    def test_frozen(self) -> None:
        entry = EvidenceTimelineEntry(
            date="2026-04-29",
            evidence_url="https://example.com/x",
            source_class="primary_artifact",
            note="n",
        )
        with pytest.raises(Exception):  # noqa: B017
            entry.note = "changed"  # type: ignore[misc]


class TestDossierClaim:
    def test_supported_fact_with_evidence(self) -> None:
        claim = DossierClaim(
            text="The model adds streaming output.",
            label=FactualityLabel.SUPPORTED_FACT,
            evidence_url="https://example.com/release-notes",
        )
        assert claim.label == FactualityLabel.SUPPORTED_FACT

    def test_supported_fact_without_evidence_rejected(self) -> None:
        with pytest.raises(ValueError, match="evidence_url"):
            DossierClaim(
                text="A claim",
                label=FactualityLabel.SUPPORTED_FACT,
            )

    def test_supported_fact_with_blank_evidence_rejected(self) -> None:
        with pytest.raises(ValueError, match="evidence_url"):
            DossierClaim(
                text="A claim",
                label=FactualityLabel.SUPPORTED_FACT,
                evidence_url="   ",
            )

    def test_inference_no_evidence_required(self) -> None:
        claim = DossierClaim(
            text="This likely affects long-context inference.",
            label=FactualityLabel.INFERENCE,
        )
        assert claim.evidence_url is None

    def test_speculation_no_evidence_required(self) -> None:
        claim = DossierClaim(
            text="May enable agentic workflows in future.",
            label=FactualityLabel.SPECULATION_OR_WATCH_ITEM,
        )
        assert claim.label == FactualityLabel.SPECULATION_OR_WATCH_ITEM

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(ValueError, match="text"):
            DossierClaim(
                text="   ",
                label=FactualityLabel.INFERENCE,
            )

    def test_non_http_evidence_rejected(self) -> None:
        with pytest.raises(ValueError, match="http"):
            DossierClaim(
                text="x",
                label=FactualityLabel.SUPPORTED_FACT,
                evidence_url="ftp://example.com/x",
            )

    def test_frozen(self) -> None:
        claim = DossierClaim(
            text="x",
            label=FactualityLabel.INFERENCE,
        )
        with pytest.raises(Exception):  # noqa: B017
            claim.text = "y"  # type: ignore[misc]
