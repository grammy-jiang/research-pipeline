"""Phase F source-expansion governance and side-by-side evaluation helpers.

These helpers do not poll any source. They only inspect registry entries and
already-produced report markdown to decide whether a new source can be enabled
under Phase F's "increase coverage without increasing noise" rule.
"""

from __future__ import annotations

import re
from collections import Counter

from pydantic import BaseModel, ConfigDict

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingSourceConfig,
    SourceClass,
)
from research_pipeline.briefing.registry import SourceRegistry

# Access methods that are safe to enable without explicit Phase F review,
# because they were sanctioned in earlier phases.
_PHASE_A_OR_B_ACCESS_METHODS: frozenset[AccessMethod] = frozenset(
    {
        AccessMethod.GITHUB_RELEASES,
        AccessMethod.RSS_ATOM,
        AccessMethod.HTML_SCRAPE,
        AccessMethod.MANUAL,
    }
)

# Source classes that, by Phase F default posture, must be disabled by default
# unless they have an explicit `last_reviewed_at` review record.
_DEFAULT_DISABLED_CLASSES: frozenset[SourceClass] = frozenset(
    {
        SourceClass.SOCIAL_SIGNAL,
        SourceClass.TECHNICAL_DISCUSSION,
        SourceClass.MEDIA_NEWS,
        SourceClass.VIDEO_AUDIO,
    }
)

_TOP_ITEM_RE = re.compile(r"^### \d+\.\s+", re.MULTILINE)
_LINK_RE = re.compile(r"\]\((https?://[^)]+)\)")


class SourceEvaluationResult(BaseModel):
    """Outcome of evaluating a single source registry entry for enablement."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    sanctioned: bool
    enabled_safely: bool
    reasons: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """Return True when the source can be safely added to a registry."""
        return self.sanctioned and self.enabled_safely


class ReportComparisonResult(BaseModel):
    """Outcome of comparing a baseline report to a with-new-source report."""

    model_config = ConfigDict(frozen=True)

    baseline_item_count: int
    candidate_item_count: int
    baseline_link_count: int
    candidate_link_count: int
    item_count_delta: int
    link_count_delta: int
    new_links: tuple[str, ...]
    removed_links: tuple[str, ...]
    noise_increase: bool
    coverage_increase: bool
    notes: tuple[str, ...] = ()


def evaluate_source_for_enablement(
    source: BriefingSourceConfig,
) -> SourceEvaluationResult:
    """Check that ``source`` complies with Phase F enablement preconditions."""
    governance_reasons: list[str] = []
    enablement_reasons: list[str] = []

    if not source.retention_policy.strip():
        governance_reasons.append("missing retention_policy")
    if not source.rate_limit_policy.strip():
        governance_reasons.append("missing rate_limit_policy")
    if not source.cadence.strip():
        governance_reasons.append("missing cadence")
    if source.fixture_path is None:
        governance_reasons.append("missing offline fixture_path")

    sanctioned_method = source.access_method in AccessMethod.__members__.values()
    if not sanctioned_method:
        governance_reasons.append("unsupported access_method")

    expansion = source.access_method not in _PHASE_A_OR_B_ACCESS_METHODS
    if expansion and source.last_reviewed_at is None and source.enabled:
        enablement_reasons.append(
            "expansion source enabled without last_reviewed_at review"
        )

    if (
        source.source_class in _DEFAULT_DISABLED_CLASSES
        and source.enabled
        and source.last_reviewed_at is None
    ):
        enablement_reasons.append(
            "social/discussion/media/video_audio class must be disabled by default"
        )

    if (
        source.access_method == AccessMethod.X_API
        and source.enabled
        and (source.last_reviewed_at is None or source.auth_required is False)
    ):
        enablement_reasons.append(
            "X/Twitter requires explicit policy review (last_reviewed_at) and "
            "auth_required=True before enablement"
        )

    return SourceEvaluationResult(
        source_id=source.source_id,
        sanctioned=sanctioned_method and not governance_reasons,
        enabled_safely=not enablement_reasons,
        reasons=tuple(governance_reasons + enablement_reasons),
    )


def evaluate_registry(registry: SourceRegistry) -> tuple[SourceEvaluationResult, ...]:
    """Evaluate every source in ``registry`` and return ordered results."""
    return tuple(evaluate_source_for_enablement(s) for s in registry.sources)


def assert_disabled_by_default(source: BriefingSourceConfig) -> None:
    """Raise ``ValueError`` if a Phase F expansion source is enabled by default.

    A source is considered an expansion source when its access method is not in
    the Phase A/B sanctioned set or when its source class is one of the default
    Phase F disabled classes. Such sources must either be ``enabled=False`` or
    carry an explicit ``last_reviewed_at`` review record.
    """
    expansion = source.access_method not in _PHASE_A_OR_B_ACCESS_METHODS
    requires_review = expansion or source.source_class in _DEFAULT_DISABLED_CLASSES
    if requires_review and source.enabled and source.last_reviewed_at is None:
        raise ValueError(
            f"source {source.source_id!r} must be disabled by default "
            "unless last_reviewed_at is set",
        )


def _count_top_items(markdown: str) -> int:
    return len(_TOP_ITEM_RE.findall(markdown))


def _extract_links(markdown: str) -> list[str]:
    return [match.group(1) for match in _LINK_RE.finditer(markdown)]


def compare_reports(
    baseline_md: str,
    candidate_md: str,
    *,
    max_item_growth_ratio: float = 1.5,
    max_link_growth_ratio: float = 2.0,
) -> ReportComparisonResult:
    """Compare two daily reports for coverage/noise impact.

    Returns a structured diff. ``noise_increase`` flags either an item-count or
    link-count blow-up beyond the configured ratios. ``coverage_increase`` is
    true when the candidate adds at least one new canonical link without
    triggering the noise threshold.
    """
    baseline_items = _count_top_items(baseline_md)
    candidate_items = _count_top_items(candidate_md)
    baseline_links = _extract_links(baseline_md)
    candidate_links = _extract_links(candidate_md)

    base_link_set = set(baseline_links)
    cand_link_set = set(candidate_links)
    new_links = tuple(sorted(cand_link_set - base_link_set))
    removed_links = tuple(sorted(base_link_set - cand_link_set))

    notes: list[str] = []

    def _ratio(numerator: int, denominator: int) -> float:
        if denominator == 0:
            return float("inf") if numerator > 0 else 0.0
        return numerator / denominator

    item_ratio = _ratio(candidate_items, baseline_items)
    link_ratio = _ratio(len(candidate_links), max(len(baseline_links), 1))

    noise = False
    if item_ratio > max_item_growth_ratio and candidate_items - baseline_items > 1:
        noise = True
        notes.append(
            f"item count grew {candidate_items - baseline_items} (ratio "
            f"{item_ratio:.2f} > {max_item_growth_ratio})"
        )
    if (
        link_ratio > max_link_growth_ratio
        and len(candidate_links) - len(baseline_links) > 2
    ):
        noise = True
        notes.append(
            f"link count grew {len(candidate_links) - len(baseline_links)} "
            f"(ratio {link_ratio:.2f} > {max_link_growth_ratio})"
        )

    coverage = bool(new_links) and not noise
    if coverage:
        notes.append(f"coverage increased by {len(new_links)} new link(s)")

    return ReportComparisonResult(
        baseline_item_count=baseline_items,
        candidate_item_count=candidate_items,
        baseline_link_count=len(baseline_links),
        candidate_link_count=len(candidate_links),
        item_count_delta=candidate_items - baseline_items,
        link_count_delta=len(candidate_links) - len(baseline_links),
        new_links=new_links,
        removed_links=removed_links,
        noise_increase=noise,
        coverage_increase=coverage,
        notes=tuple(notes),
    )


def summarize_source_mix(registry: SourceRegistry) -> dict[str, int]:
    """Return a histogram of enabled sources by ``source_class``."""
    counter: Counter[str] = Counter()
    for source in registry.enabled_sources():
        counter[source.source_class.value] += 1
    return dict(counter)
