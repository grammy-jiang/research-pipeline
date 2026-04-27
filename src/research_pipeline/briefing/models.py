"""Pydantic models for daily AI intelligence briefings."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class SourceClass(StrEnum):
    """Governed source classes used for ranking posture."""

    PRIMARY_ARTIFACT = "primary_artifact"
    ACADEMIC_SOURCE = "academic_source"
    IMPLEMENTATION_SOURCE = "implementation_source"
    TECHNICAL_DISCUSSION = "technical_discussion"
    SOCIAL_SIGNAL = "social_signal"
    MEDIA_NEWS = "media_news"
    NEWSLETTER = "newsletter"
    VIDEO_AUDIO = "video_audio"


class AccessMethod(StrEnum):
    """Supported briefing source access methods."""

    GITHUB_RELEASES = "github_releases"
    RSS_ATOM = "rss_atom"
    MANUAL = "manual"
    ARXIV = "arxiv"
    HACKER_NEWS = "hacker_news"
    HUGGINGFACE_PAPERS = "huggingface_papers"
    API = "api"


class SourcePolicy(StrEnum):
    """Source storage and use policy."""

    PUBLIC_OFFICIAL = "public_official"
    PUBLIC_CURATED = "public_curated"
    LOCAL_MANUAL = "local_manual"
    DISCUSSION_ONLY = "discussion_only"


class BriefingSourceConfig(BaseModel):
    """Registry entry for a single allowed briefing source."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    source_name: str
    source_class: SourceClass
    access_method: AccessMethod
    official_url: AnyUrl | None = None
    auth_required: bool = False
    rate_limit_policy: str = "polite"
    cadence: str = "daily"
    retention_policy: str = "metadata_only"
    allowed_raw_storage: bool = True
    trust_weight: float = Field(default=1.0, ge=0.0, le=5.0)
    noise_weight: float = Field(default=0.0, ge=0.0, le=5.0)
    enabled: bool = True
    last_reviewed_at: str | None = None
    max_events_per_run: int = Field(default=10, ge=0, le=100)
    tags: tuple[str, ...] = ()
    feed_url: AnyUrl | None = None
    repo_owner: str | None = None
    repo_name: str | None = None
    api_url: AnyUrl | None = None
    fixture_path: str | None = None
    query: str | None = None
    manual_items: tuple[ManualBriefingItem, ...] = ()

    @model_validator(mode="after")
    def validate_access_fields(self) -> BriefingSourceConfig:
        """Ensure each access method has the fields needed to poll it."""
        if self.access_method == AccessMethod.GITHUB_RELEASES:
            has_repo = self.repo_owner is not None and self.repo_name is not None
            if not has_repo and self.api_url is None and self.fixture_path is None:
                raise ValueError(
                    "github_releases sources require repo, api_url, or fixture_path"
                )
        if (
            self.access_method == AccessMethod.RSS_ATOM
            and self.feed_url is None
            and self.fixture_path is None
        ):
            raise ValueError("rss_atom sources require feed_url or fixture_path")
        if self.access_method == AccessMethod.MANUAL and not self.manual_items:
            raise ValueError("manual sources require at least one manual item")
        if (
            self.access_method
            in {
                AccessMethod.HACKER_NEWS,
                AccessMethod.HUGGINGFACE_PAPERS,
                AccessMethod.ARXIV,
            }
            and self.fixture_path is None
            and self.api_url is None
            and self.feed_url is None
            and self.query is None
        ):
            raise ValueError(
                f"{self.access_method} sources require fixture, URL, or query"
            )
        return self


class ManualBriefingItem(BaseModel):
    """A hand-curated source item that still passes normal governance."""

    model_config = ConfigDict(frozen=True)

    title: str
    url: AnyUrl
    summary_hint: str = ""
    published_at: str | None = None
    source_native_id: str | None = None
    item_type: str = "manual"


class BriefingRunMetadata(BaseModel):
    """Metadata for one daily briefing run."""

    model_config = ConfigDict(frozen=True)

    run_date: str
    brief_id: str
    workspace: str
    source_count: int = 0
    started_at: str
    completed_at: str | None = None
    status: Literal["running", "validated", "failed", "partial"] = "running"


class IntelligenceEvent(BaseModel):
    """Common normalized event for releases, feeds, papers, and curated items."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    source_name: str
    source_id: str
    source_type: SourceClass
    source_policy: SourcePolicy = SourcePolicy.PUBLIC_OFFICIAL
    item_type: str
    canonical_url: str
    title: str
    retrieved_at: str
    collection_method: AccessMethod
    content_hash: str
    dedup_key: str
    author_or_org: str | None = None
    published_at: str | None = None
    updated_at: str | None = None
    source_native_id: str | None = None
    identifiers: dict[str, str] = Field(default_factory=dict)
    summary_hint: str = ""
    excerpt: str = ""
    topics: tuple[str, ...] = ()
    relevance_score: float = 0.0
    novelty_score: float = 0.0
    artifact_links: tuple[str, ...] = ()
    inferred_entities: tuple[str, ...] = ()
    ranking_explanation: str = ""
    confidence: Literal["high", "medium", "low"] = "medium"
    evidence_type: Literal[
        "supported_fact", "inference", "speculation_or_watch_item"
    ] = "supported_fact"
    raw_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("title")
    @classmethod
    def title_must_not_be_empty(cls, value: str) -> str:
        """Reject empty event titles."""
        title = value.strip()
        if not title:
            raise ValueError("title must not be empty")
        return title


class BriefingCluster(BaseModel):
    """A ranked cluster of one or more duplicate/corroborating events."""

    model_config = ConfigDict(frozen=True)

    cluster_id: str
    title: str
    primary_event_id: str
    event_ids: tuple[str, ...]
    topic_ids: tuple[str, ...] = ()
    canonical_urls: tuple[str, ...]
    first_seen_at: str
    last_seen_at: str
    source_classes: tuple[SourceClass, ...]
    primary_artifact_present: bool
    novelty_type: Literal["new", "active", "cooling", "dormant", "resurfaced"] = "new"
    confidence: Literal["high", "medium", "low"] = "medium"
    suggested_action: Literal["read", "try", "watch", "ignore"] = "read"
    evidence_type: Literal[
        "supported_fact", "inference", "speculation_or_watch_item"
    ] = "supported_fact"
    novelty_score: float = 0.0
    authority_score: float = 0.0
    engineering_usefulness_score: float = 0.0
    personal_interest_score: float = 0.0
    hype_penalty: float = 0.0
    duplicate_penalty: float = 0.0
    fatigue_penalty: float = 0.0
    resurfaced_boost: float = 0.0
    rank_score: float = 0.0
    ranking_explanation: str = ""
    events: tuple[IntelligenceEvent, ...] = ()


class TopicMemory(BaseModel):
    """Durable topic state used for fatigue and resurfacing."""

    model_config = ConfigDict(frozen=True)

    topic_id: str
    name: str
    aliases: tuple[str, ...] = ()
    first_seen_at: str
    last_seen_at: str
    status: Literal["new", "active", "cooling", "dormant", "resurfaced"] = "new"
    summary: str = ""
    key_entities: tuple[str, ...] = ()
    canonical_clusters: tuple[str, ...] = ()
    obsidian_note: str = ""
    interest_score: float = 0.0
    fatigue_score: float = 0.0
    last_reported_at: str | None = None
    report_count_7d: int = 0
    report_count_30d: int = 0


class TopicAliasSuggestion(BaseModel):
    """Reviewable topic alias/merge suggestion."""

    model_config = ConfigDict(frozen=True)

    suggestion_id: str
    created_at: str
    topic_id: str
    suggested_alias: str
    reason: str
    status: Literal["pending", "approved", "rejected"] = "pending"
    review_record: str = ""


class BriefingWorkflowState(BaseModel):
    """Replayable briefing workflow state."""

    model_config = ConfigDict(frozen=True)

    run_date: str
    current_stage: Literal[
        "planned",
        "polled",
        "ranked",
        "generated",
        "validated",
        "archived",
        "failed",
    ] = "planned"
    completed_stages: tuple[str, ...] = ()
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    last_error: str = ""


class FeedbackSignal(StrEnum):
    """Explicit briefing feedback signals."""

    KEEP = "keep"
    HIDE = "hide"
    MORE_LIKE_THIS = "more_like_this"
    LESS_LIKE_THIS = "less_like_this"
    TOO_NOISY = "too_noisy"
    ALREADY_KNOWN = "already_known"
    NOT_ACTIONABLE = "not_actionable"
    USEFUL = "useful"
    NEUTRAL = "neutral"
    NOT_USEFUL = "not_useful"
    WRONG_CADENCE = "wrong_cadence"


class FeedbackEvent(BaseModel):
    """Auditable explicit feedback record."""

    model_config = ConfigDict(frozen=True)

    feedback_id: str
    timestamp: str
    target_type: Literal["event", "cluster", "topic", "source", "dossier"]
    target_id: str
    signal_type: FeedbackSignal
    strength: float = Field(default=1.0, ge=0.0, le=5.0)
    reason: str = ""
    context: dict[str, str] = Field(default_factory=dict)


class TopicDossier(BaseModel):
    """Single-topic dossier linked from a daily brief."""

    model_config = ConfigDict(frozen=True)

    dossier_id: str
    topic_id: str
    cluster_ids: tuple[str, ...]
    title: str
    why_it_matters: str
    what_changed: str
    prior_context: str = ""
    evidence_timeline: tuple[dict[str, str], ...] = ()
    linked_artifacts: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    try_next: tuple[str, ...] = ()
    obsidian_note: str = ""


BriefingSourceConfig.model_rebuild()
