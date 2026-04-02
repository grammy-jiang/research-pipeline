"""Manifest and artifact tracking models."""

from datetime import datetime

from pydantic import BaseModel, Field


class LLMCallRecord(BaseModel):
    """Record of a single LLM API call for audit purposes."""

    call_id: str = Field(description="Unique call identifier.")
    provider: str = Field(description="LLM provider name.")
    model: str = Field(description="Model identifier.")
    prompt_version: str = Field(description="Prompt template version.")
    input_hash: str = Field(description="SHA-256 of the input payload.")
    output_hash: str = Field(description="SHA-256 of the raw response.")
    token_usage: dict[str, int] = Field(
        default_factory=dict,
        description="Token usage stats (prompt_tokens, completion_tokens, etc.).",
    )
    called_at: datetime = Field(description="Call timestamp (UTC).")
    duration_ms: int = Field(description="Call duration in milliseconds.")


class ArtifactRecord(BaseModel):
    """Record of a single artifact produced during a run."""

    artifact_id: str = Field(description="Unique artifact identifier.")
    artifact_type: str = Field(
        description="Type: atom_xml, metadata_jsonl, pdf, markdown, etc."
    )
    path: str = Field(description="Relative path within the run directory.")
    sha256: str = Field(description="SHA-256 hash of the artifact.")
    producer: str = Field(description="Stage that produced this artifact.")
    inputs: list[str] = Field(
        default_factory=list,
        description="Input artifact IDs used to produce this artifact.",
    )
    tool_fingerprint: str | None = Field(
        default=None,
        description="Fingerprint of the tool used (converter version, etc.).",
    )
    created_at: datetime = Field(description="Creation timestamp (UTC).")


class StageRecord(BaseModel):
    """Execution record for a single pipeline stage."""

    stage_name: str = Field(description="Stage identifier.")
    status: str = Field(description="completed, failed, skipped, pending.")
    started_at: datetime | None = Field(default=None)
    ended_at: datetime | None = Field(default=None)
    duration_ms: int | None = Field(default=None)
    input_hash: str | None = Field(
        default=None,
        description="Content hash of stage inputs.",
    )
    output_paths: list[str] = Field(
        default_factory=list,
        description="Artifact paths produced.",
    )
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class RunManifest(BaseModel):
    """Top-level manifest for an entire pipeline run."""

    schema_version: str = Field(default="1")
    run_id: str = Field(description="Unique run identifier.")
    created_at: datetime = Field(description="Run creation timestamp (UTC).")
    package_version: str = Field(description="research-pipeline version.")
    config_snapshot: dict = Field(  # type: ignore[type-arg]
        default_factory=dict,
        description="Configuration snapshot at run start.",
    )
    topic_input: str = Field(description="Original topic input.")
    stages: dict[str, StageRecord] = Field(
        default_factory=dict,
        description="Per-stage execution records.",
    )
    artifacts: list[ArtifactRecord] = Field(
        default_factory=list,
        description="All artifacts produced during the run.",
    )
    llm_calls: list[LLMCallRecord] = Field(
        default_factory=list,
        description="All LLM calls made during the run.",
    )
