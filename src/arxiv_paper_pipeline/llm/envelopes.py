"""Input/output envelope wrappers for LLM calls."""

import logging

from pydantic import BaseModel, Field

from arxiv_paper_pipeline.infra.hashing import sha256_str

logger = logging.getLogger(__name__)


class LLMInputEnvelope(BaseModel):
    """Standardized input envelope for all LLM calls."""

    task_type: str = Field(description="Task type identifier.")
    schema_id: str = Field(description="Expected output schema ID.")
    prompt_version: str = Field(description="Prompt template version.")
    source_artifact_ids: list[str] = Field(default_factory=list)
    source_hashes: list[str] = Field(default_factory=list)
    llm_profile: str = Field(default="default")
    input_payload: dict = Field(default_factory=dict)  # type: ignore[type-arg]

    @property
    def input_hash(self) -> str:
        """Compute a deterministic hash of the input."""
        import json

        content = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return sha256_str(content)


class LLMOutputEnvelope(BaseModel):
    """Standardized output envelope from all LLM calls."""

    schema_id: str = Field(description="Output schema ID.")
    decision: dict = Field(default_factory=dict)  # type: ignore[type-arg]
    evidence_refs: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    abstain: bool = Field(
        default=False,
        description="Whether the LLM abstained from answering.",
    )
