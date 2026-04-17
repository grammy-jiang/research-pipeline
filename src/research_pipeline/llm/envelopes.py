"""Input/output envelope wrappers for LLM calls.

Includes structured-output enforcement: every ``LLMOutputEnvelope``
can be validated against field requirements (evidence, confidence,
source citations) in strict or lenient mode.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from research_pipeline.infra.hashing import sha256_str
from research_pipeline.llm.structured_output import (
    EnforcementMode,
    EnforcementResult,
    FieldRequirement,
    enforce,
)

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
    """Standardized output envelope from all LLM calls.

    Call :meth:`enforce_structure` after construction to validate
    that the ``decision`` dict contains required evidence / confidence
    / source fields.
    """

    schema_id: str = Field(description="Output schema ID.")
    decision: dict = Field(default_factory=dict)  # type: ignore[type-arg]
    evidence_refs: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    abstain: bool = Field(
        default=False,
        description="Whether the LLM abstained from answering.",
    )

    def enforce_structure(
        self,
        requirements: list[FieldRequirement] | None = None,
        mode: EnforcementMode = EnforcementMode.LENIENT,
    ) -> EnforcementResult:
        """Validate ``decision`` dict against structured-output requirements.

        Args:
            requirements: Field requirements.  Defaults to standard
                evidence + confidence + citation checks.
            mode: Strict (raise) or lenient (repair with defaults).

        Returns:
            :class:`EnforcementResult` with validity, violations, and
            the repaired decision dict.
        """
        result = enforce(dict(self.decision), requirements=requirements, mode=mode)
        if result.repaired != self.decision:
            object.__setattr__(self, "decision", result.repaired)
        return result

    def to_flat_dict(self) -> dict[str, Any]:
        """Return a flattened dict merging envelope fields and decision.

        Useful for downstream consumers that expect a single dict.
        """
        merged: dict[str, Any] = {
            "schema_id": self.schema_id,
            "evidence_refs": self.evidence_refs,
            "notes": self.notes,
            "abstain": self.abstain,
        }
        merged.update(self.decision)
        return merged
