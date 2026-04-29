"""X (Twitter) policy stub (Phase F).

This adapter intentionally REFUSES to operate unless the source has been
explicitly reviewed and policy-gated. Live access requires the official API
(no scraping) plus organization approval. Until then, every poll raises
``RuntimeError`` so the source cannot accidentally be enabled.

Required configuration to allow polling (and even then, polling is the
caller's responsibility — the stub returns no events):

* ``source.enabled is True``
* ``source.auth_required is True``
* ``source.last_reviewed_at`` is set
* ``"policy_gate_passed"`` is in ``source.tags``
"""

from __future__ import annotations

from pathlib import Path

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingSourceConfig,
    IntelligenceEvent,
)


class XApiPolicyError(RuntimeError):
    """Raised when the X (Twitter) source is polled without policy approval."""


class XApiSource:
    """Policy-gated stub for the X API.

    Construction succeeds; ``poll`` enforces the gate.
    """

    POLICY_TAG = "policy_gate_passed"

    def __init__(
        self,
        source: BriefingSourceConfig,
        *,
        fixture_base_dir: Path | None = None,
    ) -> None:
        if source.access_method != AccessMethod.X_API:
            raise ValueError(f"XApiSource requires X_API (got {source.access_method})")
        self.source = source
        self.fixture_base_dir = fixture_base_dir

    def poll(self) -> list[IntelligenceEvent]:
        """Refuse to poll unless every policy condition is met."""
        self._check_policy()
        # Even when the gate passes, the stub returns no events; live polling
        # will be implemented in a later phase under the official API.
        return []

    def _check_policy(self) -> None:
        reasons: list[str] = []
        if not self.source.enabled:
            reasons.append("source is disabled by policy")
        if not self.source.auth_required:
            reasons.append("auth_required must be True for X API")
        if not self.source.last_reviewed_at:
            reasons.append("last_reviewed_at is not set")
        if self.POLICY_TAG not in self.source.tags:
            reasons.append(f"missing required tag '{self.POLICY_TAG}' in source.tags")
        if reasons:
            raise XApiPolicyError("X API source refused: " + "; ".join(reasons))
