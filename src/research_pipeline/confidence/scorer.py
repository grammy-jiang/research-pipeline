"""Per-claim confidence scoring with multi-signal aggregation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from research_pipeline.llm.base import LLMProvider
from research_pipeline.models.claim import (
    AtomicClaim,
    ClaimDecomposition,
    EvidenceClass,
)

logger = logging.getLogger(__name__)


# Hedging language patterns (LVU — Linguistic Verification of Uncertainty)
_HEDGE_STRONG = re.compile(
    r"\b(might|may|could|possibly|perhaps|likely|unlikely|appears?\s+to|"
    r"seems?\s+to|suggests?|tentative|preliminary|speculative|hypothesize)\b",
    re.IGNORECASE,
)
_HEDGE_WEAK = re.compile(
    r"\b(generally|often|sometimes|typically|usually|tends?\s+to|"
    r"in\s+some\s+cases|to\s+some\s+extent|partially|arguably)\b",
    re.IGNORECASE,
)
_CERTAINTY = re.compile(
    r"\b(clearly|definitively|conclusively|proven|established|"
    r"demonstrates?|confirms?|shows?\s+that|evidence\s+confirms)\b",
    re.IGNORECASE,
)


@dataclass
class ConfidenceSignals:
    """Individual confidence signals for a claim."""

    evidence_signal: float = 0.0
    hedging_signal: float = 0.5
    citation_density: float = 0.0
    retrieval_quality: float = 0.0
    consistency_signal: float | None = None

    def aggregate(self, weights: dict[str, float] | None = None) -> float:
        """Compute weighted aggregate confidence score.

        Args:
            weights: Signal weights. Defaults provided if None.

        Returns:
            Aggregate score in [0, 1].
        """
        if weights is None:
            if self.consistency_signal is not None:
                weights = {
                    "evidence": 0.25,
                    "hedging": 0.15,
                    "citation": 0.15,
                    "retrieval": 0.15,
                    "consistency": 0.30,
                }
            else:
                weights = {
                    "evidence": 0.35,
                    "hedging": 0.20,
                    "citation": 0.20,
                    "retrieval": 0.25,
                }

        score = 0.0
        score += weights.get("evidence", 0) * self.evidence_signal
        score += weights.get("hedging", 0) * self.hedging_signal
        score += weights.get("citation", 0) * self.citation_density
        score += weights.get("retrieval", 0) * self.retrieval_quality
        if self.consistency_signal is not None:
            score += weights.get("consistency", 0) * self.consistency_signal

        return round(min(max(score, 0.0), 1.0), 4)


def compute_evidence_signal(evidence_class: EvidenceClass) -> float:
    """Map evidence class to a confidence signal value."""
    mapping = {
        EvidenceClass.SUPPORTED: 0.9,
        EvidenceClass.PARTIAL: 0.6,
        EvidenceClass.CONFLICTING: 0.3,
        EvidenceClass.INCONCLUSIVE: 0.2,
        EvidenceClass.UNSUPPORTED: 0.05,
    }
    return mapping.get(evidence_class, 0.0)


def compute_hedging_signal(text: str) -> float:
    """Detect hedging/certainty language (LVU).

    Returns:
        Score in [0, 1]: 0 = highly hedged, 1 = highly certain.
    """
    if not text:
        return 0.5

    strong_hedges = len(_HEDGE_STRONG.findall(text))
    weak_hedges = len(_HEDGE_WEAK.findall(text))
    certainty = len(_CERTAINTY.findall(text))

    hedge_score = strong_hedges * 1.0 + weak_hedges * 0.5
    certainty_score = certainty * 1.0

    if hedge_score + certainty_score == 0:
        return 0.5  # neutral

    # Ratio: certainty / (certainty + hedging)
    ratio = certainty_score / (certainty_score + hedge_score)
    return round(ratio, 4)


def compute_citation_density(claim: AtomicClaim, max_evidence: int = 5) -> float:
    """Compute citation density signal based on evidence count.

    Args:
        claim: The atomic claim.
        max_evidence: Maximum evidence pieces for normalization.

    Returns:
        Score in [0, 1].
    """
    if not claim.evidence:
        return 0.0
    return round(min(len(claim.evidence) / max_evidence, 1.0), 4)


def compute_retrieval_quality(claim: AtomicClaim) -> float:
    """Compute retrieval quality signal from max evidence score.

    Returns:
        Maximum relevance score across evidence, clamped to [0, 1].
    """
    if not claim.evidence:
        return 0.0
    max_score = max(e.relevance_score for e in claim.evidence)
    return round(min(max(max_score, 0.0), 1.0), 4)


def score_claim(
    claim: AtomicClaim,
    llm_provider: LLMProvider | None = None,
) -> tuple[float, ConfidenceSignals]:
    """Score confidence for a single atomic claim.

    Args:
        claim: The claim to score.
        llm_provider: Optional LLM for multi-sample consistency.

    Returns:
        Tuple of (aggregate_score, signals).
    """
    signals = ConfidenceSignals(
        evidence_signal=compute_evidence_signal(claim.evidence_class),
        hedging_signal=compute_hedging_signal(claim.statement),
        citation_density=compute_citation_density(claim),
        retrieval_quality=compute_retrieval_quality(claim),
    )

    if llm_provider is not None:
        try:
            consistency = _compute_consistency(claim, llm_provider, samples=5)
            signals.consistency_signal = consistency
        except Exception as exc:
            logger.warning(
                "LLM consistency check failed for %s: %s", claim.claim_id, exc
            )

    aggregate = signals.aggregate()
    return aggregate, signals


def score_decomposition(
    decomposition: ClaimDecomposition,
    llm_provider: LLMProvider | None = None,
) -> ClaimDecomposition:
    """Score all claims in a decomposition, updating confidence scores.

    Args:
        decomposition: The claim decomposition to score.
        llm_provider: Optional LLM for multi-sample consistency.

    Returns:
        Updated decomposition with confidence scores.
    """
    scored_claims = []
    for claim in decomposition.claims:
        score, signals = score_claim(claim, llm_provider)
        scored = claim.model_copy(update={"confidence_score": score})
        scored_claims.append(scored)
        logger.debug(
            "Scored %s: %.4f (evidence=%.2f, hedging=%.2f, "
            "citation=%.2f, retrieval=%.2f)",
            claim.claim_id,
            score,
            signals.evidence_signal,
            signals.hedging_signal,
            signals.citation_density,
            signals.retrieval_quality,
        )

    return decomposition.model_copy(update={"claims": scored_claims})


def _compute_consistency(
    claim: AtomicClaim,
    llm_provider: LLMProvider,
    samples: int = 5,
) -> float:
    """Compute multi-sample consistency via LLM.

    Asks LLM to verify the claim N times at temperature > 0,
    then measures agreement ratio.

    Args:
        claim: The claim to verify.
        llm_provider: LLM provider instance.
        samples: Number of verification samples (M=5 minimum).

    Returns:
        Agreement ratio in [0, 1].
    """
    prompt = (
        f"Evaluate whether this claim is well-supported by scientific evidence.\n"
        f"Claim: {claim.statement}\n"
        f"Evidence class: {claim.evidence_class.value}\n\n"
        f"Respond with a JSON object: "
        f'{{"supported": true/false, "reasoning": "brief explanation"}}'
    )

    agreements = 0
    valid_samples = 0

    for _ in range(samples):
        try:
            result = llm_provider.call(
                prompt=prompt,
                schema_id="claim_verification",
                temperature=0.7,
            )
            if isinstance(result, dict) and "supported" in result:
                valid_samples += 1
                if result["supported"]:
                    agreements += 1
        except Exception as exc:
            logger.debug("LLM sample failed: %s", exc)

    if valid_samples == 0:
        return 0.5  # Neutral if no valid samples

    return round(agreements / valid_samples, 4)
