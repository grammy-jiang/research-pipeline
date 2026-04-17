"""4-Layer Confidence Architecture.

Implements a layered confidence scoring system based on research from
Atomic Calibration (Zhang et al.), AGSC, DINCO, LoVeC, and CCPS.

Architecture layers:
  L1 — Fast signal: hedging + evidence heuristics (every claim)
  L2 — Decomposition: adaptive NLI triage (skip/keep/decompose)
  L3 — Calibration: DINCO distractor normalization + Platt-scaling
  L4 — Verification: sampling consistency for low-confidence claims only

References:
  - Atomic Calibration (arXiv): per-claim decompose → 5 methods → fusion
  - AGSC: adaptive granularity (NLI triage, 60% faster)
  - DINCO: distractor normalization (training-free, ECE 0.076)
  - LoVeC: inline confidence tags (GRPO+LoRA, 2-4x better)
  - CCPS: hidden-state probing (55% ECE reduction)
  - Negative constraints: 685K-response benchmark (20 LLMs)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from research_pipeline.confidence.scorer import (
    compute_citation_density,
    compute_evidence_signal,
    compute_hedging_signal,
    compute_retrieval_quality,
)
from research_pipeline.models.claim import AtomicClaim, EvidenceClass

logger = logging.getLogger(__name__)


class ConfidenceLayer(str, Enum):
    """Confidence architecture layer identifiers."""

    L1_FAST_SIGNAL = "L1_fast_signal"
    L2_DECOMPOSITION = "L2_decomposition"
    L3_CALIBRATION = "L3_calibration"
    L4_VERIFICATION = "L4_verification"


class GranularityDecision(str, Enum):
    """NLI triage decision for adaptive granularity (AGSC)."""

    SKIP = "skip"
    KEEP = "keep"
    DECOMPOSE = "decompose"


class CalibrationMethod(str, Enum):
    """Calibration correction method applied."""

    IDENTITY = "identity"
    DINCO = "dinco"
    PLATT = "platt"


@dataclass
class L1Result:
    """Layer 1 — Fast signal result."""

    fast_score: float
    hedging_signal: float
    evidence_signal: float
    citation_density: float
    retrieval_quality: float
    skip_deeper: bool = False


@dataclass
class L2Result:
    """Layer 2 — Adaptive granularity result."""

    decision: GranularityDecision
    adjusted_score: float
    claim_complexity: float = 0.0
    evidence_count: int = 0


@dataclass
class L3Result:
    """Layer 3 — Calibration correction result."""

    raw_score: float
    calibrated_score: float
    method: CalibrationMethod = CalibrationMethod.IDENTITY
    calibration_delta: float = 0.0


@dataclass
class L4Result:
    """Layer 4 — Selective verification result."""

    triggered: bool = False
    sampling_score: float | None = None
    samples_used: int = 0
    agreement_ratio: float = 0.0


@dataclass
class LayeredConfidenceResult:
    """Complete 4-layer confidence scoring result for a claim."""

    claim_id: str
    l1: L1Result
    l2: L2Result
    l3: L3Result
    l4: L4Result
    final_score: float
    layers_executed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return {
            "claim_id": self.claim_id,
            "final_score": round(self.final_score, 4),
            "layers_executed": self.layers_executed,
            "l1": {
                "fast_score": round(self.l1.fast_score, 4),
                "hedging_signal": round(self.l1.hedging_signal, 4),
                "evidence_signal": round(self.l1.evidence_signal, 4),
                "citation_density": round(self.l1.citation_density, 4),
                "retrieval_quality": round(self.l1.retrieval_quality, 4),
                "skip_deeper": self.l1.skip_deeper,
            },
            "l2": {
                "decision": self.l2.decision.value,
                "adjusted_score": round(self.l2.adjusted_score, 4),
                "claim_complexity": round(self.l2.claim_complexity, 4),
                "evidence_count": self.l2.evidence_count,
            },
            "l3": {
                "raw_score": round(self.l3.raw_score, 4),
                "calibrated_score": round(self.l3.calibrated_score, 4),
                "method": self.l3.method.value,
                "calibration_delta": round(self.l3.calibration_delta, 4),
            },
            "l4": {
                "triggered": self.l4.triggered,
                "sampling_score": (
                    round(self.l4.sampling_score, 4)
                    if self.l4.sampling_score is not None
                    else None
                ),
                "samples_used": self.l4.samples_used,
                "agreement_ratio": round(self.l4.agreement_ratio, 4),
            },
        }


# ---------------------------------------------------------------------------
# Calibration Metrics (ECE + Brier + AUROC)
# ---------------------------------------------------------------------------


def compute_ece(
    predictions: list[float],
    actuals: list[float],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    ECE measures the average gap between predicted confidence and actual
    accuracy, weighted by bin population.

    Args:
        predictions: Predicted confidence scores.
        actuals: Ground truth binary labels (0 or 1).
        n_bins: Number of calibration bins.

    Returns:
        ECE in [0, 1]. Lower is better.
    """
    if not predictions or not actuals or len(predictions) != len(actuals):
        return 0.0

    bin_width = 1.0 / n_bins
    total = len(predictions)
    ece = 0.0

    for i in range(n_bins):
        lo = i * bin_width
        hi = lo + bin_width
        indices = [
            j
            for j, p in enumerate(predictions)
            if (lo <= p < hi) or (i == n_bins - 1 and p == hi)
        ]
        if not indices:
            continue
        avg_conf = sum(predictions[j] for j in indices) / len(indices)
        avg_acc = sum(actuals[j] for j in indices) / len(indices)
        ece += (len(indices) / total) * abs(avg_conf - avg_acc)

    return round(ece, 6)


def compute_brier(predictions: list[float], actuals: list[float]) -> float:
    """Compute Brier score (mean squared error of probabilities).

    Args:
        predictions: Predicted confidence scores.
        actuals: Ground truth binary labels (0 or 1).

    Returns:
        Brier score in [0, 1]. Lower is better.
    """
    if not predictions or not actuals or len(predictions) != len(actuals):
        return 0.0
    total = sum((p - a) ** 2 for p, a in zip(predictions, actuals, strict=False))
    mse = total / len(predictions)
    return round(mse, 6)


def compute_auroc(predictions: list[float], actuals: list[float]) -> float:
    """Compute Area Under ROC Curve via trapezoidal rule.

    Uses rank-based (Wilcoxon-Mann-Whitney) estimation.

    Args:
        predictions: Predicted confidence scores.
        actuals: Ground truth binary labels (0 or 1).

    Returns:
        AUROC in [0, 1]. Higher is better.
    """
    if not predictions or not actuals or len(predictions) != len(actuals):
        return 0.5

    n_pos = sum(1 for a in actuals if a >= 0.5)
    n_neg = len(actuals) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Pair-wise comparison
    concordant = 0
    tied = 0
    for _i, (pi, ai) in enumerate(zip(predictions, actuals, strict=False)):
        for _j, (pj, aj) in enumerate(zip(predictions, actuals, strict=False)):
            if ai >= 0.5 and aj < 0.5:
                if pi > pj:
                    concordant += 1
                elif pi == pj:
                    tied += 1

    return round((concordant + 0.5 * tied) / (n_pos * n_neg), 6)


@dataclass
class CalibrationReport:
    """Multi-metric calibration evaluation."""

    ece: float = 0.0
    brier: float = 0.0
    auroc: float = 0.5
    n_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "ece": round(self.ece, 6),
            "brier": round(self.brier, 6),
            "auroc": round(self.auroc, 6),
            "n_samples": self.n_samples,
        }


def evaluate_calibration(
    predictions: list[float], actuals: list[float]
) -> CalibrationReport:
    """Evaluate calibration quality with multiple metrics.

    Deep research negative constraint: ECE alone is misleading — must use
    multi-metric (ECE + AUROC + Brier + calibration plots).

    Args:
        predictions: Predicted confidence scores.
        actuals: Ground truth binary labels.

    Returns:
        CalibrationReport with ECE, Brier, and AUROC.
    """
    return CalibrationReport(
        ece=compute_ece(predictions, actuals),
        brier=compute_brier(predictions, actuals),
        auroc=compute_auroc(predictions, actuals),
        n_samples=len(predictions),
    )


# ---------------------------------------------------------------------------
# DINCO Calibration (training-free, distractor normalization)
# ---------------------------------------------------------------------------


def dinco_calibrate(
    raw_score: float,
    distractor_scores: list[float] | None = None,
    temperature: float = 1.0,
) -> float:
    """DINCO-inspired distractor normalization calibration.

    Training-free calibration: normalizes confidence by comparing to
    distractor (irrelevant) responses. If a model gives similar confidence
    to distractors as to the actual claim, true confidence is low.

    Based on: DINCO (ECE 0.076, training-free).

    Args:
        raw_score: Raw confidence score for the claim.
        distractor_scores: Confidence scores for distractor claims.
            If None, uses a default distribution assumption.
        temperature: Temperature for softmax normalization.

    Returns:
        Calibrated score in [0, 1].
    """
    if distractor_scores is None:
        # Default: assume distractors cluster around 0.3-0.5
        distractor_mean = 0.4
    else:
        if not distractor_scores:
            return raw_score
        distractor_mean = sum(distractor_scores) / len(distractor_scores)

    if temperature <= 0:
        temperature = 1.0

    # Normalize: how much does the real score exceed distractor mean?
    scaled_real = raw_score / temperature
    scaled_dist = distractor_mean / temperature

    # Softmax-style normalization
    try:
        exp_real = math.exp(min(scaled_real, 50.0))
        exp_dist = math.exp(min(scaled_dist, 50.0))
    except OverflowError:
        return raw_score

    calibrated = exp_real / (exp_real + exp_dist)
    return round(min(max(calibrated, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Platt Scaling (simple logistic calibration)
# ---------------------------------------------------------------------------


@dataclass
class PlattParams:
    """Platt scaling parameters (logistic regression)."""

    a: float = 1.0  # slope
    b: float = 0.0  # intercept

    def transform(self, score: float) -> float:
        """Apply Platt scaling: 1 / (1 + exp(-(a*x + b)))."""
        z = self.a * score + self.b
        z = max(min(z, 50.0), -50.0)  # prevent overflow
        return round(1.0 / (1.0 + math.exp(-z)), 4)


def fit_platt_scaling(
    predictions: list[float],
    actuals: list[float],
    learning_rate: float = 0.01,
    iterations: int = 100,
) -> PlattParams:
    """Fit Platt scaling parameters via gradient descent.

    Simple logistic regression calibration (a, b) such that
    calibrated(x) = sigmoid(a*x + b) best fits actuals.

    Args:
        predictions: Uncalibrated confidence scores.
        actuals: Ground truth binary labels.
        learning_rate: Gradient descent step size.
        iterations: Number of optimization iterations.

    Returns:
        Fitted PlattParams.
    """
    if not predictions or not actuals or len(predictions) != len(actuals):
        return PlattParams()

    a = 1.0
    b = 0.0
    n = len(predictions)

    for _ in range(iterations):
        grad_a = 0.0
        grad_b = 0.0
        for x, y in zip(predictions, actuals, strict=False):
            z = a * x + b
            z = max(min(z, 50.0), -50.0)
            p = 1.0 / (1.0 + math.exp(-z))
            err = p - y
            grad_a += err * x
            grad_b += err
        a -= learning_rate * grad_a / n
        b -= learning_rate * grad_b / n

    return PlattParams(a=round(a, 6), b=round(b, 6))


# ---------------------------------------------------------------------------
# Damped Fusion (AdjustedAlpha / DampedFusion)
# ---------------------------------------------------------------------------


def damped_fusion(
    signals: list[float],
    weights: list[float] | None = None,
    damping: float = 0.8,
) -> float:
    """Damped fusion of multiple confidence signals.

    Outperforms fixed-weight aggregation per deep research findings.
    Uses power-damped weighted average to reduce overconfidence.

    Args:
        signals: Confidence signal values.
        weights: Signal weights (uniform if None).
        damping: Damping exponent in (0, 1]. 1.0 = no damping.

    Returns:
        Fused confidence score in [0, 1].
    """
    if not signals:
        return 0.0

    if weights is None:
        weights = [1.0 / len(signals)] * len(signals)

    if len(weights) != len(signals):
        weights = [1.0 / len(signals)] * len(signals)

    # Normalize weights
    w_sum = sum(weights)
    if w_sum <= 0:
        return 0.0
    weights = [w / w_sum for w in weights]

    # Power-damped weighted average
    damping = max(0.01, min(damping, 1.0))
    weighted = sum(w * (s**damping) for w, s in zip(weights, signals, strict=False))
    return round(min(max(weighted, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Claim Complexity Estimation (for L2 adaptive granularity)
# ---------------------------------------------------------------------------


def estimate_claim_complexity(claim: AtomicClaim) -> float:
    """Estimate complexity of a claim for NLI triage.

    Simple heuristic: longer claims with more evidence and conflicting
    evidence are more complex. AGSC-inspired: neutral triage determines
    skip/keep/decompose.

    Args:
        claim: The atomic claim to evaluate.

    Returns:
        Complexity score in [0, 1].
    """
    # Length factor (longer = more complex)
    words = len(claim.statement.split())
    length_factor = min(words / 50.0, 1.0)

    # Evidence complexity
    n_evidence = len(claim.evidence) if claim.evidence else 0
    evidence_factor = min(n_evidence / 5.0, 1.0)

    # Conflicting evidence indicator
    conflict_bonus = 0.3 if claim.evidence_class == EvidenceClass.CONFLICTING else 0.0

    complexity = 0.4 * length_factor + 0.3 * evidence_factor + 0.3 * conflict_bonus
    return round(min(max(complexity, 0.0), 1.0), 4)


def nli_triage(
    claim: AtomicClaim,
    l1_score: float,
    high_threshold: float = 0.85,
    low_threshold: float = 0.15,
    complexity_threshold: float = 0.6,
) -> GranularityDecision:
    """Adaptive NLI triage for claim analysis granularity (AGSC-inspired).

    Decides how much analysis a claim needs based on L1 score and complexity:
    - SKIP: very high/low confidence → no further analysis needed
    - KEEP: moderate confidence, low complexity → keep current score
    - DECOMPOSE: moderate confidence, high complexity → deeper analysis

    This achieves ~60% savings per AGSC paper.

    Args:
        claim: The atomic claim.
        l1_score: Layer 1 fast signal score.
        high_threshold: Score above which to skip (very confident).
        low_threshold: Score below which to skip (very unconfident).
        complexity_threshold: Complexity above which to decompose.

    Returns:
        GranularityDecision.
    """
    if l1_score >= high_threshold or l1_score <= low_threshold:
        return GranularityDecision.SKIP

    complexity = estimate_claim_complexity(claim)
    if complexity >= complexity_threshold:
        return GranularityDecision.DECOMPOSE

    return GranularityDecision.KEEP


# ---------------------------------------------------------------------------
# 4-Layer Architecture Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class ArchitectureConfig:
    """Configuration for the 4-layer confidence architecture."""

    # L1 thresholds
    l1_skip_high: float = 0.85
    l1_skip_low: float = 0.15

    # L2 thresholds
    complexity_threshold: float = 0.6

    # L3 calibration
    dinco_temperature: float = 1.0
    platt_params: PlattParams | None = None

    # L4 verification
    l4_threshold: float = 0.50
    l4_samples: int = 5

    # Fusion
    damping: float = 0.8


def _run_l1(claim: AtomicClaim, config: ArchitectureConfig) -> L1Result:
    """Layer 1: Fast signal computation."""
    evidence_sig = compute_evidence_signal(claim.evidence_class)
    hedging_sig = compute_hedging_signal(claim.statement)
    citation_den = compute_citation_density(claim)
    retrieval_q = compute_retrieval_quality(claim)

    # Quick aggregation with equal weights for fast estimate
    fast_score = damped_fusion(
        [evidence_sig, hedging_sig, citation_den, retrieval_q],
        [0.35, 0.25, 0.20, 0.20],
        damping=config.damping,
    )

    skip = fast_score >= config.l1_skip_high or fast_score <= config.l1_skip_low
    return L1Result(
        fast_score=fast_score,
        hedging_signal=hedging_sig,
        evidence_signal=evidence_sig,
        citation_density=citation_den,
        retrieval_quality=retrieval_q,
        skip_deeper=skip,
    )


def _run_l2(claim: AtomicClaim, l1: L1Result, config: ArchitectureConfig) -> L2Result:
    """Layer 2: Adaptive granularity (AGSC-inspired NLI triage)."""
    if l1.skip_deeper:
        return L2Result(
            decision=GranularityDecision.SKIP,
            adjusted_score=l1.fast_score,
            claim_complexity=0.0,
            evidence_count=len(claim.evidence) if claim.evidence else 0,
        )

    complexity = estimate_claim_complexity(claim)
    decision = nli_triage(
        claim,
        l1.fast_score,
        high_threshold=config.l1_skip_high,
        low_threshold=config.l1_skip_low,
        complexity_threshold=config.complexity_threshold,
    )

    # For DECOMPOSE, adjust score with complexity penalty
    if decision == GranularityDecision.DECOMPOSE:
        penalty = complexity * 0.1
        adjusted = max(l1.fast_score - penalty, 0.0)
    else:
        adjusted = l1.fast_score

    return L2Result(
        decision=decision,
        adjusted_score=round(adjusted, 4),
        claim_complexity=complexity,
        evidence_count=len(claim.evidence) if claim.evidence else 0,
    )


def _run_l3(
    raw_score: float,
    config: ArchitectureConfig,
    distractor_scores: list[float] | None = None,
) -> L3Result:
    """Layer 3: Calibration correction (DINCO + optional Platt)."""
    # Try Platt scaling if parameters are available
    if config.platt_params is not None:
        calibrated = config.platt_params.transform(raw_score)
        return L3Result(
            raw_score=raw_score,
            calibrated_score=calibrated,
            method=CalibrationMethod.PLATT,
            calibration_delta=round(calibrated - raw_score, 4),
        )

    # Otherwise use DINCO (training-free)
    calibrated = dinco_calibrate(
        raw_score,
        distractor_scores=distractor_scores,
        temperature=config.dinco_temperature,
    )
    return L3Result(
        raw_score=raw_score,
        calibrated_score=calibrated,
        method=CalibrationMethod.DINCO,
        calibration_delta=round(calibrated - raw_score, 4),
    )


def _run_l4(
    claim: AtomicClaim,
    l3_score: float,
    config: ArchitectureConfig,
    llm_provider: Any | None = None,
) -> L4Result:
    """Layer 4: Selective verification for low-confidence claims."""
    if l3_score >= config.l4_threshold:
        return L4Result(triggered=False)

    if llm_provider is None:
        # No LLM → cannot verify, return as-is
        return L4Result(triggered=True, samples_used=0)

    # Import here to avoid circular dependency
    from research_pipeline.confidence.scorer import _compute_consistency

    try:
        consistency = _compute_consistency(
            claim, llm_provider, samples=config.l4_samples
        )
        return L4Result(
            triggered=True,
            sampling_score=consistency,
            samples_used=config.l4_samples,
            agreement_ratio=consistency,
        )
    except Exception as exc:
        logger.warning("L4 verification failed for %s: %s", claim.claim_id, exc)
        return L4Result(triggered=True, samples_used=0)


def score_claim_layered(
    claim: AtomicClaim,
    config: ArchitectureConfig | None = None,
    llm_provider: Any | None = None,
    distractor_scores: list[float] | None = None,
) -> LayeredConfidenceResult:
    """Score a claim through the 4-layer confidence architecture.

    Processes claim through L1→L2→L3→L4, where each layer
    progressively refines confidence. L4 only triggers for
    low-confidence claims (< threshold).

    Args:
        claim: Atomic claim to score.
        config: Architecture configuration. Uses defaults if None.
        llm_provider: Optional LLM for L4 sampling consistency.
        distractor_scores: Optional distractor scores for L3 DINCO.

    Returns:
        Complete layered confidence result.
    """
    if config is None:
        config = ArchitectureConfig()

    layers_executed: list[str] = []

    # L1: Fast signal
    l1 = _run_l1(claim, config)
    layers_executed.append(ConfidenceLayer.L1_FAST_SIGNAL.value)
    logger.debug(
        "L1 %s: fast_score=%.4f skip=%s",
        claim.claim_id,
        l1.fast_score,
        l1.skip_deeper,
    )

    # L2: Adaptive granularity
    l2 = _run_l2(claim, l1, config)
    layers_executed.append(ConfidenceLayer.L2_DECOMPOSITION.value)
    logger.debug(
        "L2 %s: decision=%s adjusted=%.4f",
        claim.claim_id,
        l2.decision.value,
        l2.adjusted_score,
    )

    # L3: Calibration
    l3 = _run_l3(l2.adjusted_score, config, distractor_scores)
    layers_executed.append(ConfidenceLayer.L3_CALIBRATION.value)
    logger.debug(
        "L3 %s: raw=%.4f calibrated=%.4f method=%s",
        claim.claim_id,
        l3.raw_score,
        l3.calibrated_score,
        l3.method.value,
    )

    # L4: Selective verification (only for low-confidence)
    l4 = _run_l4(claim, l3.calibrated_score, config, llm_provider)
    if l4.triggered:
        layers_executed.append(ConfidenceLayer.L4_VERIFICATION.value)
        logger.debug(
            "L4 %s: triggered, agreement=%.4f",
            claim.claim_id,
            l4.agreement_ratio,
        )

    # Final score: fuse L3 calibrated + L4 if triggered
    if l4.triggered and l4.sampling_score is not None:
        final = damped_fusion(
            [l3.calibrated_score, l4.sampling_score],
            [0.6, 0.4],
            damping=config.damping,
        )
    else:
        final = l3.calibrated_score

    return LayeredConfidenceResult(
        claim_id=claim.claim_id,
        l1=l1,
        l2=l2,
        l3=l3,
        l4=l4,
        final_score=final,
        layers_executed=layers_executed,
    )


def score_batch_layered(
    claims: list[AtomicClaim],
    config: ArchitectureConfig | None = None,
    llm_provider: Any | None = None,
) -> list[LayeredConfidenceResult]:
    """Score a batch of claims through the 4-layer architecture.

    Args:
        claims: List of atomic claims.
        config: Architecture configuration.
        llm_provider: Optional LLM for L4 verification.

    Returns:
        List of layered confidence results.
    """
    if config is None:
        config = ArchitectureConfig()

    results = []
    for claim in claims:
        result = score_claim_layered(claim, config, llm_provider)
        results.append(result)

    # Log summary
    scores = [r.final_score for r in results]
    if scores:
        avg = sum(scores) / len(scores)
        l4_count = sum(1 for r in results if r.l4.triggered)
        skip_count = sum(
            1 for r in results if r.l2.decision == GranularityDecision.SKIP
        )
        logger.info(
            "Scored %d claims: avg=%.4f, L4_triggered=%d, L2_skipped=%d",
            len(results),
            avg,
            l4_count,
            skip_count,
        )

    return results


def batch_calibration_report(
    results: list[LayeredConfidenceResult],
    ground_truth: list[float] | None = None,
) -> CalibrationReport:
    """Generate calibration report for a batch of scored claims.

    Args:
        results: Layered confidence results.
        ground_truth: Optional ground truth labels for calibration eval.

    Returns:
        CalibrationReport with multi-metric evaluation.
    """
    predictions = [r.final_score for r in results]
    if ground_truth is None:
        # Use evidence signal as proxy for ground truth
        ground_truth = [r.l1.evidence_signal for r in results]

    # Binarize ground truth for AUROC
    binary_truth = [1.0 if g >= 0.5 else 0.0 for g in ground_truth]

    return evaluate_calibration(predictions, binary_truth)
