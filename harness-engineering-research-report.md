# Harness Engineering of AI Agents — Research Report

> **Topic**: Evaluation harnesses, test frameworks, and QA methodologies for
> LLM-based autonomous agents
>
> **Date**: April 2025
>
> **Pipeline Run**: `2edcc350b784` (arXiv search + HuggingFace supplemental)
>
> **Papers Analyzed**: 4 (1 from pipeline screening, 3 from HuggingFace daily papers)
>
> **Readiness Verdict**: `HAS_GAPS` (~65% implementation-ready)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Methodology](#methodology)
3. [Papers Reviewed](#papers-reviewed)
4. [Key Design Patterns](#key-design-patterns)
5. [Unified Metric Framework](#unified-metric-framework)
6. [Architecture Blueprint](#architecture-blueprint)
7. [Points of Agreement](#points-of-agreement)
8. [Points of Contradiction](#points-of-contradiction)
9. [Research Gaps](#research-gaps)
10. [Practical Recommendations](#practical-recommendations)
11. [Evidence Map](#evidence-map)
12. [Readiness Assessment](#readiness-assessment)
13. [Future Directions](#future-directions)
14. [References](#references)

---

## Executive Summary

Four recent papers (all April 2025) collectively define the emerging discipline
of **agent evaluation harness engineering**. They converge on a shared
architectural insight: single-metric, end-to-end evaluation is insufficient for
diagnosing agent failures. Effective harnesses must:

1. **Decompose evaluation** into orthogonal capability dimensions
2. **Use matched-pair experimental designs** to isolate causal factors
3. **Provide dense trajectory-level annotations** for failure localization

The papers span complementary evaluation surfaces — safety (ATBench),
adaptability under interruption (InterruptBench), proactive initiative (PARE),
and contextual file reasoning (HippoCamp) — and together supply a near-complete
blueprint for a modular, multi-surface agent evaluation framework.

**Key finding**: Even frontier models achieve only ~42% success (PARE), ~76.7%
safety F1 (ATBench), and remain far below human performance across all
benchmarks. Agent evaluation is a long-term challenge requiring robust, evolving
harness infrastructure.

---

## Methodology

### Pipeline Execution

| Stage | Result |
|-------|--------|
| Plan | 2 must_terms ("agent", "evaluation"), 9 nice_terms, 10 query variants, 12-month window |
| Search | 274 candidates from arXiv (10 variants × 100 results, deduped) |
| Screen | 274 → 8 shortlisted (BM25 + category + recency) |
| Download | 8/8 PDFs successfully downloaded |
| Convert | 8/8 via docling backend |
| Extract | 8 papers, 391 chunks |
| Summarize | 8 template summaries |

### Screening Quality Assessment

The BM25 screening produced mostly off-topic results due to the broad nature
of must_terms ("agent" + "evaluation"):

- **1/8 relevant**: ATBench (2604.02022, score 0.731) — agent safety benchmark
- **2/8 marginal**: SKILL0, APEX — tangentially related
- **5/8 irrelevant**: radiology, finance, RL, legal, UAV papers

**Root cause**: "agent" and "evaluation" are high-frequency terms across many
domains. BM25 keyword scoring cannot disambiguate domain context.

### Supplemental Discovery

HuggingFace daily papers provided 5 additional relevant papers, 3 of which
were selected for deep analysis:

- InterruptBench (2604.00892) — interruptible agent evaluation
- PARE (2604.00842) — proactive agent environment
- HippoCamp (2604.01221) — contextual agent benchmark

These were downloaded, converted via docling, and analyzed by paper-analyzer
sub-agents.

---

## Papers Reviewed

| # | Paper | ArXiv ID | Rating | Key Contribution |
|---|-------|----------|--------|------------------|
| 1 | **ATBench**: Diverse Trajectory Benchmark for Agent Safety | 2604.02022 | ★★★★☆ | Taxonomy-driven test generation; detection vs diagnosis gap |
| 2 | **InterruptBench**: Evaluating Interruptible Agents | 2604.00892 | ★★★★★ | Perturbation injection; paired-outcome quadrants; SR(k) curves |
| 3 | **PARE**: Proactive Agent Research Environment | 2604.00842 | ★★★★½ | FSM environments; decomposed capability/reliability metrics |
| 4 | **HippoCamp**: Benchmarking Contextual Agents | 2604.01221 | ★★★★☆ | Dense annotations (79:1); 5-stage failure pipeline |

### Paper 1: ATBench (2604.02022)

Agent safety benchmark with 1,000 trajectory-level test cases spanning 2,084
tools. Uses a 3-dimensional orthogonal taxonomy:

- **Dimension 1**: Risk source (user intent, tool misuse, environmental)
- **Dimension 2**: Failure mode (permission violation, data leak, etc.)
- **Dimension 3**: Harm type (privacy, financial, physical, etc.)

**Critical finding**: GPT-5.4 achieves 76.7% F1 on binary safety detection but
only 33.6% on fine-grained diagnosis — a **2.3× detection→diagnosis gap**.
This means models can flag problems but cannot explain them.

**Transferable patterns**:
- Taxonomy-as-scaffold for systematic test generation
- Paired safe/unsafe trajectory construction
- Delayed-trigger protocol (harm appears after benign steps)
- Multi-layer QA: rule-based → LLM-based → human annotation
- Causal chain annotation for root-cause analysis

### Paper 2: InterruptBench (2604.00892)

First benchmark specifically targeting **interruptible agents** in web
navigation. Tests 3 interruption types:

- **Addition**: New goal added mid-task
- **Revision**: Existing goal parameters changed
- **Retraction**: Goal cancelled entirely

**Critical finding**: Stale-state is the dominant failure mode — agents update
their plans but not their environment state after interruptions. Success traces
treat interrupts as state-changing events; failure traces treat them as
surface-level messages.

**Transferable patterns**:
- Trajectory-grounded perturbation injection (naturalistic timing)
- Budget-limited success curves SR(k) — captures adaptation dynamics
- Paired-outcome quadrant analysis (S/F, F/S, S/S, F/F) for causal isolation
- Perturbation timing as diagnostic variable

### Paper 3: PARE (2604.00842)

Proactive agent evaluation via **FSM-based environment simulation**. Models
applications as finite state machines with:

- **Asymmetric interfaces**: Users see FSM-constrained views; agents access
  flat APIs (mirrors real deployment)
- **Stackelberg turn-taking**: User acts first, agent observes and decides
  whether to intervene

**Critical finding**: Even frontier models cap at ~42% success. Claude shows
a distinctive pattern: low proposal rate (12.8%) with high acceptance rate
(78.2%) — "less is more" for proactive agents.

**Transferable patterns**:
- FSM-based environment modeling
- Decomposed metrics: capability S@k vs reliability S^k
- Proposal rate × acceptance rate × execution success
- Simulated user as test primitive
- Noise injection for robustness testing

### Paper 4: HippoCamp (2604.01221)

Personal file system benchmark: 42.4 GB, 2K+ files, 581 QA pairs with
**46,100 annotations** (79:1 annotation-to-question ratio).

**Critical finding**: Perception (not search/retrieval) is the universal
bottleneck — perception accuracy is roughly half of search accuracy across
all methods (e.g., ChatGPT: 56.5% search → 28.5% perception).

**Transferable patterns**:
- Multi-granularity diagnostic framework
- 5-stage canonical failure pipeline:
  1. Off-target retrieval
  2. Grounding avoidance
  3. Evidence hallucination
  4. Entity misbinding
  5. Missing verification
- Decoupled retrieval/reasoning metrics
- Atomic evidence units for fine-grained evaluation
- Difficulty scoring for calibrated benchmarking

---

## Key Design Patterns

Six design patterns emerge across the papers, ranked by cross-paper validation:

### Pattern 1: Decomposed Multi-Level Metrics (4/4 papers)

**Strongest consensus finding.** Every paper independently rejects single
aggregate metrics:

| Paper | Decomposition |
|-------|--------------|
| ATBench | Detection F1 vs Diagnosis F1 (2.3× gap) |
| InterruptBench | SR(k) curves + paired-outcome quadrants |
| PARE | Capability S@k vs Reliability S^k × (proposal × acceptance × execution) |
| HippoCamp | Search → Perception → Reasoning (cascade) |

**Implementation principle**: Every evaluation must measure at minimum:
1. **Outcome** — did the agent succeed?
2. **Process** — did it succeed for the right reasons?
3. **Efficiency** — at what cost (tokens, time, steps)?

### Pattern 2: Taxonomy-Driven Test Generation (3/4 papers)

Systematic test construction via dimensional cross-products:

| Paper | Taxonomy Dimensions |
|-------|-------------------|
| ATBench | Risk source × Failure mode × Harm type |
| InterruptBench | 3 interruption types × task complexity × timing |
| PARE | 4 capability axes × app domain × complexity |

**Implementation principle**: Define orthogonal taxonomy dimensions. Generate
test cases from the cross-product. Coverage is guaranteed by construction.

### Pattern 3: Matched-Pair Causal Analysis (3/4 papers)

| Paper | Pair Design |
|-------|-------------|
| ATBench | Safe ↔ Unsafe trajectories (same structure) |
| InterruptBench | Interrupted ↔ Baseline (same task) |
| PARE | Noisy ↔ Clean conditions (same scenario) |

**Implementation principle**: For every perturbation, run a paired baseline on
the same task. Classify outcomes into quadrants for causal attribution.

### Pattern 4: Trajectory-Level Failure Diagnosis (3/4 papers)

| Paper | Diagnostic Approach |
|-------|-------------------|
| ATBench | Causal chain annotation (risk → failure → harm) |
| InterruptBench | Stale-state identification via trajectory comparison |
| HippoCamp | 5-stage failure cascade with step-wise localization |

### Pattern 5: Multi-Layer Quality Assurance (3/4 papers)

All benchmark-constructing papers use multi-layer QA:

```
Rule-based filtering → LLM-based validation → Human annotation
```

This is a **necessary** (not optional) component.

### Pattern 6: Environment Simulation & State Management (2/4 papers)

PARE (FSM-based) and InterruptBench (WebArena stateful) both demonstrate
that **stateful environment simulation is essential** for evaluating agent
adaptability. Flat API-based evaluation misses state-related failure modes.

---

## Unified Metric Framework

Synthesizing all four papers, a comprehensive agent evaluation should measure:

```
Level 0: Aggregate
├── Success Rate (binary)
├── Cost (tokens, API calls, wall-clock time)

Level 1: Decomposed Outcome
├── Detection metrics (can the agent identify the right action?)
├── Diagnosis metrics (can it explain why?)
├── Capability S@k (success when it acts)
├── Reliability S^k (consistency across runs)

Level 2: Process Metrics
├── Search/retrieval accuracy (HippoCamp)
├── Perception accuracy (HippoCamp, ~50% of search)
├── Reasoning accuracy (HippoCamp)
├── Proposal rate × Acceptance rate × Execution success (PARE)
├── Budget-limited success curves SR(k) (InterruptBench)

Level 3: Causal Analysis
├── Paired-outcome quadrants: S/F, F/S, S/S, F/F (InterruptBench)
├── Detection→Diagnosis gap ratio (ATBench, expect ~2×)
├── Perturbation timing sensitivity (InterruptBench)
├── Noise injection degradation curves (PARE)

Level 4: Failure Diagnosis
├── 5-stage failure pipeline (HippoCamp):
│   retrieval → grounding → evidence → binding → verification
├── Stale-state detection (InterruptBench)
├── Causal chain extraction (ATBench)
└── Difficulty-stratified performance (HippoCamp, ATBench)
```

---

## Architecture Blueprint

Combining insights from all four papers, a comprehensive agent evaluation
harness would have the following modular architecture:

```
┌────────────────────────────────────────────────────────────────┐
│                    EVALUATION HARNESS                          │
│                                                                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │  Test Generation  │  │   Execution      │  │  Analysis    │ │
│  │                  │  │   Engine         │  │              │ │
│  │  • Taxonomy      │  │                  │  │  • Metric    │ │
│  │    cross-product │→│  • Agent runner   │→│    engine    │ │
│  │  • Perturbation  │  │  • Environment   │  │  • Failure   │ │
│  │    injection     │  │    adapters      │  │    pipeline  │ │
│  │  • Paired        │  │  • State tracker │  │  • Causal    │ │
│  │    construction  │  │  • Budget mgr    │  │    analysis  │ │
│  └──────────────────┘  └──────────────────┘  └──────────────┘ │
│                                                                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │  QA Pipeline     │  │  Storage         │  │  Reporting   │ │
│  │                  │  │                  │  │              │ │
│  │  • Rule checker  │  │  • Trajectory DB │  │  • Dashboard │ │
│  │  • LLM judge     │  │  • Metric store  │  │  • Regression│ │
│  │  • Human review  │  │  • Artifact      │  │    detection │ │
│  │  • IAA scoring   │  │    storage       │  │  • A/B diffs │ │
│  └──────────────────┘  └──────────────────┘  └──────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

### Module Responsibilities

| Module | Draws From | Responsibility |
|--------|-----------|---------------|
| Test Generation | ATBench taxonomy, InterruptBench perturbation, PARE FSM | Generate test suites from taxonomies; inject perturbations; construct matched pairs |
| Execution Engine | PARE FSM runner, InterruptBench WebArena, HippoCamp file systems | Run agents in sandboxed environments with state tracking and budget limits |
| Analysis | All 4 papers' metric decompositions | Compute decomposed metrics at all 5 levels; perform failure diagnosis |
| QA Pipeline | ATBench rule→LLM→human, HippoCamp 5-phase QA | Validate test cases, trajectories, and annotations through multi-layer QA |
| Storage | HippoCamp annotation format, ATBench causal chains | Persist trajectories, metrics, annotations with versioning |
| Reporting | InterruptBench quadrants, PARE S@k/S^k, HippoCamp difficulty | Generate dashboards, regression reports, and A/B comparison diffs |

### Environment Adapter Interface

```python
class EnvironmentAdapter(Protocol):
    def reset(self) -> State: ...
    def step(self, action: Action) -> tuple[State, Observation, bool]: ...
    def get_state(self) -> State: ...
    def inject_perturbation(self, perturbation: Perturbation) -> None: ...
    def snapshot(self) -> Checkpoint: ...
    def restore(self, checkpoint: Checkpoint) -> None: ...
```

Adapters needed:
- **FSM adapter** (PARE-style): For interactive app simulation
- **Web adapter** (InterruptBench-style): Selenium/Playwright wrapper
- **File system adapter** (HippoCamp-style): Sandboxed file system
- **API adapter**: For tool-use evaluation (ATBench-style)

---

## Points of Agreement

1. **Single metrics are insufficient** — All 4 papers independently arrive at
   multi-metric decomposition. This is the strongest consensus finding.

2. **Trajectory-level evaluation is essential** — All 4 operate at the
   trajectory level. Agent failures manifest over sequences, not individual
   decisions.

3. **Matched-pair designs enable causal reasoning** — 3/4 papers independently
   adopt paired experimental designs. Convergence validates it as fundamental.

4. **Post-detection diagnosis is dramatically harder** — ATBench's 2.3× gap
   and HippoCamp's perception bottleneck both show identifying ≠ explaining.

5. **Frontier models have fundamental limitations** — ~42% success (PARE),
   ~76.7% safety F1 (ATBench). Agent evaluation is a long-term infrastructure
   challenge.

---

## Points of Contradiction

### 1. Structured vs Unstructured Environments

PARE advocates FSM-based structured modeling; HippoCamp operates on
unstructured 42.4 GB file systems. **Resolution**: Both are needed — the right
abstraction depends on the evaluation surface.

### 2. User Simulation Fidelity

PARE uses LLM-based user simulators; InterruptBench uses synthetic injection
without live simulation; ATBench/HippoCamp use static benchmarks. **Resolution**:
User simulation is critical for interactive evaluation, less so for safety or
retrieval benchmarks.

### 3. Annotation Density Trade-offs

HippoCamp's 79:1 ratio provides finest granularity at extreme cost. ATBench/PARE
use lighter schemes. **Resolution**: Configurable density — dense for diagnosis,
light for regression testing.

---

## Research Gaps

### Academic Gaps (require further research)

| # | Gap | Severity | Suggested Queries |
|---|-----|----------|-------------------|
| 1 | Multi-agent evaluation methodology | HIGH | "multi-agent system evaluation benchmark" |
| 2 | Adversarial/red-team test generation | MEDIUM | "adversarial testing LLM agents red team" |
| 3 | Metric measurement theory (validity, reliability) | MEDIUM | "construct validity AI agent evaluation metrics" |
| 4 | Cross-domain harness transferability | LOW | "cross-domain benchmark transfer agent evaluation" |
| 5 | Human-agent co-evaluation | LOW | "human-in-the-loop evaluation AI agents" |

### Engineering Gaps (fillable without new research)

| # | Gap | Severity | Resolution Approach |
|---|-----|----------|-------------------|
| 1 | Harness orchestration framework | HIGH | DAG-based pipeline (CI/CD-inspired) |
| 2 | Persistent storage & state management | HIGH | SQLite/PostgreSQL + object storage |
| 3 | API design for harness integration | HIGH | REST API + Python SDK |
| 4 | CI/CD regression testing integration | MEDIUM | GitHub Actions + baseline management |
| 5 | Cost-aware evaluation budgets | MEDIUM | SR(k)-inspired Pareto curves |
| 6 | Pluggable environment adapters | MEDIUM | Protocol-based adapter interface |
| 7 | LLM-as-judge reliability infrastructure | MEDIUM | Multi-judge ensemble + drift detection |
| 8 | Test case versioning | LOW | Content-addressable storage |

---

## Practical Recommendations

### 1. Adopt Taxonomy-Driven Test Generation

Use ATBench's dimensional cross-product pattern:
```
dim₁ × dim₂ × dim₃ → test cases (diverse by construction)
```
For a general harness: task_complexity × tool_count × state_depth ×
interruption_type.

### 2. Implement ≥2-Level Metric Hierarchy

Every evaluation must measure:
- **Outcome**: Did the agent succeed?
- **Process**: For the right reasons? (the 2.3× gap matters)
- **Efficiency**: At what cost?

### 3. Default to Matched-Pair Experimental Design

For every perturbation, run the paired baseline. Classify into InterruptBench's
four quadrants:

| | Baseline Succeeds | Baseline Fails |
|-|-------------------|----------------|
| **Perturbed Succeeds** | S/S (robust) | F/S (recovery — model capability) |
| **Perturbed Fails** | S/F (fragile — perturbation impact) | F/F (both fail — highest cost) |

### 4. Build a 5-Stage Failure Pipeline

Adapt HippoCamp's canonical cascade:
1. Retrieval → 2. Grounding → 3. Evidence → 4. Binding → 5. Verification

Each stage inherits errors from the previous one. Different agents break at
different stages, explaining divergent capability profiles.

### 5. Invest in Multi-Layer QA

All benchmark papers use rule → LLM → human QA. Automate the first two layers;
reserve human effort for edge cases and final validation.

---

## Evidence Map

| Design Question | ATBench | InterruptBench | PARE | HippoCamp |
|-----------------|---------|----------------|------|-----------|
| Taxonomy-driven generation | ✓ | ✓ | ✓ | Partial |
| Decomposed metrics | ✓ | ✓ | ✓ | ✓ |
| Matched-pair analysis | ✓ | ✓ | ✓ | — |
| Trajectory annotation | ✓ | Partial | — | ✓ |
| Environment simulation | — | Partial | ✓ | Partial |
| User simulation | — | — | ✓ | — |
| Multi-layer QA | ✓ | ✓ | ✓ | ✓ |
| Failure pipeline | Partial | ✓ | — | ✓ |
| Noise/robustness testing | — | ✓ | ✓ | — |
| Budget/cost analysis | — | ✓ | — | Partial |
| Difficulty calibration | ✓ | Partial | — | ✓ |

---

## Readiness Assessment

### Verdict: `HAS_GAPS` (~65% implementation-ready)

The four papers provide a **strong conceptual foundation** and several
immediately implementable design patterns. The evaluation logic layer
(test generation, metrics, failure diagnosis) is well-defined. However,
the **systems engineering layer** (orchestration, persistence, API design,
CI/CD, security) is not addressed by any paper and requires additional
engineering work.

**What's ready to implement**:
- Taxonomy-driven test generation engine
- Decomposed metric computation at all levels
- Matched-pair experimental framework
- 5-stage failure diagnosis pipeline
- Multi-layer QA pipeline

**What needs engineering work** (no new research required):
- Harness orchestration framework (HIGH)
- Storage and state management (HIGH)
- API design and SDK (HIGH)
- CI/CD integration (MEDIUM)
- Environment adapter abstraction (MEDIUM)

**What needs more research**:
- Multi-agent evaluation (HIGH)
- Adversarial test generation (MEDIUM)
- Metric validity theory (MEDIUM)

---

## Future Directions

1. **Unified agent evaluation framework** — Combine all 4 papers' patterns
   into a single modular harness with pluggable evaluation surfaces.

2. **Continuous evaluation infrastructure** — Extend one-shot benchmarks into
   CI/CD-integrated regression harnesses.

3. **Adaptive evaluation difficulty** — Computerized adaptive testing that
   adjusts difficulty to agent capability level.

4. **Multi-agent interaction testing** — Extend PARE's asymmetric interface
   pattern to agent-agent evaluation.

5. **Adversarial harness generation** — LLM-generated adversarial test cases
   targeting known failure modes.

---

## References

1. **ATBench**: "A Diverse and Realistic Trajectory Benchmark for Long-Horizon
   Agent Safety." arXiv:2604.02022, April 2025. Trajectory-level safety
   benchmark, 1000 test cases, 3D taxonomy, multi-layer QA.

2. **InterruptBench**: "Evaluating Interruptible Agents in Long-Horizon Web
   Navigation." arXiv:2604.00892, April 2025. First interruptible agent
   benchmark, 3 interruption types, paired-outcome quadrants, SR(k) curves.

3. **PARE**: "Proactive Agent Research Environment." arXiv:2604.00842, April
   2025. FSM-based environment simulation, asymmetric interfaces, Stackelberg
   turn-taking, decomposed S@k/S^k metrics.

4. **HippoCamp**: "Benchmarking Contextual Agents on Personal Computers."
   arXiv:2604.01221, April 2025. 42.4GB file system benchmark, 79:1
   annotation density, 5-stage failure pipeline, perception bottleneck.

---

## Pipeline Issues Identified

### This Run (2edcc350b784)

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | BM25 screening 7/8 irrelevant for broad topic terms | MEDIUM | Known limitation — keyword screening cannot disambiguate domain context |
| 2 | HuggingFace papers not in pipeline search | MEDIUM | Supplemented manually; adding as formal source would improve recall |
| 3 | No new code bugs discovered | — | Pipeline ran cleanly |

### Prior Run (871064082502) — Fixes Already Committed

| # | Fix | Commit |
|---|-----|--------|
| 1 | Download path doubled `pdf/` | 76fbddc |
| 2 | BM25 must_terms not weighted higher | c28b072 |
| 3 | `convert-file` missing `--config` | 28f2d4e |

---

*Generated by research-pipeline v0.3.0 with paper-screener, paper-analyzer,
and paper-synthesizer sub-agents.*
