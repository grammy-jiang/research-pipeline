# Step 1 Extraction Framework Design / 第一步提取框架设计

## 1. Executive Summary / 执行摘要

A well-designed Step 1 extraction is **critical** for later synthesis and system design.  Rather than a free-form summary, Step 1 should produce a **structured data record** for each paper, with discrete fields for claims, methods, metrics, results, assumptions, etc.  This structured approach preserves provenance (with citations or quotes) and uncertainty flags, avoiding the information loss and hallucination common in naïve summarization【50†L33-L39】【38†L51-L59】.  We recommend a **schema-first, multi-pass pipeline**: for each paper, parse and analyze the text to fill a predefined template (e.g. JSON or structured markdown) with *evidence‑backed* entries (including explicit source spans) and clear labels (author‑stated vs model‑inferred).  The template should cover **operational details** (throughput, latency, scale, hardware, cost, security, reliability, etc.) and distinguish general reusable components from paper-specific novelties.  We propose concrete schemas (both a minimal and an advanced version) and a multi-step workflow (ingestion, extraction, evidence linking, review).  A rigorous quality checklist and scoring rubric ensure completeness and accuracy.  Anti‑patterns (free-text narrations, missing citations, unsupported claims) are identified and avoided.  The result is a practical, high-fidelity extraction methodology that maximizes downstream synthesis value.【50†L146-L148】【38†L177-L186】

## 2. Why Step 1 is a Critical Bottleneck / 第一步为何是关键瓶颈

Step 1 provides the **ground truth** for all later cross‑paper synthesis and design.  If Step 1 is weak—merely a narrative summary—crucial information is lost or blurred【50†L33-L39】【38†L51-L59】.  One-shot summarization “tends to blur retrieval and reasoning, obscure intermediate evidence, and make it difficult to inspect [the] provenance” of claims【50†L33-L39】.  Important details (edge‑case behaviors, context conditions, negative results) vanish as each layer of summarization compounds errors【38†L51-L59】【38†L79-L88】.  Conversely, a structured extraction explicitly **separates retrieval from reasoning**: each paper’s claims, methods, metrics, and evidence are captured as discrete units.  This guarantees that no evidence is inadvertently omitted or reinterpreted by the LLM.  In turn, cross-paper synthesis (Step 2) can reliably compare claims, identify contradictions, and assemble reusable patterns.  In short, downstream design choices hinge on the richness and fidelity of the extracted material【38†L177-L186】【60†L497-L500】.  A robust Step 1 is the **foundation** – without it, synthesis “behaves like a lossy codec” and critical insights are lost【38†L51-L59】【38†L79-L88】.

## 3. Failure Modes of Weak Step 1 Extraction / 弱化第一步提取的失败模式

When Step 1 defaults to a generic summary, several failure modes arise:

- **Information Loss (“Broken Links”)**: Hierarchical summaries drop details across chunk boundaries【38†L51-L59】【38†L79-L88】.  References to “it” or nuanced conditions are lost, so the final report may omit key context.  (E.g. summarizing separate sections independently can “lose the referent” of earlier observations【38†L79-L88】.)
- **Semantic Drift (Telephone Game Effect)**: Repeated rewriting pushes the narrative toward model priors【38†L79-L88】. Minor sentiments or qualifications become exaggerated (e.g. “slightly annoyed” → “very unhappy”), and subtle assumptions vanish.
- **Hallucination Anchoring**: Spurious inferences in one layer get treated as fact in the next【38†L79-L88】. An incorrect assumption early (e.g. hardware used) will skew all downstream reasoning, yet an ordinary summary pipeline never flags it.  Early hallucinations “harden” and propagate【38†L79-L88】.
- **Opaque Mix of Sources**: A blanket summary often conflates claims from multiple papers, obscuring who said what.  Without structured attributions, one cannot trace a design idea back to its origin.  (This breaks auditability and violates the need to “inspect and challenge model inferences”【60†L497-L500】.)
- **Ignored Contradictions**: Contradictory findings across papers are easily glossed over.  For example, clinical studies often have conflicting results, which literature surveys must flag【56†L37-L40】.  A naive summary might average them out, hiding the tension.  The result is that Step 2 can’t build a contradiction map, missing critical engineering trade-offs.
- **Over-Generality or Vagueness**: Free-form summaries can introduce generic filler (“the authors also discuss related work” etc.) that dilutes actionable content.  This noise consumes context without adding value for design.
- **Unanchored Claims**: Claims in summaries often lack explicit evidence markers (e.g. page numbers or quotes), making validation impossible.   A weak Step 1 makes it impossible to verify “which sentence came from where,” undermining trust.
- **Disconnected Insights**: Technical papers contain operational specifics (e.g. protocol details, hardware requirements) that get dropped. Summaries rarely preserve throughput, latency, or reliability implications unless explicitly queried.

In sum, a weak Step 1 extraction turns the pipeline into a “Chinese whispers” game【38†L79-L88】.  Later design suffers from missed nuance, unchecked hallucinations, and undocumented assumptions – exactly what a structured extraction aims to prevent【50†L33-L39】【38†L177-L186】.

## 4. Design Principles for Step 1 / 第一步设计原则

Based on the above failure modes, Step 1 should be designed around these principles:

- **Structured, Schema-First Output**: Define a strict schema (JSON, XML, or marked-up Markdown) for extracted fields.  Every extracted item belongs to a known field (e.g. *Claims*, *Methods*, *Results*, *Limitations*)【50†L146-L148】.  This avoids narrative drift and ensures uniform coverage across papers.
- **Explicit Evidence Linking**: Each fact or claim must be tied back to the source text.  Include citations, page spans, or verbatim quotes for key points【20†L272-L280】【60†L497-L500】.  Maintain “document-level provenance” so that every field entry is auditable【17†L324-L330】【30†L1-L4】.
- **Preservation of Uncertainty and Inference**: Separate *explicit paper claims* (what the authors stated) from *model inferences*.  Use separate fields or flags (e.g. “AuthorClaim” vs “Inferred”)【20†L272-L280】.  Attach a confidence score or uncertainty note to each entry so downstream analysis can weigh credibility【20†L272-L280】.
- **Modularity and Reusability**: Distinguish paper‑specific details from reusable concepts.  E.g. isolate general algorithms, network architectures, or design patterns that appear across papers.  Flag novelty separately.  This supports later cataloging of reusable modules vs unique contributions.
- **Comprehensive Category Coverage**: Do not restrict to classic summary elements (intro, method, result).  Include *operational attributes*: scalability assumptions, performance metrics (latency/throughput), hardware/infra requirements, security/privacy considerations, observability/data needed, etc.  These are often omitted in summaries but crucial for system design.
- **Explicit Assumptions & Constraints**: Capture any stated or implicit assumptions (e.g. “experiments assume unlimited GPU”, “single-thread performance” etc.) and failure modes.  Document limitations in quantitative terms where possible (e.g. input size limits, rare case breakdown).  Flag any “weakly supported” claims as such.
- **Multi-Pass and Validated Extraction**: Use an iterative approach rather than forcing all fields at once.  For example, one LLM pass identifies key concepts (with chain-of-thought), another fills fields under schema constraints【37†L248-L257】【54†L83-L86】.  Optionally employ a “review” pass where the LLM checks its own output for consistency and completeness.
- **Dual-Format or Hybrid Output**: Consider formats that are both human- and machine-friendly.  For instance, a Markdown template with clearly delimited sections (for readability) that can be parsed by a script, or a JSON output that LLMs can generate reliably.  (Trade-offs are discussed in section 7.)
- **Auditability and Versioning**: Log the exact prompt, model version, and timestamp for each extraction.  Maintain version-controlled schemas so that Step 1 can be re-run reproducibly when needed【17†L324-L330】.

Each recommended schema field (in sections 5 and 7) should be explicitly justified by how it feeds Step 2: e.g. *“PerformanceMetrics”* field exists so the synthesis can compare results across papers; *“Limitations”* so contradictions can be mapped; *“Assumptions”* so design constraints are known, etc.

## 5. Required Information Categories / 必需的信息类别

Every paper should be mined for at least the following categories of information (more if context-specific):

- **Identification & Metadata**: Title, authors, venue, year, and DOI/URL. (Baseline for traceability.)
- **Problem Statement / Context**: The domain and challenge the paper addresses (e.g. “This paper tackles X in Y setting”), including any context like datasets or benchmarks used.
- **Core Contributions (Claims)**: Explicit claims or contributions listed by the authors (e.g. “We propose a new scheduling algorithm achieving Z”).
- **Approach/Methodology**: Detailed description of methods, architectures, or algorithms.  This includes system diagrams, architectural layers, protocols, mathematical models – anything that could become a reusable component.
- **Datasets/Environment**: What data or environment is used (simulated or real, scale of dataset, hardware setup, cloud services, specialized devices).  This category supports understanding of *scale assumptions* and engineering context.
- **Evaluation Metrics & Results**: The performance outcomes, quantified.  Include exact metrics (latency, throughput, accuracy, memory use, etc.) and numeric results.  If results are only in plots, note the extracted values.  Flag which comparisons are being made.
- **Operational Characteristics**: Any information on latency, throughput, concurrency limits, scalability behavior, cost model (e.g. “runs in 4 ms per request”), and infrastructure requirements.  E.g. “The system assumes an SSD for caching” or “requires a GPU with >=16GB memory.”
- **Security/Privacy**: If mentioned, note data protection or security constraints (e.g. “uses differential privacy”, “trust model assuming honest nodes”).
- **Reliability/Fault Tolerance**: Any discussion of failure modes or robustness (e.g. “graceful degradation under load”).
- **Assumptions & Preconditions**: Explicit assumptions (e.g. “Assumes centralized coordinator”) or hidden ones (e.g. maybe implied like “the network is synchronous”).  List them in an *Assumptions* field.
- **Limitations and Future Work**: From the paper’s conclusion or discussion: what the authors say about limitations (e.g. “only tested on single-threaded CPU”).
- **Evidence Snippets**: Actual quotes or data excerpts supporting key points (linked to pages/paragraphs).
- **Reusability/Generality**: Notes on whether a mechanism is general-purpose. For example, if a technique is described as generic or applicable to other domains, mark it as *reusable*; if it’s heavily tuned for one task, mark it accordingly.

Categories often **missing in ordinary summaries** but critical here include: **explicit assumptions**, **failures/faults**, **infrastructure requirements**, **observability needs**, and **quantified limitations**.  For system design, we must know “under what conditions this works” – so fields like *ScaleAssumptions*, *Hardware*, *CostDrivers*, *Observability* are high-value and should not be omitted.

Cross-paper synthesis will rely on comparability, so fields that should always exist for cross-contrast are: **Problem**, **Claims**, **Metrics/Results**, **Limitations/Failure Modes**, and any **architectural components**.  Inconsistencies or overlaps across these fields can then be automatically aligned.

## 6. Evidence and Uncertainty Handling Model / 证据与不确定性处理模型

Step 1 must **track provenance and confidence** for every extraction.  We recommend:

- **Citation Anchors**: Every non-trivial claim or datum should carry a reference to its location in the paper (e.g. “Source: [Title, page X, paragraph Y]” or direct URL).  This can be an explicit field or annotation.  If the LLM generates a fact, it must attach the *source quote* or a paraphrase with page number【20†L272-L280】【60†L497-L500】.
- **Statement Tagging**: Classify entries as *Author-Claim*, *Empirical-Result*, *Interpretation*, or *Model-Inference*.  For example, if the paper says “We assume X,” label that as Author-Claim; if the LLM extrapolates beyond the text, mark *Inferred*.  This distinction helps later judges weigh what is factual vs speculative.
- **Evidence Snippets**: For key fields, capture actual text snippets.  For instance, if recording the claim “Algorithm A scales linearly,” include the sentence that states this or the data row that shows it.  This grounds the record.
- **Confidence Scores / Flags**: Include a confidence value or ordinal flag (e.g. *high/medium/low confidence*) on each field.  These can be model‑provided (if available) or filled by an evaluator pass (e.g. “I’m 90% sure this matches the paper’s claim”).  Low-confidence or ambiguous extractions should be marked for later review.
- **Explicit Uncertainty Field**: For subjective fields (e.g. “strength of evidence” or “consensus level”), have the extractor indicate if the paper’s support is weak.  E.g. “EvidenceStrength: weak” if a result is anecdotal rather than statistically validated.
- **Contradiction Marker**: If an extraction conflicts with common knowledge or prior extractions, flag it (the contradiction analysis step can use this).  For example, if a paper claims higher accuracy but uses a simpler dataset than others, we should note the tension.
- **Provenance Logging**: In addition to content fields, log metadata about the extraction itself (model used, prompt version, timestamp).  This ensures we can trace back if errors are found in Step 2 synthesis.

Ultimately, every fact in Step 1 should form an “evidence package” consisting of (fact, source, confidence).  This is essential so that the synthesis agent can justify conclusions: e.g. “Claim X holds (C=0.92) because paper Y stated Y (p. 5)【20†L272-L280】【60†L497-L500】.”  Without this, the system design derived later would lack verifiable foundations.

## 7. Recommended Step 1 Schema Options / 推荐的第一步输出模式选项

We compare several output formats:

- **Narrative Markdown**: Prose paragraphs with markdown headings/bullets. *Pros:* Readable for humans; allows free explanation. *Cons:* Hard to programmatically parse and validate; model might introduce extraneous text or hallucinations. Good as a *fallback* if stricter formats fail or for initial brainstorming, but risky as primary output.
- **Structured Markdown**: A hybrid using explicit labels or headings for fields (e.g. `## Methods`, `## Results` sections with bullet points). *Pros:* More machine-readable than free prose; still relatively human-friendly. *Cons:* Lacks strict schema enforcement (different papers may use slightly different phrasing or ordering), so some NLP post-processing is needed.
- **JSON/YAML (Strict Schema)**: The model outputs JSON or YAML with fixed keys (e.g. `"Claims": [...], "Methods": "...", "Results": {"throughput":..., "latency":...}, "Limitations": "..."`). *Pros:* Fully structured and machine-parseable; fields can be strongly validated; easy to run downstream code on. *Cons:* Less forgiving to LLM errors (syntax mistakes break JSON); less readable for quick human inspection; requires very clear prompts or function-calling support in the model.
- **XML-like Tagging**: Similar to JSON but with XML tags (`<claim>...</claim>`).  *Pros:* Very explicit structure, and some LLMs (e.g. Claude’s XML support) handle this well【37†L198-L207】. *Cons:* Token-inefficient (lots of tag overhead) and less standard in modern pipelines.
- **Function Calling / TOON**: Emerging approaches like function-call interfaces or TOON (token-optimized notation) aim to get structured output efficiently【37†L159-L169】. *Pros:* Reduces repetition and enforces schema. *Cons:* Cutting-edge and may not yet be supported in all models.

**Trade-offs:**
Structured outputs (JSON/XML) reduce hallucination and improve downstream processing【37†L214-L223】【37†L248-L257】, but improper use can inhibit the LLM’s reasoning. For example, requiring the “answer” field first can tempt the model to jump to conclusions【37†L214-L223】. Positioning “reasoning” before “answer” encourages chain-of-thought. As a result, a hybrid approach often works best: allow the model to “think” (optionally in an unconstrained initial pass), then enforce structure on the final extraction step.

**Recommendation:**
- **Primary Schema:** A strict JSON (or YAML) output schema.  Define required keys with clear prompts. Use function-calling or schema conformance checks if possible. Enforce ordering of reasoning/evidence fields before conclusions to guide the model【37†L214-L223】. JSON maximizes machine-readability and ensures consistent data for synthesis.
- **Fallback Schema:** A structured Markdown template (with headings and bullet lists).  If JSON output is unstable, a well-defined markdown form (perhaps with automatic parsing rules) is acceptable. For example:

```markdown
**Title:** …
**Year:** …
**Context:** …
**Problem:** …
**Methods:** …
**Results:** …
- *Latency:* …
- *Throughput:* …
- *Dataset:* …
**Limitations:** …
**Assumptions:** …
**References/Evidence:** …
```

In pilot experiments, JSON with carefully placed “reasoning” fields has matched or exceeded free-form output accuracy【37†L214-L223】【54†L83-L86】. Whichever format is chosen, validation is essential (e.g. checking JSON validity, ensuring no field is empty). The schema should be documented so that each field’s purpose (for Step 2) is clear.

## 8. Recommended Step 1 Workflow / 推荐的第一步工作流程

A practical multi-pass workflow for each paper might be:

1. **Document Ingestion**: Convert PDFs to text (OCR if needed). Identify sections (title, abstract, intro, etc.). Tag figures and tables if relevant.
2. **First-Pass Skim (Concept Extraction)**: Use the LLM to read the paper (e.g. abstract, intro, conclusion) and generate a bulleted list of candidate *topics* and *claims*.  This can be an unconstrained or lightly structured “brainstorm” to highlight what’s important. (Low-medium reasoning effort.)
3. **Extraction Pass (Structured Fill)**: Prompt the LLM with the paper text (or abstract) plus the final schema template. Instruct it to fill each field. For JSON, ask explicitly for valid JSON with the required keys. For markdown, ask for each section. This is the core extraction step. (Medium reasoning effort; model focuses on filling fields precisely.)
4. **Evidence Pass (Anchoring)**: Independently (or within the same prompt), ask the LLM to supply source anchors. For each claim or result field, it should output the cited text snippet and page number. Alternatively, this pass can cross-check the filled fields by asking “Which part of the paper justifies this?” (High reasoning effort.)
5. **Quality-Control Pass**: Run a **validator prompt** or simple code to check completeness (no empty critical fields) and consistency (e.g. numerical values parse as numbers). Optionally have the LLM critique its own output or answer a set of yes/no rubric questions.
6. **Normalization**: Normalize terminology (e.g. “Mb/s” to “Mbps”), units, and formatting so that fields are consistent across papers.
7. **Final Review**: A human or automated review of flagged issues. This might include seeing if any *redundant claims* (the same info captured twice) or *missing fields* remain. Corrections can be fed back into the workflow.

**One-pass vs Multi-pass:**  We strongly advocate multi-pass.  Empirical studies show a two-step extraction (identify entities, then relations) is markedly more accurate than a single shot【54†L83-L86】.  Similarly, “separating thinking from formatting” avoids the reasoning degradation seen in one-step structured prompts【37†L248-L257】.  In practice, the modest extra time for a second pass is justified by the higher fidelity: the first pass can use chain-of-thought to gather insights, and the second pass can strictly conform to the schema.  This rolling state update avoids losing early cues【38†L177-L186】.  Thus, **Step 1 should be at least two-pass**: a free-interpretation pass to gather candidates, followed by a structured output pass.  Additional validator or reviewer passes improve robustness.

## 9. Prompting and Model Strategy / 提示词与模型策略

We recommend leveraging GPT-5.4 or Codex with these prompting strategies:

- **Schema-Driven Prompt**: Provide the LLM with the exact output schema (in JSON or template form) in the system or user prompt. For example: *“Extract the following fields in JSON format: {Title, Year, Context, Problem, Methods, Results, Limitations, Assumptions, Evidence}. Ensure valid JSON.”*  This helps enforce structure from the start.
- **Chain-of-Thought (CoT)**: For complex inference (e.g. assessing support for an assumption), use a *“moderate to high reasoning”* approach: explicitly ask the model to think step-by-step before finalizing a field. Example: *“First identify key claims in the abstract with brief reasoning, then fill the ‘Claims’ field.”*  Alternatively, adopt the 2-step approach: let the model outline in text, then convert to structured output【37†L248-L257】【54†L83-L86】.
- **Multi-Turn Interactions**: Instead of one giant prompt, break it into multiple sub-questions. E.g., ask “What are the paper’s main contributions?”, capture the answer, then ask “For each contribution, provide the supporting quote.” Then ask to reformat these into the final schema.
- **Few-Shot Examples**: If using structured output, include 1-2 examples in the prompt that show the model how to map source text to the schema.
- **Model Choice**: For purely textual extraction, GPT-5.4 is suitable; for strict format enforcement, Codex or models with robust JSON parsing may excel. Use “xhigh” reasoning settings for fields requiring judgment (assumptions, limitations), and “medium” for straightforward fields (e.g. title, numeric results).
- **Validation Prompt**: After the extraction, run a secondary prompt like “Given the extracted JSON, is each claim accurately supported by the provided evidence? List any inconsistencies.”  This can catch hallucinations.
- **Zero-Shot vs Few-Shot**: We anticipate having a fixed schema, so after initial engineering we prefer zero- or one-shot prompting (the schema itself serves as the “few-shot example”).
- **Error-Handling**: Use temperature=0 or low to minimize randomness. If the model outputs malformed JSON, use a follow-up prompt: “The JSON above has a syntax error. Please fix it.”

In summary, **structured prompts with clear output instructions** are key. We suggest a **two-phase prompting**: (1) an unconstrained reasoning phase (Chain-of-Thought) to gather info, and (2) a constrained output phase to fill the schema【37†L248-L257】【54†L83-L86】.  This aligns with best practices for “complex reasoning tasks”【37†L248-L257】. Validator passes should be used liberally to verify each field’s support and consistency.

## 10. Quality Assurance Framework / 质量保证框架

To ensure Step 1 is “good enough” for synthesis, we propose:

- **Acceptance Criteria**:
  - *Completeness:* All required fields are non-empty for each paper.
  - *Traceability:* Every key fact in each field has a source citation or quote attached.
  - *Accuracy:* Fields correctly reflect the paper (no obvious errors or hallucinations).
  - *Clarity:* Entries are specific (e.g. numerical values not “high”) and units are consistent.
- **Review Checklist** (for manual or automated checks):
  - [ ] *Meta Fields:* Title/year/ID match the source.
  - [ ] *Claims vs Methods:* Clear distinction between *what* the paper claims and *how* it does it.
  - [ ] *Evidence:* For each claim or result, verify at least one direct quote/link is recorded.
  - [ ] *Consistency:* No contradiction between fields (e.g. “Results” field values align with quoted figures).
  - [ ] *Units/Scale:* Units are standardized (e.g. ms vs seconds); scale assumptions noted.
  - [ ] *Implied info:* Assumptions/limitations are present even if implicit (e.g. check common ones like scale-out limits).
- **Scoring Rubric**: Rate each paper’s extraction (0–5) on categories like *Coverage*, *Factual Accuracy*, *Provenance*, *Completeness*, *Clarity*. For instance:
  - *5 – Excellent:* All fields present, well-cited, no errors.
  - *3 – Acceptable:* Minor omissions or one missing citation, but mostly usable.
  - *1 – Poor:* Major fields missing or incorrect, many hallucinated facts.
- **Red Flags (warnings)**:
  - Fields with **no citation** (requires manual review).
  - Generic or vague entries (e.g. “Claims: as above”).
  - Conflicting data (e.g. a metric that doesn’t match the quoted value).
  - Overly high certainty without evidence (e.g. claiming “provably the best” with no quote).
  - No “Limitations” field when it should exist (most papers have some).
- **Evaluation Harness/Benchmark**:
  - Build a small **gold-standard dataset**: select representative papers, manually create the ideal extraction according to the schema.
  - Then run alternative prompting templates or model versions on the same papers and compare. Measure precision/recall of extracted facts.
  - Use automated checks like: do extracted numeric results match the paper’s reported tables/graphs? (e.g. text extraction of numbers vs system output).
  - Example benchmark: for factual fields, compute BERTScore or exact match between output and reference extraction (treat fields independently). For structure, check JSON schema validity rate.
  - Over time, track these metrics to tune the extraction prompt and schema (akin to how SciTrue evaluated traceability【30†L1-L4】【58†L415-L423】).
  - Optionally, use peer-review: have multiple agents (or models) do the extraction and cross-validate each other, similar to contradiction detection.

By routinely scoring extractions and reviewing samples, the team can calibrate the template and prompt.  As Google’s *ML Test Score* emphasizes, numerical thresholds help – e.g. “≥90% of extractions must have evidence quotes”【46†L236-L244】.  Such quantitative QA guards Step 1 quality so that Step 2 can proceed on a solid basis.

## 11. Recommended Final Methodology / 推荐的最终方法

**Step 1 Methodology:** For each selected paper, apply a **two-pass extraction**.  First, ask the LLM to read the paper (or abstract/introduction) and list all key elements (contributions, methods, metrics) with rough reasoning. Second, feed this (or the text) to a structured prompt for filling the schema.  Use validation prompts to attach evidence and check consistency.  Store outputs as structured objects (JSON records).  Maintain both human-readable notes and machine‑readable files for each paper.

Concretely, we recommend:
- A **JSON template** with fields for `Title, Authors, Year, Context, Problem, Methods, Datasets, Results, Performance{latency,throughput}, Limitations, Assumptions, ReuseCandidates, SecurityImplications, EvidenceSnippets, ConfidenceScores`.
- A multi-step **prompt/agent workflow** (analogous to ResearchPilot’s ExtractionAgent) that enforces this schema【50†L146-L148】.
- **Cross-check step**: after all papers are processed, run a contradiction analyzer (Step 2) as a sanity check on fields like “Results” and “Claims” (e.g. flag if two papers irreconcilably differ on a core claim【56†L37-L40】).
- Store the structured extractions in a database or spreadsheet (as LitPilot does with an Excel tracker【20†L272-L280】) for human browsing if needed.

This pipeline ensures that Step 2 (synthesis) receives **rich, precise inputs**: every claim has a source, every assumption is explicit, and no field is left to guesswork.  It becomes straightforward to programmatically build synthesis matrices, spot contradictions, and enumerate patterns directly from these records.

## 12. Minimum Viable Template / 最低可行模板

A minimal Step 1 schema might include:

- **Title:**
- **Year:**
- **Context/Problem:** (Brief statement of the problem or domain)
- **Approach/Method:** (One-line summary of what was done)
- **Key Results:** (Numeric outcomes, e.g. “Throughput: 10K ops/s, Latency: 1ms”)
- **Limitations/Assumptions:** (Any obvious constraint)
- **Evidence (Quote):** (Exact quote supporting the *main claim*)

For example (in structured form):

```json
{
  "Title": "High-Throughput Cache Design (2025)",
  "Year": "2025",
  "Context": "Improving web cache scalability in data centers.",
  "Problem": "Existing caches saturate under high concurrency.",
  "Approach": "Proposes a lock-free algorithm using multiple shards.",
  "Results": {"Throughput": "10M req/s", "Latency": "0.5ms"},
  "Limitations": "Only evaluated on workload X; relies on custom hardware.",
  "Evidence": "\"our throughput reaches 10M/s\" (p.12)"
}
```

This **minimum template** ensures that the downstream synthesis at least sees *what was done*, *what was achieved*, and *what limitations were reported*, with a concrete evidence snippet.  Every field directly serves cross-paper comparison (e.g. all “Results” can be tabulated).

## 13. Advanced Template / 高级模板

A more advanced schema might expand the minimum fields. For instance:

```json
{
  "Title": "",
  "Authors": "",
  "Venue": "",
  "Year": "",
  "ProblemStatement": "",
  "Contributions": ["", "..."],
  "Methods": "...",
  "Dataset/Env": "...",
  "Software/Tools": "",
  "Performance": {
    "Throughput": "",
    "Latency": "",
    "Accuracy": "",
    "ResourceUsage": {"CPU": "", "Memory": ""}
  },
  "ScaleAssumptions": "",
  "CostDrivers": "",
  "Security/Privacy": "",
  "ObservabilityImplications": "",
  "Reliability/FailureModes": "",
  "ReusabilityNotes": "",
  "Limitations": "",
  "Assumptions": "",
  "OpenIssues": "",
  "EvidenceQuotes": [
    {"field": "Contributions", "quote": "", "page": ""},
    {"field": "Performance", "quote": "", "page": ""}
  ],
  "Confidence": ""
}
```

- *ReusabilityNotes*: Describe which parts could generalize.
- *OpenIssues*: Any unknowns or follow-up questions.
- *EvidenceQuotes*: A small array of key quotes linking fields to sources.
- *Confidence*: A summary confidence score or level.

An example Markdown excerpt might look like:

> **Methods:** We use a *multi-threaded pipeline* with local caching (quoting: *“We implement caching shards to parallelize reads”*).
> **Results:** Sustains **200K ops/s** at 500µs latency on 8-core CPU (p.9).
> **Assumptions:** Assumes data fits in L2 cache; fault tolerance not addressed.

This advanced template, while more verbose, captures finer details needed for engineering design (e.g. hardware assumptions, observability).  It separates concerns so that future designers can quickly find relevant pieces (like security or cost) without wading through prose.

## 14. Anti-Patterns / 反模式

Common mistakes to avoid in Step 1 extraction:

- **Free-text Narrative Summarizing:** Allowing long prose answers. This leads to the “salad” problem where facts are buried. Avoid answers like, “The paper then discusses….” Instead, stick to the template fields.
- **Missing Evidence Links:** Any claim without a citation should be rejected. For instance, “This result is state-of-the-art” without quoting the paper’s assertion is a red flag.
- **Mixing Papers:** Do not combine insights from multiple papers in one entry. Each record is for a single paper only.
- **Broad Generalizations:** Statements like “always” or “never” that are not in the text. These are likely hallucinations. Check if the source explicitly claimed it.
- **Omitting Scale and Env Details:** Phrases like “we assume normal scale” are unhelpful – prefer explicit numbers or scenarios.
- **No Uncertainty Handling:** Giving a definite score or claim without noting if it was experimentally verified or only simulated. Every inference beyond the text must be flagged.
- **Inconsistent Units/Format:** e.g. writing “Throughput: 100” without units. This makes cross-paper comparison impossible. Normalize these immediately.
- **Answer-First JSON:** Putting an “Answer” field before reasoning in JSON encourages jumping to conclusions【37†L214-L223】. Ensure fields like *Reasoning* or *Evidence* come first.
- **Empty or Excessive Fields:** A field left blank or filled with “N/A” is a loss; conversely, dumping the full abstract text under a field clutters the data. Stick to the key phrases or quotes that answer the field’s purpose.

These anti-patterns lead to noise and mistrust. By adhering to the structured template and citing sources, we avoid these pitfalls【38†L177-L186】【56†L37-L40】.

## 15. Open Questions / 开放问题与未来改进

While the above methodology is grounded in current best practices, several open issues remain:

- **Full-Text vs Abstract Extraction:** Can we scale to ingest entire papers (especially long systems papers) rather than abstracts? *ResearchPilot* only handles abstracts by default, citing “abstract-only extraction” as a limitation【50†L21-L23】. Processing full papers with LLMs (while retaining context) is an ongoing challenge.
- **Dynamic Schema Evolution:** As we synthesize findings, new needed fields may emerge (e.g. “energy consumption”). How should Step 1 templates adapt over time? Automating schema updates is non-trivial.
- **Contradiction Detection:** Automatically spotting contradictions across papers is still hard. Tools like SciTrue【60†L497-L500】 and dedicated contradiction models can help, but integrating these into the pipeline (Step 1 to tag potential contradictions) needs more research.
- **LLM Hallucinations and Bias:** Even with prompts, LLMs may invent plausible-sounding data. Developing methods (like SciTrue’s evidence-trace methods【60†L497-L500】) to catch hallucinations remains a critical area.
- **Benchmarking Extraction Quality:** How to quantitatively evaluate Step 1 at scale? Manual QA is slow. Building public benchmarks (similar to AID’s query dataset【52†L203-L208】) or open corpora of paper-extraction pairs would accelerate progress.
- **Multi-Language Papers:** Technical papers in other languages might be relevant. Extending Step 1 to handle or translate non-English papers is an unresolved extension.
- **Human-in-the-Loop Tuning:** What is the optimal human/AI balance? For instance, should experts correct extractions on-the-fly and feed that back to fine-tune models? The best interface for collaboration is an open question.
- **Tooling and Automation:** Developing extraction tools (like LitPilot or ResearchPilot) that allow visual inspection of linked quotes, version control of schemas, and branching design paths is needed. The community lacks standardized platforms for this multi-stage process.

In summary, while the proposed Step 1 framework sets a high bar for engineering literature analysis, it should evolve with advances in LLM capabilities (e.g. future GPT versions, RAG with improved traceability【28†L94-L103】) and with feedback from actual system design experiences. Ongoing refinement – guided by metrics from the QA harness – will be crucial to improving both the extraction quality and its integration into downstream synthesis.

**Sources:** Concepts above are drawn from recent AI systems for literature analysis【50†L33-L39】【54†L83-L86】【20†L272-L280】【38†L177-L186】【60†L497-L500】, as well as empirical studies of summarization pipelines【38†L51-L59】 and user expectations【52†L203-L208】. Each recommended field and process is motivated by maximizing the fidelity and utility of the extracted information for later engineering synthesis.
