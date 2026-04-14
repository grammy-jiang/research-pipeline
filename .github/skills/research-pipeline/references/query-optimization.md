# Query Plan Optimization

After the CLI generates `query_plan.json`, review and improve it before
searching. Auto-generated plans often produce narrow, synonym-blind queries.

## Validation Checklist

Apply ALL of these before proceeding to search:

### 1. Remove Stop Words

Remove "of", "the", "for", "in", "a", "an" from `must_terms`.
These waste query slots.

### 2. Cap `must_terms` to 2 (max 3)

More AND-ed terms exponentially reduce recall. Move excess to `nice_terms`.

### 3. Generate Synonym-Expanded Query Variants

The CLI now **auto-generates Q2D (Query-to-Document) variants** that mimic
academic phrasing (e.g., "this paper presents …", "we propose a method for …",
"a survey of …"). These are added automatically alongside keyword reordering
variants, so you get document-style recall for free.

You should still add **synonym-expanded variants** manually. For each key
concept, identify alternate terminology authors may use:

| Concept | Synonyms |
|---------|----------|
| AI agent | LLM agent, language model agent, autonomous agent |
| LLM | large language model, foundation model, AI model |
| harness | scaffold, framework, wrapper, orchestration |
| evaluation | benchmark, assessment, testing |
| engineering | design, architecture, implementation |
| multi-agent | multi-model, collaborative agents |

Create **at least 5 query variants** using different synonym combinations.
Each variant should use different vocabulary so collectively they cover
the full terminology space.

Example for "harness engineering of AI agents":
```
"AI agent harness engineering"
"LLM harness optimization"
"language model scaffold framework"
"agent evaluation benchmark framework"
"model harness code optimization"
```

### 3b. Expand Beyond Synonyms

Synonym expansion alone is insufficient. Also consider:

| Expansion Type | Examples | Why |
|---------------|---------|-----|
| **Acronyms** | RAG → Retrieval-Augmented Generation; RL → Reinforcement Learning | Many papers use only the acronym or only the full form |
| **Benchmark names** | MMLU, HumanEval, SWE-bench, MATH | Papers may not mention the task name, only the benchmark |
| **Dataset names** | MS MARCO, Natural Questions, HotpotQA | Important for retrieval/QA topics |
| **Canonical system names** | LangChain, LlamaIndex, AutoGPT, CrewAI | Papers often reference specific systems |
| **Task reformulations** | "code generation" ↔ "program synthesis" ↔ "automated programming" | Different communities use different terms |
| **Exclusion terms** | Add `negative_terms` for known irrelevant senses | e.g., "agent" in chemistry vs AI |
| **Terminology drift** | "prompt engineering" (2023) ↔ "instruction tuning" (2022) | Vocabulary shifts across years and subfields |

For each query variant, verify:
- Would this catch a paper that uses **only** the acronym?
- Would this catch a paper from a **different subfield** studying the same problem?
- Are there **known system names** that should be explicit search terms?

### 4. Include a Core-Concept-Only Variant

Use ONLY the most distinctive term without domain qualifiers.
E.g., for "harness engineering of AI agents":
`"model harness" OR "harness engineering"`

This catches papers that define the core concept without expected
domain vocabulary.

### 5. Cross-Check Vocabulary Coverage

Test: "Would a paper titled '<core concept> for <different domain term>'
be matched by at least one variant?" If not, add a variant.

### 6. Set Time Window

Match `primary_months` to user's requested window (default: 6 months).

**Write the improved plan back to `query_plan.json`** before searching.
