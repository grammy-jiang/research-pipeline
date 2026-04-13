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

For each key concept, identify alternate terminology authors may use:

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
