# Proof Pack — A09: Report and Event Validator

**Ticket:** A09_report_and_event_validator
**Status:** VERIFIED
**Date:** 2025-01-20
**Verified Commit:** 19c27936f87a482d2ae885bc6779a5588f39911f

## Feature Summary

A09 implements comprehensive validation for Daily AI Intelligence briefing artifacts:

1. **Daily Report Validator** (`validate_daily_report()`)
   - Checks required sections (Agent Read Map, Executive Signal, Top Items, Suppressed/Not Reported, Follow-up Queue, Feedback Targets)
   - Enforces link budget (≤15 for active days, ≤5 for low-signal days)
   - Enforces word count (900–1400 for active days ≥6 items; no limit for low-signal)
   - Requires "## No Material Updates" section for low-signal days
   - Detects duplicate cluster titles
   - Verifies evidence URLs are present in markdown
   - Validates evidence type labels ([FACT]/[INFERENCE]/[WATCH])
   - Rejects boilerplate/template text

2. **Dossier Report Validator** (`validate_dossier_report()`)
   - Checks required sections specific to dossier format
   - Validates factuality labels for evidence

3. **Obsidian Note Validator** (`validate_obsidian_note()`)
   - Verifies required YAML frontmatter
   - Validates Agent Read Map structure

4. **Serialization** (`validation_to_json()`)
   - Converts ValidationResult to JSON for artifact logging

## Acceptance Contract

### In-Scope Behavior

- Deterministic validation with Pydantic-based ValidationResult model
- Exactly 14 test cases covering all validators and error paths
- Offline tests with fixture-based cluster/event objects
- Zero network calls during test execution
- Proper Markdown parsing and validation

### Out-of-Scope Behavior

- LLM-based content evaluation (Phase A is template/extractive)
- Real-time validation during briefing generation
- Multi-language support (Phase A English only)

## Test Coverage

**Test File:** `tests/unit/test_briefing_validate.py`

### Test Results

```
tests/unit/test_briefing_validate.py::test_validate_daily_report_passes_valid_report PASSED [  7%]
tests/unit/test_briefing_validate.py::test_validate_daily_report_fails_missing_required_sections PASSED [ 14%]
tests/unit/test_briefing_validate.py::test_validate_daily_report_enforces_link_budget PASSED [ 21%]
tests/unit/test_briefing_validate.py::test_validate_daily_report_enforces_word_count_on_active_days PASSED [ 28%]
tests/unit/test_briefing_validate.py::test_validate_daily_report_detects_duplicate_cluster_titles PASSED [ 35%]
tests/unit/test_briefing_validate.py::test_validate_daily_report_verifies_evidence_urls_in_markdown PASSED [ 42%]
tests/unit/test_briefing_validate.py::test_validate_daily_report_checks_evidence_type_labels PASSED [ 50%]
tests/unit/test_briefing_validate.py::test_validate_daily_report_flags_low_signal_day_without_no_material_updates PASSED [ 57%]
tests/unit/test_briefing_validate.py::test_validate_daily_report_rejects_boilerplate_content PASSED [ 64%]
tests/unit/test_briefing_validate.py::test_validate_dossier_report_passes_valid_dossier PASSED [ 71%]
tests/unit/test_briefing_validate.py::test_validate_dossier_report_fails_missing_sections PASSED [ 78%]
tests/unit/test_briefing_validate.py::test_validate_obsidian_note_passes_valid_note PASSED [ 85%]
tests/unit/test_briefing_validate.py::test_validate_obsidian_note_fails_missing_frontmatter PASSED [ 92%]
tests/unit/test_briefing_validate.py::test_validation_result_serializes_to_json PASSED [100%]

============================== 14 passed in 0.33s ==============================
```

### Coverage Details

| Test Case | Description | Expected Behavior | Status |
|-----------|-------------|-------------------|--------|
| `test_validate_daily_report_passes_valid_report` | Valid 6-cluster active day report | Passes validation | ✓ PASS |
| `test_validate_daily_report_fails_missing_required_sections` | Missing Executive Signal section | Validation fails with error | ✓ PASS |
| `test_validate_daily_report_enforces_link_budget` | Link count exceeds 15 on active day | Validation fails with error | ✓ PASS |
| `test_validate_daily_report_enforces_word_count_on_active_days` | Word count below 900 on active day | Validation fails with error | ✓ PASS |
| `test_validate_daily_report_detects_duplicate_cluster_titles` | Duplicate cluster titles in markdown | Validation fails with error | ✓ PASS |
| `test_validate_daily_report_verifies_evidence_urls_in_markdown` | Missing evidence URL from markdown | Validation fails with error | ✓ PASS |
| `test_validate_daily_report_checks_evidence_type_labels` | Mismatched evidence type label | Validation fails with error | ✓ PASS |
| `test_validate_daily_report_flags_low_signal_day_without_no_material_updates` | Low-signal day missing required section | Validation fails with error | ✓ PASS |
| `test_validate_daily_report_rejects_boilerplate_content` | Boilerplate text in markdown | Validation fails with warning | ✓ PASS |
| `test_validate_dossier_report_passes_valid_dossier` | Valid dossier report | Passes validation | ✓ PASS |
| `test_validate_dossier_report_fails_missing_sections` | Missing dossier section | Validation fails with error | ✓ PASS |
| `test_validate_obsidian_note_passes_valid_note` | Valid Obsidian note with frontmatter | Passes validation | ✓ PASS |
| `test_validate_obsidian_note_fails_missing_frontmatter` | Missing YAML frontmatter | Validation fails with error | ✓ PASS |
| `test_validation_result_serializes_to_json` | ValidationResult to JSON serialization | Produces valid JSON | ✓ PASS |

## Quality Verification

### Linting (ruff)
```bash
$ uv run ruff check src/research_pipeline/briefing tests/unit/test_briefing_validate.py
All checks passed!
```

### Type Checking (mypy strict)
```bash
$ uv run mypy src/research_pipeline/briefing --strict
Success: no issues found in 27 source files
```

### Test Execution
```bash
$ uv run pytest tests/unit/test_briefing_validate.py -v
============================== 14 passed in 0.33s ==============================
```

## Implementation Notes

### Key Functions Validated

- **`validate_daily_report(markdown: str, clusters: list[BriefingCluster]) → ValidationResult`**
  - Core validator for daily briefings
  - Parses Markdown sections
  - Enforces all phase A contracts for briefing format

- **`validate_dossier_report(markdown: str) → ValidationResult`**
  - Validator for dossier-format reports
  - Checks dossier-specific section structure

- **`validate_obsidian_note(markdown: str) → ValidationResult`**
  - Validator for Obsidian vault notes
  - Verifies YAML frontmatter and metadata

- **`validation_to_json(result: ValidationResult) → dict[str, Any]`**
  - Serializes ValidationResult to JSON
  - Used for artifact logging and audit trails

### Test Fixtures

The test suite uses deterministic, offline fixtures:
- `_cluster()`: Creates BriefingCluster objects with configurable title, source, URL, etc.
- `_event()`: Creates IntelligenceEvent objects for dossier validation

### Deterministic Validation

All validators use:
- Exact string matching for section headers
- No regex or fuzzy matching
- Deterministic Markdown parsing
- Repeatable validation results across runs

## Dependencies

**Implementation:** `src/research_pipeline/briefing/validate.py` (pre-existing, fully implemented)
**Test Suite:** `tests/unit/test_briefing_validate.py` (new, 14 comprehensive tests)

### Runtime Dependencies
- Pydantic v2 (BaseModel, ValidationError)
- Standard library (re, json)

### Test Dependencies
- pytest
- research_pipeline.briefing.models (BriefingCluster, IntelligenceEvent, SourceClass, SourcePolicy, AccessMethod)
- research_pipeline.briefing.validate (all validators)

## Verification Commands

### Run Full Test Suite
```bash
uv run pytest tests/unit/test_briefing_validate.py -v
```

### Run with Coverage
```bash
uv run pytest tests/unit/test_briefing_validate.py --cov=src/research_pipeline/briefing --cov-report=term-missing
```

### Type Check
```bash
uv run mypy src/research_pipeline/briefing --strict
```

### Lint
```bash
uv run ruff check src/research_pipeline/briefing tests/unit/test_briefing_validate.py
```

### Format
```bash
uv run ruff format src/research_pipeline/briefing tests/unit/test_briefing_validate.py
```

## Files Modified

| File | Status | Changes |
|------|--------|---------|
| `tests/unit/test_briefing_validate.py` | NEW | 14 test functions covering all validators and error paths |
| `src/research_pipeline/briefing/validate.py` | UNCHANGED | Pre-existing implementation, fully functional |

## Audit Sign-Off

✓ All 14 tests pass
✓ Ruff linting passes
✓ MyPy strict type checking passes
✓ No fixture data issues
✓ Deterministic, offline validation
✓ Zero boilerplate/placeholder text
✓ Acceptance contract satisfied

**Readiness:** ✓ VERIFIED — Ready for A10
