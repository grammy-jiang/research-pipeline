# ADR-004: Backend Registry Pattern for PDF Conversion

## Status
Accepted

## Date
2024

## Context

PDF→Markdown conversion is the most variable part of the pipeline:
- Different backends have different quality, speed, and licence trade-offs
  (docling, marker, pymupdf4llm, mathpix, datalab, llamaparse, mistral_ocr,
  openai_vision, mineru)
- Some backends are local; others are cloud APIs requiring credentials
- Some backends are commercial; others are open source
- The optimal backend depends on the type of PDF (math-heavy, scanned, etc.)
- Users may have access to different sets of backends

A hard-coded `if backend == "docling"` approach would require modifying core
code to add a new backend and would not support runtime selection.

## Decision

Use a **registry pattern** for PDF conversion backends:

1. Each backend lives in its own module (`conversion/<name>_backend.py`)
2. Each backend subclasses `ConverterBackend` (from `conversion/base.py`)
3. Each backend registers itself with `@register_backend("name")` decorator
   (from `conversion/registry.py`)
4. The CLI and MCP server select a backend by name at runtime
5. Optional dependencies are declared per-backend in `pyproject.toml` extras
6. `FallbackConverter` (`conversion/fallback.py`) wraps multiple backends and
   tries them in order if one fails

Additionally, cloud backends support **multi-account rotation**: when a quota
or rate limit is hit, the pipeline automatically rotates to the next configured
account via `[[conversion.<backend>.accounts]]` TOML configuration.

## Consequences

**Positive:**
- Adding a new backend requires no changes to core pipeline code
- Users choose their backend via configuration only
- Multi-account rotation provides resilience against API quota exhaustion
- FallbackConverter provides graceful degradation

**Negative:**
- The registry requires backends to be imported for registration (handled by
  importing the backend module at CLI startup)
- Licence concerns: `marker` (GPL-3.0) and `pymupdf4llm` (AGPL-3.0 via PyMuPDF)
  cannot be included in default installs without copyleft implications

**Licence policy**: The CI pipeline checks for GPL/AGPL dependencies and fails
if they appear in non-optional installs.
