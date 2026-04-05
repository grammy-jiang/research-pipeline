---
paths:
  - "src/research_pipeline/models/**/*.py"
---

# Pydantic model conventions

- All domain objects must use Pydantic BaseModel or dataclasses.
- Use `model_validator` for cross-field validation.
- Provide `model_config` with `frozen=True` for immutable models where possible.
- Export models from the package `__init__.py`.
- Ensure models are JSON-serializable for manifest storage.
- Add roundtrip tests: construct → serialize → deserialize → assert equal.
