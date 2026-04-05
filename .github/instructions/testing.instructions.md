---
applyTo: "tests/**/*.py"
---

- Never modify existing tests without explicit approval.
- Write tests first when implementing new features (TDD).
- Each test file maps to a source module: `test_<module>.py`.
- Mark tests requiring network access with `@pytest.mark.live`.
- Use VCR cassettes in `tests/fixtures/http_cassettes/` for HTTP-dependent tests.
- Run only the specific test under development: `uv run pytest tests/unit/test_foo.py::test_bar -xvs`
