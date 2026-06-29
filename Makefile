.PHONY: verify lint format typecheck test security audit

# Run all verification checks — must exit 0 before every commit and PR.
verify: lint typecheck test security

# Format & lint (ruff handles both)
format:
	uv run ruff format .

lint:
	uv run ruff format --check .
	uv run ruff check .

# Type checking (strict mypy — 0 errors required)
typecheck:
	uv run mypy src/

# Fast unit tests (no network)
test:
	uv run pytest tests/unit/ -x -q

# Security: secret scanning + SAST + dependency audit
security:
	uv run detect-secrets scan --baseline .secrets.baseline
	uv run bandit -r src/ -c pyproject.toml -q
	uv run pip-audit --ignore-vuln CVE-2026-3219 --ignore-vuln CVE-2025-3000

# Full pre-commit run (includes all of the above via hooks)
audit:
	uv run pre-commit run --all-files
