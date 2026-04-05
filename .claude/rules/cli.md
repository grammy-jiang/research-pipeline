---
paths:
  - "src/research_pipeline/cli/**/*.py"
---

# CLI command conventions

- Each CLI subcommand lives in its own file: `cmd_<name>.py`.
- Register new commands in `app.py` using `app.command()`.
- Use `typer.Option()` with help strings for all parameters.
- Log progress with the `logging` module, never `print()` or `typer.echo()` for
  operational output (version display is the only exception).
- Follow the existing pattern: load config → resolve paths → execute → log result.
- Always accept `--run-id` for stage commands that operate within a run context.
