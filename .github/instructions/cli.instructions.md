---
applyTo: "src/research_pipeline/cli/**/*.py"
---

- Each CLI subcommand lives in its own file: `cmd_<name>.py`.
- Register new commands in `app.py` using `app.command()`.
- Use `typer.Option()` with help strings for all parameters.
- Log progress with the `logging` module, never `print()`.
- Follow the existing pattern: load config → resolve paths → execute → log result.
- Always accept `--run-id` for stage commands that operate within a run context.
