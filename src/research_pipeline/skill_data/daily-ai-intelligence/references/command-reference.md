# Command reference

Use `research-pipeline brief run --registry <path>` for a full local run.

Stage commands:

- `brief poll`: writes raw and normalized event JSONL.
- `brief rank`: writes clusters and ranked clusters.
- `brief generate-daily`: writes `reports/daily.md`.
- `brief validate`: writes `validation/validation.json`.
- `brief feedback`: records explicit local feedback.
- `brief export-obsidian`: writes daily/topic/source notes inside the vault.
- `brief dossier`: generates one selected hot-topic dossier.
- `brief dossier --auto --max-count 1`: auto-generates limited dossiers.
- `brief weekly-synthesis`: creates a weekly trend memo from daily reports.
- `brief topic-aliases`: reviews durable topic alias suggestions.
- `brief preferences`: computes or rolls back explicit-feedback preferences.
- `brief resume`: resumes from saved briefing artifacts.
- `brief compare-sources`: compares base vs expanded source registries.
