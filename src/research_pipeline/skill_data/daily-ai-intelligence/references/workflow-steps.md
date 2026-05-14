# Workflow Steps — daily-ai-intelligence

> **Governed by the orchestrator.** The runner (`runners/runner.py`) drives
> all tasks automatically. Load this reference only to understand what a task
> does or to diagnose a failure — not to run commands manually.
>
> Manifest task IDs are shown in brackets: `[task-id]`. Mandatory gates are
> marked **⛔ GATE** — they cannot be skipped.

## Environment setup

```bash
# Claude Code / GitHub Copilot
SKILL_DIR=~/.claude/skills/daily-ai-intelligence

# Codex
# SKILL_DIR=~/.codex/skills/daily-ai-intelligence

REG=~/my-daily-registry.toml   # user's reviewed registry
WS=./workspace/briefing
DATE=$(date -u +%F)
```

On first use, copy and edit the registry template:

```bash
cp "$SKILL_DIR/config.toml" ~/my-daily-registry.toml
# Edit: enable real sources, set cadences, fixtures, trust weights
REG=~/my-daily-registry.toml
```

## Task `[validate-registry]` ⛔ GATE — Registry validation

The runner executes this gate before polling. Checks that at least one
source in the registry is enabled.

```bash
# Runner invokes this automatically. Direct invocation (debug only):
bash "$SKILL_DIR/scripts/validate-registry.sh" "$REG"
```

Exit 1 = stop and ask the user to enable at least one source.
Exit 0 = proceed.

## Tasks `[poll]` + `[rank]` + `[generate-daily]` + `[validate-brief]` — Core daily pipeline

```bash
# One-shot (runner preferred):
python3 $SKILL_DIR/runners/runner.py --registry "$REG" --workspace "$WS"

# Step-by-step (debug/partial reruns):
research-pipeline brief poll           --registry "$REG" --workspace "$WS" --date "$DATE"
research-pipeline brief rank           --workspace "$WS" --date "$DATE"
research-pipeline brief generate-daily --workspace "$WS" --date "$DATE"
research-pipeline brief validate       --workspace "$WS" --date "$DATE"
```

Artifacts produced:

- `<WS>/<DATE>/raw/*.jsonl` — raw events per source
- `<WS>/<DATE>/normalized/events.jsonl` — deduped, normalized
- `<WS>/<DATE>/clusters/ranked.jsonl` — ranked clusters
- `<WS>/<DATE>/reports/daily.md` — the validated daily brief
- `<WS>/<DATE>/validation/validation.json` — pass/fail + reasons

Use `research-pipeline brief resume --workspace "$WS" --date "$DATE"` to
skip stages whose artifacts already exist on disk.

## Task `[review-ranked]` — Optional reviewer gate (llm_reviewer)

When the runner encounters this task, it delegates to the `rank_reviewer`
sub-agent (see `runners/subagent_contracts/rank_reviewer.yaml`).

The reviewer checks deduplication, cluster coherence, ranking plausibility,
and source diversity. On `status: "rejected"`, manually reset `[rank]` to
`pending` in `workflow_state.json` and re-run the runner.

## Task `[validate-brief]` ⛔ GATE — Validation gate

The runner blocks on this gate. The brief must pass `validation.json`
before `[check-completion]` runs.

```bash
# Direct invocation (debug only):
research-pipeline brief validate --workspace "$WS" --date "$DATE"
```

Exit 1 = fix reported issues before delivering.

## Task `[check-completion]` ⛔ GATE — Completion gate

```bash
# Runner invokes this automatically. Direct invocation (debug only):
bash "$SKILL_DIR/scripts/check_completion.sh" --workspace "$WS" --date "$DATE"
```

Exit 1 = do not surface the brief. Fix issues first.
Exit 0 = brief is complete and validated; safe to deliver.

## Task `[dossier]` — Hot-topic dossier (optional)

Triggered when `--cluster-id` is passed to the runner.

```bash
research-pipeline brief dossier --workspace "$WS" --date "$DATE" --cluster <CLUSTER_ID>
# or auto-select:
research-pipeline brief dossier --workspace "$WS" --date "$DATE" --auto --max-count 1
```

## Task `[feedback]` — Feedback (optional)

See `references/feedback-loop.md` for the full contract.

```bash
research-pipeline brief feedback --workspace "$WS" --date "$DATE" \
    --cluster <CLUSTER_ID> --signal keep --reason "tracking this MCP RFC"
research-pipeline brief feedback --workspace "$WS" --date "$DATE" \
    --topic   <TOPIC_ID>   --signal too_noisy
research-pipeline brief feedback --workspace "$WS" --date "$DATE" \
    --source  <SOURCE_ID>  --signal less_like_this

research-pipeline brief preferences --workspace "$WS"            # apply
research-pipeline brief preferences --workspace "$WS" --rollback # undo
```

## Task `[export-obsidian]` — Obsidian export (opt-in, optional)

Triggered when `--vault` is passed to the runner.

```bash
# Always dry-run first:
research-pipeline brief export-obsidian \
    --vault    /path/to/Obsidian/AI-Intelligence \
    --registry "$REG" --workspace "$WS" --date "$DATE" --dry-run

# Execute:
research-pipeline brief export-obsidian \
    --vault    /path/to/Obsidian/AI-Intelligence \
    --registry "$REG" --workspace "$WS" --date "$DATE"
```

The export refuses to overwrite notes that lack generator-owned frontmatter.

## Task `[weekly-synthesis]` — Weekly synthesis (optional)

Triggered when `--week` is passed to the runner.

```bash
research-pipeline brief weekly-synthesis --workspace "$WS" --week "$(date -u +%G-W%V)"
```

## Topic alias review (periodic, not per-run)

```bash
research-pipeline brief topic-aliases --workspace "$WS" --list
research-pipeline brief topic-aliases --workspace "$WS" --approve <ALIAS_ID>
research-pipeline brief topic-aliases --workspace "$WS" --reject  <ALIAS_ID>
```

Durable topic aliases require explicit approval — never auto-approve.

## Schedule (cron example)

```cron
0 6 * * * cd ~/projects && research-pipeline brief run \
    --registry ~/my-daily-registry.toml \
    --workspace ~/workspace/briefing \
    >> ~/.local/state/daily-brief.log 2>&1
```

`--date` defaults to today (UTC); omit it for cron.
