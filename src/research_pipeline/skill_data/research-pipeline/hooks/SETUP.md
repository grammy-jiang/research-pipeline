# Hook Setup — research-pipeline

Two lifecycle hooks are provided:

| Hook event | Script | Purpose |
|------------|--------|---------|
| `Stop` / `agentStop` | `hooks/stop-check.sh` | Block agent from stopping when research is incomplete |
| `UserPromptSubmit` / `userPromptSubmitted` | `hooks/resume-inject.sh` | Inject prior session state before each prompt is processed |

Both scripts exit 0 immediately when no `round_state.json` is found in the
current working directory, so they are safe to leave active globally.

---

## How the Stop hook works

At each turn end (`Stop` / `agentStop`):

1. Checks for `round_state.json` in CWD. If absent → exit 0 (no-op).
2. Runs `scripts/check_completion.py` with `run_id` and `topic_slug`.
3. **Pass** → exit 0 → agent stops normally.
4. **Fail** → exit 2 → agent receives the failure output as context and
   continues working until all artifacts are in place.

## How the UserPromptSubmit hook works

Before each user prompt is processed:

1. Checks for `round_state.json` in CWD. If absent → exit 0 (no output, no-op).
2. Reads `run_id`, `topic_slug`, `current_round`, `status`, `open_gaps`.
3. Emits JSON to stdout: `{"hookSpecificOutput": {"additionalContext": "..."}}`
4. The agent automatically sees prior session state before it reads your prompt.
   It will resume — not restart — the research session when appropriate.

---

## Claude Code (manual registration required)

Claude Code does **not** auto-load hooks from a skill's frontmatter — this
skill's `SKILL.md` frontmatter carries only `name` / `description` / `license`,
with no `hooks:` block. To activate both hooks (Stop + UserPromptSubmit), merge
`hooks/claude-code-hooks.json` into `~/.claude/settings.json`:

```bash
# Backup existing settings
cp ~/.claude/settings.json ~/.claude/settings.json.bak

# Merge hooks with jq (both Stop and UserPromptSubmit)
jq -s '.[0] * .[1]' \
  ~/.claude/settings.json \
  ~/.claude/skills/research-pipeline/hooks/claude-code-hooks.json \
  > /tmp/merged.json && mv /tmp/merged.json ~/.claude/settings.json
```

Verify with `/hooks` inside Claude Code.

---

## GitHub Copilot CLI

Copilot CLI does not support skill-scoped hooks. Deploy the hooks **project-locally**
in each repository where you run research (not in `~/.copilot/hooks/` globally).

```bash
# In your RESEARCH PROJECT root:
mkdir -p .github/hooks
cp ~/.copilot/skills/research-pipeline/hooks/copilot-hooks.json \
   .github/hooks/research-pipeline.json
```

This ensures `agentStop` (Stop) and `userPromptSubmitted` only fire when you are
working in that project, not in unrelated sessions.

> **Tip:** Copilot CLI also reads `.claude/settings.json` in project roots.
> If you already have Claude Code hooks configured there, Copilot will pick
> them up automatically — you only need to add the `hooks` block once.

---

## Codex CLI

Codex CLI does not support skill-scoped hooks. Deploy the hooks **project-locally**
in each repository where you run research (not in `~/.codex/hooks.json` globally).

1. Enable the hooks feature flag in `.codex/config.toml` (project-local):

   ```toml
   [features]
   codex_hooks = true
   ```

2. Copy the hook config to `.codex/hooks.json` in your research project:

   ```bash
   # In your RESEARCH PROJECT root:
   mkdir -p .codex
   cp ~/.agents/skills/research-pipeline/hooks/codex-hooks.json \
      .codex/hooks.json
   ```

If `.codex/hooks.json` already exists, manually merge the `Stop` and
`UserPromptSubmit` blocks from `codex-hooks.json` into the existing file.

---

## Testing the hooks

```bash
# Create a minimal round_state.json (simulates an active session)
echo '{"run_id":"test-run","topic_slug":"test-slug","current_round":1,"status":"in_progress","open_gaps":[]}' \
  > round_state.json

# Test the Stop hook (should exit 2 and print missing-artifacts info)
bash ~/.claude/skills/research-pipeline/hooks/stop-check.sh
echo "Stop hook exit: $?"

# Test the UserPromptSubmit hook (should print JSON with additionalContext)
bash ~/.claude/skills/research-pipeline/hooks/resume-inject.sh
echo "Resume hook exit: $?"

# Clean up
rm round_state.json
```

---

## Uninstalling

- **Claude Code**: Remove the skill's `Stop` and `UserPromptSubmit` entries from
  `~/.claude/settings.json`, or set `"disableAllHooks": true` in settings.
- **Copilot CLI**: `rm .github/hooks/research-pipeline.json` in the project.
- **Codex CLI**: `rm .codex/hooks.json` (or remove the hook blocks) and set
  `codex_hooks = false` in `.codex/config.toml` if no other hooks are configured.
