# Hook Setup — daily-ai-intelligence

The Stop lifecycle hook enforces the brief completion gate deterministically.
When the hook fires and `check_completion.sh` detects a missing or invalid
brief, the agent is blocked from stopping and receives a description of the
failure. The agent must resolve the issues before the session ends.

## How the Stop hook works

At each turn end (`Stop` / `agentStop`):

1. Locates `check_completion.sh` in the skill's install directory.
2. `check_completion.sh` looks for today's brief workspace (`./workspace/`).
3. If no brief has been run today → exit 0 (no-op, no disruption).
4. If today's brief exists, checks `validation/validation.json`.
5. **Pass** → exit 0 → agent stops normally.
6. **Fail** → exit 2 → agent receives the failure output as context and
   continues working until the brief passes validation.

---

## Claude Code (automatic — no setup required)

The `hooks:` block in this skill's YAML frontmatter is loaded by Claude Code
whenever the skill is active. **No manual registration is needed.**

For global registration (hook active outside skill invocation), merge
`hooks/claude-code-hooks.json` into `~/.claude/settings.json`:

```bash
cp ~/.claude/settings.json ~/.claude/settings.json.bak

jq -s '.[0] * .[1]' \
  ~/.claude/settings.json \
  ~/.claude/skills/daily-ai-intelligence/hooks/claude-code-hooks.json \
  > /tmp/merged.json && mv /tmp/merged.json ~/.claude/settings.json
```

Verify with `/hooks` inside Claude Code.

---

## GitHub Copilot CLI

Copilot CLI does not support skill-scoped hooks. Deploy the hook **project-locally**
in each repository where you run daily briefs (not in `~/.copilot/hooks/` globally).

```bash
# In your BRIEFING PROJECT root:
mkdir -p .github/hooks
cp ~/.copilot/skills/daily-ai-intelligence/hooks/copilot-hooks.json \
   .github/hooks/daily-ai-intelligence.json
```

> **Tip:** Copilot CLI also reads `.claude/settings.json` in project roots.
> If you already have Claude Code hooks configured there, Copilot will pick
> them up automatically — no need to add the `.github/hooks/` file.

---

## Codex CLI

Codex CLI does not support skill-scoped hooks. Deploy the hook **project-locally**
in each repository where you run daily briefs (not in `~/.codex/hooks.json` globally).

1. Enable hooks in `.codex/config.toml` (project-local):

   ```toml
   [features]
   codex_hooks = true
   ```

2. Copy the hook config to `.codex/hooks.json` in your briefing project:

   ```bash
   # In your BRIEFING PROJECT root:
   mkdir -p .codex
   cp ~/.agents/skills/daily-ai-intelligence/hooks/codex-hooks.json \
      .codex/hooks.json
   ```

---

## Testing the hook

```bash
# Run manually — should exit 0 (no-op) when no brief workspace is present
bash ~/.claude/skills/daily-ai-intelligence/hooks/stop-check.sh; echo "Exit: $?"
```

---

## Uninstalling

- **Claude Code**: Remove `hooks:` block from `SKILL.md` or set
  `"disableAllHooks": true` in settings.
- **Copilot CLI**: `rm .github/hooks/daily-ai-intelligence.json` in the project.
- **Codex CLI**: Remove the `Stop` block from `.codex/hooks.json` and set
  `codex_hooks = false` in `.codex/config.toml` if no other hooks are configured.
