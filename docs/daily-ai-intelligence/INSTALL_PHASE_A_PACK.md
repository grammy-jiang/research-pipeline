# Install Phase A Copilot Pack

## 1. Copy files

Copy `.github/`, `docs/`, and `prompts/` from this pack into the repository root.

## 2. Append Copilot instructions

Do not overwrite the existing Copilot instructions.

Append:

```text
.github/copilot-instructions.daily-ai-intelligence-phase-a.append.md
```

to the repository's existing Copilot instructions file.

## 3. Start Copilot CLI

```bash
copilot --autopilot --allow-all --max-autopilot-continues 20 \
  -p "$(cat prompts/daily-ai-intelligence/continue-current-ticket.md)"
```

## 4. First expected ticket

The first ticket is:

```text
A01_briefing_package_skeleton_and_models
```

## 5. Do not start Phase B

Phase B remains blocked until all Phase A tickets are `audit_pass`.
