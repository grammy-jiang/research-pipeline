#!/usr/bin/env bash
# run-claude-issue-loop.sh
# Fix a batch of GitHub issues, one full cycle at a time, in dependency order.
#
# FULL CYCLE per issue:
#   1. git fetch; branch fix/issue-<n> off origin/<base>   (fresh trunk each time)
#   2. spawn the assigned agent (Claude Code or Codex) to fix + commit locally
#   3. push the branch
#   4. open a PR ("Closes #<n>")
#   5. WAIT for GitHub Actions CI to go green   <-- the only merge gate (no branch
#      protection on master, so nothing else stops a red merge)
#   6. squash-merge + delete branch
#   7. next issue branches off the now-updated origin/<base>  -> deps land in trunk
# Any failed step stops the batch (STOP_ON_FAIL, default on) so a later issue never
# builds on a broken / unmerged dependency.
#
# Two engines, load-matched (Copilot dropped — it hit its usage cap):
#   82 stale REQUIRED_SECTIONS  codex   S    mechanical test-derivation
#   81 no coherence check       claude  XL   keystone architecture (Opus max)
#   85 gate enhancements        codex   M-L  additive gate conditions + helper
#   83 trustworthy fixture      claude  L    600-line coherent regen (long ctx)
#   84 single-source dedup      codex   L    build/CI sync + dedup plumbing
#
# Usage:
#   ./run-claude-issue-loop.sh                 # full ordered batch, full cycle each
#   ./run-claude-issue-loop.sh 82              # single issue, full cycle
#   DRY_RUN=1 ./run-claude-issue-loop.sh       # print plan + prompts, do nothing
#   NO_MERGE=1 ./run-claude-issue-loop.sh 82   # fix+push+PR+wait CI, but DO NOT merge
#   ENGINE_81=codex ./run-claude-issue-loop.sh # override engine for one issue
#
# Tunables (env vars):
#   CLAUDE_MODEL   Claude model      (default: claude-opus-4-8[1m])
#   CLAUDE_EFFORT  Claude effort     (default: max)
#   CODEX_MODEL    Codex model       (default: gpt-5.5)
#   ENGINE_<n>     force engine for issue <n> (claude|codex)
#   REMOTE         git remote        (default: origin)
#   PR_BASE        PR base branch    (default: master)
#   MERGE_METHOD   squash|merge|rebase (default: squash)
#   CI_INTERVAL    CI poll seconds   (default: 20)
#   NO_MERGE       1 = stop before merge, leave green PR (default: 0)
#   STOP_ON_FAIL   1 = abort batch on first failed step (default: 1)
#   DRY_RUN        1 = plan only, spawn/push/merge nothing (default: 0)
#   SKIP_PERMISSIONS 1 = bypass all agent approvals (default: 1, see WARNING)
#
# WARNING — this script PUSHES branches and MERGES pull requests into
#   '${PR_BASE}' automatically once CI is green. It never force-pushes and never
#   writes to master except through a normal squash-merge of a CI-green PR, but
#   it IS an autonomous merge-to-trunk. It also defaults to bypassing agent
#   approval prompts (headless sessions cannot answer them). Run it only where
#   that is acceptable. Use NO_MERGE=1 to review each PR by hand before merging.

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

# --- config ----------------------------------------------------------------
CLAUDE_MODEL="${CLAUDE_MODEL:-claude-opus-4-8[1m]}"
CLAUDE_EFFORT="${CLAUDE_EFFORT:-max}"
CODEX_MODEL="${CODEX_MODEL:-gpt-5.5}"
REMOTE="${REMOTE:-origin}"
PR_BASE="${PR_BASE:-master}"
MERGE_METHOD="${MERGE_METHOD:-squash}"
CI_INTERVAL="${CI_INTERVAL:-20}"
NO_MERGE="${NO_MERGE:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_PERMISSIONS="${SKIP_PERMISSIONS:-1}"
BRANCH_PREFIX="${BRANCH_PREFIX:-fix/issue-}"
LOG_DIR="claude-issue-loop-logs"

DEFAULT_ORDER=(82 81 85 83 84)
if [[ $# -gt 0 ]]; then ISSUES=("$@"); else ISSUES=("${DEFAULT_ORDER[@]}"); fi

declare -A ENGINE=([82]=codex [81]=claude [85]=codex [83]=claude [84]=codex)

declare -A DEP_NOTE=(
    [82]="First in the batch — origin/${PR_BASE} has no batch fixes yet. Your fix also builds the 'derive the section list from the template's own '## N. Title' headings' mechanism that issue #84 reuses later — keep it reusable."
    [81]="Issue #82 (stale REQUIRED_SECTIONS) is already MERGED into origin/${PR_BASE} and is in your branch. You are the KEYSTONE: add the deterministic coherence check (new manifest task between compose-blueprint and quality-gate), add stable cross-reference anchors to templates/product_blueprint_template.md, and split each quality gate into one-condition-per-check. Ship a MINIMAL coherent+incoherent fixture pair to test your script now — the full golden-fixture regeneration is issue #83, done later."
    [85]="Issues #82 and #81 are already MERGED into origin/${PR_BASE}. #81 restructured prompts/05_quality_gate.md and added a deterministic coherence script. Extend Gate 2 (citation fidelity) and Gate 8 (agent-mode authorization boundary) and add the 'amend without a new report' playbook to references/troubleshooting.md. Reuse #81's deterministic-check script for the citation-string-exists check; do NOT redo #81's gate split."
    [83]="Issues #82, #81, #85 are already MERGED into origin/${PR_BASE}. The coherence rule, template anchors, and named checks now exist. Regenerate a COHERENT golden fixture (no MVP phase inversion) that carries the new anchors, add regression pairs tests/regressions/{precondition-currency,servicer-reachability}/{bad,fixed}.md each labelled with the check expected to catch it, a small mutation-derived negative set, and a weak-input OUTPUT fixture."
    [84]="Issues #82, #81, #85, #83 are already MERGED into origin/${PR_BASE}. Single-source the remaining duplication (frontmatter description <-> When To Trigger, and the manifest orthogonality seams). IMPORTANT: the artifact-contract cross-repo consolidation lives in ~/projects/design-pipeline (a SEPARATE repo). Do ONLY the research-pipeline-side work here; do NOT edit the other repo. Clearly flag the cross-repo remainder in your final message."
)

# --- preflight -------------------------------------------------------------
for bin in claude codex gh git; do
    command -v "$bin" &>/dev/null || { echo "ERROR: '$bin' not on PATH." >&2; exit 1; }
done
mkdir -p "$LOG_DIR"

engine_for() {
    local n="$1" ov="ENGINE_${1}"
    printf '%s' "${!ov:-${ENGINE[$n]:-claude}}"
}

# --- prompt builder --------------------------------------------------------
build_prompt() {
    local n="$1" pos="$2" total="$3" issue_text note
    issue_text="$(gh issue view "$n" --json number,title,labels,body \
        --jq '"#\(.number) \(.title)\nLabels: \([.labels[].name] | join(", "))\n\n\(.body)"')"
    note="${DEP_NOTE[$n]:-No sequence note recorded for this issue.}"
    cat <<PROMPT
Fix GitHub issue #$n in this repository (research-pipeline). This is item $pos of
$total in a dependency-ordered batch (order: ${ISSUES[*]}). All issues in the batch
concern the blueprint skill under src/research_pipeline/skill_data/blueprint/.

SEQUENCE CONTEXT (what is already merged into the trunk you branched from, and your role):
$note

Follow the repo conventions in AGENTS.md / CLAUDE.md (already loaded into your
context). In particular:
- This is a uv-managed project: prefix Python commands with 'uv run'.
- TDD: add a regression test that fails before your fix and passes after. Do not
  modify existing tests unless the issue explicitly requires it.
- Respect Hard Constraints HC1-HC6 (no plaintext secrets, stay inside the write
  path allowlist, never run destructive git/db commands autonomously, keep to the
  network egress allowlist).
- Plan internally then implement; the issue's "Proposed fix" is guidance, not
  gospel — use engineering judgment and prefer the simplest change that fully
  closes the reported defect.

DEFINITION OF DONE (all required before you finish):
1. Implement the fix for issue #$n.
2. uv run ruff format .
3. uv run ruff check . --fix
4. uv run mypy src/
5. uv run pytest tests/unit/test_skill_blueprint.py -x -q   (plus any other tests
   your change touches); add the regression test from TDD above.
6. uv run pre-commit run --all-files   (fix anything it flags).
7. Commit with a Conventional Commit message that references issue #$n
   (e.g. "fix: <summary> (#$n)"). Commit ONLY files you changed for this issue.
   Do NOT push. Do NOT open a pull request — the harness script pushes, opens the
   PR, waits for CI, and merges. Just leave a clean commit on the current branch.
8. If you cannot fully close the issue, commit the partial progress and end your
   final message with a clear "REMAINING:" section explaining what is left and why.

--- ISSUE #$n (verbatim) ---
$issue_text
--- END ISSUE #$n ---
PROMPT
}

# --- agent invocations -----------------------------------------------------
invoke_claude() {
    local prompt="$1" n="$2" perm="--dangerously-skip-permissions"
    [[ "$SKIP_PERMISSIONS" == "1" ]] || perm="--permission-mode acceptEdits"
    # shellcheck disable=SC2086
    claude -p "$prompt" --model "$CLAUDE_MODEL" --effort "$CLAUDE_EFFORT" \
        $perm --output-format text -n "fix-issue-$n"
}
invoke_codex() {
    local prompt="$1" perm="--dangerously-bypass-approvals-and-sandbox"
    [[ "$SKIP_PERMISSIONS" == "1" ]] || perm="-s workspace-write -a never"
    # shellcheck disable=SC2086
    codex exec --model "$CODEX_MODEL" $perm --color never "$prompt"
}

# --- wait for CI to complete on a PR ---------------------------------------
# Returns 0 iff every check succeeded. Handles the "checks not registered yet"
# race by waiting for at least one check row to appear before watching.
wait_for_ci() {
    local pr="$1" tries=0
    echo "   waiting for CI checks to register…"
    until gh pr checks "$pr" 2>/dev/null | grep -qiE 'pass|fail|pending|skipping'; do
        tries=$((tries + 1))
        if [[ $tries -ge 18 ]]; then
            echo "   ERROR: no CI checks appeared after ~3m." >&2
            return 3
        fi
        sleep 10
    done
    echo "   checks registered — watching until complete…"
    gh pr checks "$pr" --watch --fail-fast --interval "$CI_INTERVAL"
}

# --- one full cycle for one issue ------------------------------------------
run_issue() {
    local n="$1" pos="$2" total="$3"
    local engine branch ts log_file prompt subj body pr_url
    engine="$(engine_for "$n")"
    branch="${BRANCH_PREFIX}${n}"
    ts="$(date '+%Y%m%d-%H%M%S')"
    log_file="$LOG_DIR/issue-$(printf '%02d' "$n")-${engine}-${ts}.log"

    echo ""
    echo "════════════════════════════════════════════════════════"
    printf "  Issue #%s  (%d/%d)  engine=%s  branch=%s  %s\n" \
        "$n" "$pos" "$total" "$engine" "$branch" "$(date '+%H:%M:%S')"
    echo "════════════════════════════════════════════════════════"

    prompt="$(build_prompt "$n" "$pos" "$total")"

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "----- DRY RUN: engine=$engine, base=${REMOTE}/${PR_BASE}, would open PR '$branch' -----"
        printf '%s\n' "$prompt"
        echo "----- (no agent / push / PR / merge) -----"
        return 0
    fi

    # 1. fresh branch off the latest trunk
    echo "-> fetch ${REMOTE}/${PR_BASE} and branch $branch"
    git fetch "$REMOTE" "$PR_BASE" --quiet
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "   ERROR: worktree has uncommitted tracked changes — refusing to switch." >&2
        return 1
    fi
    git switch -C "$branch" "${REMOTE}/${PR_BASE}"

    # 2. agent fixes + commits (local only)
    echo "-> running $engine on issue #$n (log: $log_file)"
    case "$engine" in
        claude) invoke_claude "$prompt" "$n" 2>&1 | tee "$log_file" ;;
        codex)  invoke_codex  "$prompt"      2>&1 | tee "$log_file" ;;
        *) echo "   ERROR: unknown engine '$engine'." >&2; return 2 ;;
    esac
    local astatus=${PIPESTATUS[0]}
    [[ $astatus -eq 0 ]] || { echo "   ⚠ agent exited status $astatus"; return "$astatus"; }

    # verify the agent actually committed something
    if [[ "$(git rev-parse HEAD)" == "$(git rev-parse "${REMOTE}/${PR_BASE}")" ]]; then
        echo "   ERROR: agent produced no commit on $branch — nothing to PR." >&2
        return 1
    fi
    echo "   commit: $(git --no-pager log --oneline -1)"

    # 3. push
    echo "-> push $branch"
    git push -u "$REMOTE" "$branch" --force-with-lease

    # 4. PR (Closes #n so a squash-merge auto-closes the issue)
    subj="$(git --no-pager log -1 --format=%s)"
    body="$(printf 'Closes #%s\n\n%s' "$n" "$(git --no-pager log -1 --format=%b)")"
    echo "-> open PR (base=$PR_BASE, head=$branch)"
    if ! pr_url="$(gh pr create --base "$PR_BASE" --head "$branch" \
                    --title "$subj" --body "$body" 2>/dev/null)"; then
        pr_url="$(gh pr view "$branch" --json url --jq .url)"   # already exists (rerun)
        echo "   PR already exists: $pr_url"
    fi
    echo "   PR: $pr_url"

    # 5. wait for CI
    if ! wait_for_ci "$pr_url"; then
        echo "   ✗ CI failed / did not pass for #$n — NOT merging. Inspect: $pr_url" >&2
        return 1
    fi
    echo "   ✓ CI green"

    # 6. merge (unless NO_MERGE)
    if [[ "$NO_MERGE" == "1" ]]; then
        echo "   NO_MERGE=1 — leaving green PR unmerged: $pr_url"
        return 0
    fi
    echo "-> merge (--$MERGE_METHOD) + delete branch"
    # Detach HEAD first: this worktree is on the PR branch while the base branch
    # may be checked out in another worktree; letting gh switch the local
    # checkout to the base would fail ("branch already used by worktree").
    # Detached HEAD + explicit deletes keep the merge worktree-safe and
    # independent of gh's local-git behaviour.
    git switch --detach >/dev/null 2>&1 || true
    gh pr merge "$pr_url" "--$MERGE_METHOD"
    git push "$REMOTE" --delete "$branch" 2>/dev/null || true
    git branch -D "$branch" 2>/dev/null || true
    echo "   ✓ merged #$n into $PR_BASE"
    return 0
}

# --- main loop -------------------------------------------------------------
total=${#ISSUES[@]}; pos=0; failed=0
for n in "${ISSUES[@]}"; do
    pos=$((pos + 1))
    if ! run_issue "$n" "$pos" "$total"; then
        failed=$((failed + 1))
        if [[ "$STOP_ON_FAIL" == "1" && "$DRY_RUN" != "1" ]]; then
            echo ""
            echo "STOP_ON_FAIL=1 and issue #$n did not complete its cycle — aborting so"
            echo "later issues do not build on an unmerged dependency. Fix / rerun from #$n."
            break
        fi
    fi
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Batch finished.  Processed up to item $pos / $total.  Failures: $failed"
echo "  Merged PRs went into ${REMOTE}/${PR_BASE}.  Logs: $LOG_DIR/"
echo "  Pull them into your local checkout:  git fetch $REMOTE && git log --oneline ${REMOTE}/${PR_BASE}"
echo "════════════════════════════════════════════════════════"
exit "$failed"
