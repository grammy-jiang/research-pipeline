#!/usr/bin/env python3
"""check_blueprint_coherence.py — Deterministic cross-phase coherence guard.

The blueprint skill is a pure-LLM transformation. A single reasoning pass
cannot reliably hold an O(sections^2) definition->use graph over a
~1000-line document, so *cross-phase* incoherence passes the LLM
quality-gate silently. The most damaging class is a **phase inversion**: an
MVP-N node whose required servicer / precondition is itself staged MVP-N+1
(or later), or an MVP-claimed control gated on an unresolved open question.

This guard materializes that graph from stable, machine-readable coherence
anchors embedded in the blueprint and fails loudly. It runs between
``compose-blueprint`` and ``quality-gate`` in the blueprint manifest so the
incoherence is caught deterministically rather than re-reasoned by the LLM.

Coherence anchor format (one HTML comment per node, single line)::

    <!-- coherence: id=<anchor> stage=<stage> [requires=<id,id,...>]
                    [blocking=yes|no] [qualifier="<free text>"] -->

    id         Stable anchor, e.g. ``wf1.gate.contradicts`` or ``sec9.7.r1``.
    stage      ``MVP-0`` .. ``MVP-N`` | ``open`` | ``future``.
    requires   Comma-separated ids this node depends on / escalates to.
    blocking   For ``stage=open`` nodes: does the open item block the MVP?
    qualifier  Free text; its presence marks an MVP->open dependency as an
               explicit, intentional phase/condition qualifier.

Checks (deterministic, stdlib only, no network, no LLM):

    phase_inversion       For every MVP-staged node, every required servicer
                          must have ``stage(servicer) <= stage(node)``.
    open_dependency       An MVP-staged node may depend on an open question
                          only if that question is ``blocking=yes`` or the
                          edge carries an explicit ``qualifier``.
    dangling_reference    Every ``requires`` id must resolve to a declared node.
    duplicate_anchor      Each anchor id is declared at most once.
    contents_vs_headings  ``## Contents`` numbered entries must equal the
                          ``## N. Title`` headings, in order.
    placeholder_citation  No blank / ``TODO`` / ``TBD`` citations remain.
    citation_not_in_refs  If a source report is supplied, every paper-style
                          citation in the blueprint must exist verbatim in the
                          source report's ``## References`` section.
    confidence_upgrade    If a source report is supplied, a blueprint citation
                          must not carry a higher confidence grade than the
                          same citation carries in the source report.

Out of scope (tracked in issue #81 follow-ups): failure-mode<->risk-row
pairing, which needs a richer tagging convention.

Usage::

    python3 check_blueprint_coherence.py <topic-slug>-product-blueprint.md
    python3 check_blueprint_coherence.py <topic-slug>-product-blueprint.md \
        --source-report <topic-slug>-research-report.md

Exit codes:
    0 — No FAIL findings; blueprint is cross-phase coherent (warnings allowed).
    1 — One or more FAIL findings; do not pass the quality-gate.

Outputs:
    JSON result to stdout.
    Human-readable summary to stderr.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

FAIL = "FAIL"
WARNING = "WARNING"

# A coherence node is one HTML comment: ``<!-- coherence: <attrs> -->``.
_COHERENCE_RE = re.compile(r"<!--\s*coherence:\s*(?P<body>.*?)\s*-->", re.DOTALL)
# Attributes are ``key=value`` or ``key="quoted value"`` pairs.
_ATTR_RE = re.compile(r'(?P<key>[A-Za-z_]+)\s*=\s*(?:"(?P<qval>[^"]*)"|(?P<val>\S+))')
_HEADING_RE = re.compile(r"^##\s+(?P<num>\d+)\.\s+(?P<title>.+?)\s*$", re.MULTILINE)
_CONTENTS_ENTRY_RE = re.compile(
    r"^\s*-\s*\[(?P<num>\d+)\.\s*(?P<title>[^\]]+?)\]\(", re.MULTILINE
)
# Unambiguous placeholder citations (never a Markdown ``- [ ]`` task checkbox,
# which always carries a space or ``x`` and so never matches ``[]``).
_PLACEHOLDER_CITATION_RE = re.compile(
    r"\[\]|\[(?:todo|tbd|fixme|xxx|citation needed)\]", re.IGNORECASE
)
_FENCED_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_CITATION_RE = re.compile(r"\[(?P<citation>[^\[\]\n]{2,120})\](?!\()")
_REFERENCE_HEADING_RE = re.compile(
    r"^#{1,6}\s+References\s*$", re.IGNORECASE | re.MULTILINE
)
_NEXT_HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_ARXIV_CITATION_RE = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")
_YEAR_CITATION_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_SOURCE_REPORT_CITATION_PREFIX = "source report:"
_CONFIDENCE_RANKS = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "🔴": 1,
    "🟡": 2,
    "🟢": 3,
}
_CONFIDENCE_RE = re.compile(
    r"\b(?P<word>LOW|MEDIUM|HIGH)\b|(?P<symbol>[🔴🟡🟢])",
    re.IGNORECASE,
)

_OPEN_STAGES = {"open", "open-question", "unresolved"}
_FUTURE_STAGES = {"future", "later", "backlog"}


def _finding(level: str, check: str, message: str, **extra: object) -> dict:
    """Build one structured finding record."""
    record: dict = {"level": level, "check": check, "message": message}
    record.update(extra)
    return record


def _parse_stage(stage: str) -> tuple[str, int | None]:
    """Classify a stage label.

    Returns ``(kind, rank)`` where kind is ``mvp`` / ``open`` / ``future`` /
    ``unknown`` and rank is the integer MVP tier (only for ``mvp``).
    """
    text = stage.strip().lower()
    match = re.fullmatch(r"mvp-?(\d+)", text)
    if match:
        return ("mvp", int(match.group(1)))
    if text in _OPEN_STAGES:
        return ("open", None)
    if text in _FUTURE_STAGES:
        return ("future", None)
    return ("unknown", None)


def parse_nodes(text: str) -> list[dict]:
    """Extract every coherence anchor node from the blueprint text."""
    nodes: list[dict] = []
    for match in _COHERENCE_RE.finditer(text):
        attrs: dict[str, str] = {}
        for attr in _ATTR_RE.finditer(match.group("body")):
            value = attr.group("qval")
            if value is None:
                value = attr.group("val")
            attrs[attr.group("key").lower()] = value
        node_id = attrs.get("id")
        if not node_id:
            continue
        kind, rank = _parse_stage(attrs.get("stage", ""))
        requires = [
            r.strip() for r in attrs.get("requires", "").split(",") if r.strip()
        ]
        blocking_raw = attrs.get("blocking")
        blocking: bool | None = None
        if blocking_raw is not None:
            blocking = blocking_raw.strip().lower() in {"yes", "true", "1"}
        nodes.append(
            {
                "id": node_id,
                "stage": attrs.get("stage", ""),
                "kind": kind,
                "rank": rank,
                "requires": requires,
                "blocking": blocking,
                "qualifier": attrs.get("qualifier"),
                "line": text.count("\n", 0, match.start()) + 1,
            }
        )
    return nodes


def check_graph(nodes: list[dict]) -> list[dict]:
    """Verify the staged def->use graph has no phase inversion.

    For every MVP-staged node, each required servicer must be available no
    later than the node itself, and any required open question must be
    blocking (or the dependency explicitly qualified).
    """
    findings: list[dict] = []
    by_id: dict[str, dict] = {}
    for node in nodes:
        if node["id"] in by_id:
            findings.append(
                _finding(
                    FAIL,
                    "duplicate_anchor",
                    f"Coherence anchor id {node['id']!r} is declared more than once.",
                    id=node["id"],
                    line=node["line"],
                )
            )
            continue
        by_id[node["id"]] = node

    for node in nodes:
        # Only concrete MVP-staged nodes constrain their servicers' staging.
        if node["kind"] != "mvp":
            continue
        for target_id in node["requires"]:
            target = by_id.get(target_id)
            if target is None:
                findings.append(
                    _finding(
                        FAIL,
                        "dangling_reference",
                        f"{node['id']!r} (stage {node['stage']}) requires "
                        f"{target_id!r}, which is not a declared coherence anchor.",
                        id=node["id"],
                        line=node["line"],
                    )
                )
                continue
            if target["kind"] == "open":
                if not target["blocking"] and not node["qualifier"]:
                    findings.append(
                        _finding(
                            FAIL,
                            "open_dependency",
                            f"{node['id']!r} (stage {node['stage']}) depends on open "
                            f"question {target_id!r}, which is not blocking and the "
                            f"dependency carries no explicit phase qualifier.",
                            id=node["id"],
                            line=node["line"],
                        )
                    )
            elif target["kind"] == "future" or (
                target["kind"] == "mvp"
                and target["rank"] is not None
                and node["rank"] is not None
                and target["rank"] > node["rank"]
            ):
                findings.append(
                    _finding(
                        FAIL,
                        "phase_inversion",
                        f"{node['id']!r} (stage {node['stage']}) requires "
                        f"{target_id!r} (stage {target['stage']}), which is staged "
                        f"later — the servicer is unavailable when the node runs.",
                        id=node["id"],
                        line=node["line"],
                    )
                )

    if not nodes:
        findings.append(
            _finding(
                WARNING,
                "no_coherence_anchors",
                "No coherence anchors found; the phase-inversion graph could not "
                "be checked. Tag staged workflow gates, servicers, and open "
                "questions so cross-phase coherence can be verified.",
            )
        )
    return findings


def _contents_entries(text: str) -> list[tuple[int, str]] | None:
    """Return ``(number, title)`` pairs from the ``## Contents`` block."""
    start = text.find("## Contents")
    if start == -1:
        return None
    rest = text[start:]
    end = rest.find("\n---")
    block = rest if end == -1 else rest[:end]
    return [
        (int(m.group("num")), m.group("title").strip())
        for m in _CONTENTS_ENTRY_RE.finditer(block)
    ]


def check_contents_vs_headings(text: str) -> list[dict]:
    """Contents numbered entries must equal the ``## N. Title`` headings."""
    findings: list[dict] = []
    headings = {
        int(m.group("num")): m.group("title").strip()
        for m in _HEADING_RE.finditer(text)
    }
    contents = _contents_entries(text)
    if contents is None:
        return [
            _finding(
                FAIL,
                "contents_vs_headings",
                "No '## Contents' section found; the blueprint must list every "
                "numbered section.",
            )
        ]
    contents_map = dict(contents)
    for number in sorted(set(headings) | set(contents_map)):
        heading_title = headings.get(number)
        contents_title = contents_map.get(number)
        if heading_title != contents_title:
            findings.append(
                _finding(
                    FAIL,
                    "contents_vs_headings",
                    f"Section {number}: Contents lists {contents_title!r} but the "
                    f"heading is {heading_title!r}.",
                )
            )
    return findings


def check_placeholder_citations(text: str) -> list[dict]:
    """Flag blank / TODO / TBD placeholder citations."""
    findings: list[dict] = []
    for match in _PLACEHOLDER_CITATION_RE.finditer(text):
        findings.append(
            _finding(
                WARNING,
                "placeholder_citation",
                f"Placeholder citation {match.group(0)!r} left in the blueprint.",
                line=text.count("\n", 0, match.start()) + 1,
            )
        )
    return findings


def _strip_fenced_blocks(text: str) -> str:
    """Remove fenced code blocks so diagram labels are not parsed as citations."""
    return _FENCED_BLOCK_RE.sub("", text)


def _references_section(source_report_text: str) -> str | None:
    """Return the source report's references section, if present."""
    match = _REFERENCE_HEADING_RE.search(source_report_text)
    if match is None:
        return None
    rest = source_report_text[match.end() :]
    next_heading = _NEXT_HEADING_RE.search(rest)
    if next_heading is None:
        return rest
    return rest[: next_heading.start()]


def _is_paper_citation(citation: str) -> bool:
    """Return true for paper-style citations checked against References."""
    normalized = citation.strip()
    if normalized.lower().startswith(_SOURCE_REPORT_CITATION_PREFIX):
        return False
    return bool(
        _ARXIV_CITATION_RE.search(normalized) or _YEAR_CITATION_RE.search(normalized)
    )


def _paper_citations(text: str) -> set[str]:
    """Extract external paper-style citation strings from Markdown text."""
    citations: set[str] = set()
    for match in _CITATION_RE.finditer(_strip_fenced_blocks(text)):
        citation = match.group("citation").strip()
        if _is_paper_citation(citation):
            citations.add(citation)
    return citations


def _confidence_rank(text: str) -> int | None:
    """Return the strongest confidence grade named in a line of text."""
    ranks: list[int] = []
    for match in _CONFIDENCE_RE.finditer(text):
        token = match.group("word") or match.group("symbol")
        if token is None:
            continue
        ranks.append(_CONFIDENCE_RANKS[token.upper()])
    if not ranks:
        return None
    return max(ranks)


def _citation_confidence_map(text: str) -> dict[str, int]:
    """Map citations to the strongest confidence grade found on their lines."""
    confidence_by_citation: dict[str, int] = {}
    for line in _strip_fenced_blocks(text).splitlines():
        rank = _confidence_rank(line)
        if rank is None:
            continue
        for citation in _paper_citations(line):
            confidence_by_citation[citation] = max(
                confidence_by_citation.get(citation, 0), rank
            )
    return confidence_by_citation


def check_source_citation_fidelity(
    blueprint_text: str, source_report_text: str
) -> list[dict]:
    """Check blueprint paper citations against the source report."""
    findings: list[dict] = []
    references = _references_section(source_report_text)
    if references is None:
        return [
            _finding(
                FAIL,
                "source_references_missing",
                "Source report has no '## References' section; citation fidelity "
                "cannot be checked.",
            )
        ]

    source_references = _paper_citations(references)
    blueprint_citations = _paper_citations(blueprint_text)
    for citation in sorted(blueprint_citations - source_references):
        findings.append(
            _finding(
                FAIL,
                "citation_not_in_references",
                f"Blueprint citation [{citation}] does not exist verbatim in the "
                "source report's References section.",
                citation=citation,
            )
        )

    source_confidences = _citation_confidence_map(source_report_text)
    blueprint_confidences = _citation_confidence_map(blueprint_text)
    for citation in sorted(blueprint_citations & source_references):
        source_rank = source_confidences.get(citation)
        blueprint_rank = blueprint_confidences.get(citation)
        if source_rank is None or blueprint_rank is None:
            continue
        if blueprint_rank > source_rank:
            findings.append(
                _finding(
                    FAIL,
                    "confidence_silently_upgraded",
                    f"Blueprint assigns [{citation}] a higher confidence grade "
                    "than the source report.",
                    citation=citation,
                    source_rank=source_rank,
                    blueprint_rank=blueprint_rank,
                )
            )
    return findings


def run_checks(text: str, source_report_text: str | None = None) -> dict:
    """Run every coherence check and return a structured result dict."""
    nodes = parse_nodes(text)
    findings: list[dict] = []
    findings.extend(check_graph(nodes))
    findings.extend(check_contents_vs_headings(text))
    findings.extend(check_placeholder_citations(text))
    if source_report_text is not None:
        findings.extend(check_source_citation_fidelity(text, source_report_text))
    fail_count = sum(1 for f in findings if f["level"] == FAIL)
    warning_count = sum(1 for f in findings if f["level"] == WARNING)
    return {
        "findings": findings,
        "node_count": len(nodes),
        "fail_count": fail_count,
        "warning_count": warning_count,
        "all_passed": fail_count == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic cross-phase coherence guard for a composed product "
            "blueprint. Fails on phase inversion and related incoherence the "
            "LLM quality-gate cannot reliably catch."
        )
    )
    parser.add_argument(
        "blueprint",
        help="Path to the composed product blueprint markdown file.",
    )
    parser.add_argument(
        "--source-report",
        help=(
            "Optional source research report. When supplied, the guard verifies "
            "blueprint paper citations exist verbatim in its References section "
            "and confidence grades are not upgraded."
        ),
    )
    args = parser.parse_args()

    path = Path(args.blueprint).expanduser()
    if not path.exists():
        print(
            json.dumps(
                {"blueprint": str(path), "error": "not found", "all_passed": False},
                indent=2,
            )
        )
        print(f"❌ Blueprint not found: {path}", file=sys.stderr)
        return 1

    text = path.read_text(encoding="utf-8")
    source_report_text = None
    source_report_path = None
    if args.source_report is not None:
        source_report_path = Path(args.source_report).expanduser()
        if not source_report_path.exists():
            print(
                json.dumps(
                    {
                        "blueprint": str(path),
                        "source_report": str(source_report_path),
                        "error": "source report not found",
                        "all_passed": False,
                    },
                    indent=2,
                )
            )
            print(f"❌ Source report not found: {source_report_path}", file=sys.stderr)
            return 1
        source_report_text = source_report_path.read_text(encoding="utf-8")

    result = run_checks(text, source_report_text)
    result["blueprint"] = str(path)
    if source_report_path is not None:
        result["source_report"] = str(source_report_path)
    print(json.dumps(result, indent=2))

    if result["all_passed"]:
        print(
            f"✅ {path.name}: cross-phase coherent "
            f"({result['node_count']} anchor(s), "
            f"{result['warning_count']} warning(s)).",
            file=sys.stderr,
        )
        return 0

    print(
        f"❌ {path.name}: {result['fail_count']} coherence FAIL(s). "
        "Fix the staging/anchors before the quality-gate:",
        file=sys.stderr,
    )
    for finding in result["findings"]:
        if finding["level"] != FAIL:
            continue
        location = f" (line {finding['line']})" if finding.get("line") else ""
        print(
            f"   - [{finding['check']}]{location} {finding['message']}", file=sys.stderr
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
