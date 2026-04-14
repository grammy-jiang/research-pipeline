"""Export synthesis reports in structured formats (JSON, BibTeX)."""

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from research_pipeline import __version__
from research_pipeline.models.summary import PaperSummary, SynthesisReport

logger = logging.getLogger(__name__)

_ARXIV_ID_PATTERN = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}(v\d+)?$")
_BIBTEX_SPECIAL = re.compile(r"([&%$#_{}~^\\])")


def _sanitize_bibtex(value: str) -> str:
    """Escape special BibTeX characters in a string.

    Args:
        value: Raw string to sanitize.

    Returns:
        String with BibTeX special characters escaped.
    """
    return _BIBTEX_SPECIAL.sub(r"\\\1", value)


def _extract_year_from_arxiv_id(arxiv_id: str) -> str:
    """Extract the publication year from an arXiv ID.

    arXiv IDs follow the pattern YYMM.NNNNN, where YY is the two-digit year.

    Args:
        arxiv_id: The arXiv identifier string.

    Returns:
        Four-digit year string, or ``"unknown"`` if the ID does not match.
    """
    match = _ARXIV_ID_PATTERN.match(arxiv_id)
    if not match:
        return "unknown"
    yy = int(match.group(1))
    century = 20 if yy < 90 else 19
    return str(century * 100 + yy)


def _build_evidence_map(report: SynthesisReport) -> list[dict[str, object]]:
    """Build explicit paper_id → claim mappings from all evidence pointers.

    Args:
        report: The synthesis report to extract evidence from.

    Returns:
        List of dicts with ``paper_id``, ``claim``, and ``evidence`` keys.
    """
    mappings: list[dict[str, object]] = []
    for agreement in report.agreements:
        for paper_id in agreement.supporting_papers:
            mappings.append(
                {
                    "paper_id": paper_id,
                    "claim": agreement.claim,
                    "type": "agreement",
                    "evidence": [e.model_dump() for e in agreement.evidence],
                }
            )
    for disagreement in report.disagreements:
        for paper_id, position in disagreement.positions.items():
            mappings.append(
                {
                    "paper_id": paper_id,
                    "claim": f"{disagreement.topic}: {position}",
                    "type": "disagreement",
                    "evidence": [e.model_dump() for e in disagreement.evidence],
                }
            )
    return mappings


def export_json(report: SynthesisReport, output_path: Path) -> None:
    """Export the full synthesis report as pretty-printed JSON.

    Includes all evidence pointers with explicit paper_id → claim mappings
    and metadata (export timestamp, pipeline version).

    Args:
        report: The synthesis report to export.
        output_path: Destination file path for the JSON output.
    """
    payload: dict[str, object] = {
        "metadata": {
            "export_timestamp": datetime.now(tz=UTC).isoformat(),
            "pipeline_version": __version__,
        },
        "report": report.model_dump(),
        "evidence_map": _build_evidence_map(report),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Exported JSON synthesis report to %s", output_path)


def _paper_to_bibtex(summary: PaperSummary) -> str:
    """Convert a single paper summary to a BibTeX entry string.

    Uses ``@article`` for arXiv papers and ``@misc`` for non-arXiv IDs.

    Args:
        summary: The paper summary to convert.

    Returns:
        A formatted BibTeX entry string.
    """
    year = _extract_year_from_arxiv_id(summary.arxiv_id)
    is_arxiv = _ARXIV_ID_PATTERN.match(summary.arxiv_id) is not None
    entry_type = "article" if is_arxiv else "misc"
    citation_key = summary.arxiv_id.replace("/", "_")

    title = _sanitize_bibtex(summary.title)

    lines = [
        f"@{entry_type}{{{citation_key},",
        f"  title = {{{title}}},",
        f"  year = {{{year}}},",
    ]

    if is_arxiv:
        lines.append(f"  eprint = {{{summary.arxiv_id}}},")
        lines.append("  archivePrefix = {arXiv},")

    lines.append("}")
    return "\n".join(lines)


def export_bibtex(report: SynthesisReport, output_path: Path) -> None:
    """Generate BibTeX entries for every paper in the synthesis report.

    Uses the arXiv ID as the citation key. Non-arXiv papers are emitted as
    ``@misc`` entries.

    Args:
        report: The synthesis report to export.
        output_path: Destination file path for the BibTeX output.
    """
    entries = [_paper_to_bibtex(s) for s in report.paper_summaries]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    logger.info("Exported BibTeX entries to %s", output_path)


def export_report(
    report: SynthesisReport,
    output_path: Path,
    fmt: str = "json",
) -> None:
    """Dispatch export to the appropriate format handler.

    Args:
        report: The synthesis report to export.
        output_path: Destination file path for the export.
        fmt: Export format — ``"json"`` or ``"bibtex"``.

    Raises:
        ValueError: If *fmt* is not a supported format.
    """
    exporters = {
        "json": export_json,
        "bibtex": export_bibtex,
    }
    exporter = exporters.get(fmt)
    if exporter is None:
        raise ValueError(
            f"Unknown export format {fmt!r}. Supported: {sorted(exporters)}"
        )
    exporter(report, output_path)
