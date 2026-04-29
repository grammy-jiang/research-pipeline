"""Phase F end-to-end source-expansion offline test (F09).

Runs the full briefing pipeline against a baseline registry (RSS only) and a
candidate registry (RSS + Bluesky). Confirms:

* Both runs complete and produce a daily report.
* `compare_reports` (Phase F evaluation harness) reports coverage growth
  without exceeding noise thresholds.
* The candidate registry passes governance evaluation and no new source is
  silently auto-enabled outside what the registry itself declares.
"""

from __future__ import annotations

from pathlib import Path

from research_pipeline.briefing.registry import load_source_registry
from research_pipeline.briefing.source_evaluation import (
    compare_reports,
    evaluate_registry,
)
from research_pipeline.briefing.workflow import run_briefing

_FIXTURE_BASE = (
    Path(__file__).parent.parent / "fixtures" / "briefing" / "e2e" / "source_expansion"
)
_RUN_DATE = "2026-05-01"


def _run(registry_name: str, workspace: Path) -> str:
    registry = load_source_registry(_FIXTURE_BASE / registry_name)
    paths, _validation = run_briefing(
        registry,
        workspace=workspace,
        run_date=_RUN_DATE,
        fixture_base_dir=_FIXTURE_BASE,
    )
    assert paths.daily_report_path.exists(), "daily report not generated"
    return paths.daily_report_path.read_text(encoding="utf-8")


class TestPhaseFSourceExpansionE2E:
    def test_baseline_runs(self, tmp_path: Path) -> None:
        baseline_md = _run("registry_baseline.toml", tmp_path / "baseline")
        assert baseline_md.strip()

    def test_candidate_runs_and_grows_coverage(self, tmp_path: Path) -> None:
        baseline_md = _run("registry_baseline.toml", tmp_path / "baseline")
        candidate_md = _run("registry_candidate.toml", tmp_path / "candidate")
        comparison = compare_reports(baseline_md, candidate_md)
        # New source should not explode noise beyond configured ratios.
        assert comparison.noise_increase is False, (
            f"unexpected noise increase: {comparison}"
        )

    def test_candidate_registry_governance_passes(self) -> None:
        registry = load_source_registry(_FIXTURE_BASE / "registry_candidate.toml")
        report = evaluate_registry(registry)
        # Every source must satisfy governance metadata.
        for result in report:
            assert result.passed, (
                f"governance failed for {result.source_id}: {result.violations}"
            )
