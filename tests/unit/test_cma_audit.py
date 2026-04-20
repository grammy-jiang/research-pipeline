"""Unit tests for :mod:`research_pipeline.memory.cma_audit`."""

from __future__ import annotations

import pytest

from research_pipeline.memory.cma_audit import (
    CMAAuditReport,
    CMACompletenessAuditor,
    CMAProperty,
)
from research_pipeline.memory.manager import MemoryManager


def test_default_memory_manager_passes_all_six_cma_properties(tmp_path):
    """A fully-wired MemoryManager should pass all six CMA properties."""
    manager = MemoryManager(
        working_capacity=32,
        episodic_path=tmp_path / "ep.sqlite",
        kg_path=tmp_path / "kg.json",
    )
    try:
        report = CMACompletenessAuditor(manager).audit()
    finally:
        manager.close()

    assert isinstance(report, CMAAuditReport)
    assert report.total == 6
    assert report.passed_count == 6
    assert report.failed_count == 0
    assert report.passed is True
    assert report.is_rag_only is False
    for prop in CMAProperty:
        result = report.get(prop)
        assert result is not None
        assert result.passed is True, f"Property {prop.value} failed: {result.evidence}"


def test_unbounded_working_memory_fails_selective_retention(tmp_path, monkeypatch):
    """Selective retention must fail when working capacity is pathological."""
    manager = MemoryManager(
        working_capacity=32,
        episodic_path=tmp_path / "ep.sqlite",
        kg_path=tmp_path / "kg.json",
    )
    try:
        # Simulate an unbounded working memory (fails the 1..10_000 guard).
        monkeypatch.setattr(type(manager.working), "capacity", 99_999, raising=False)
        report = CMACompletenessAuditor(manager).audit()
    finally:
        manager.close()

    sel = report.get(CMAProperty.SELECTIVE_RETENTION)
    assert sel is not None
    assert sel.passed is False
    assert report.is_rag_only is True
    assert report.passed is False
    assert "Missing" in report.summary


def test_get_returns_none_for_missing_property(tmp_path):
    """Report.get returns None when property not in results."""
    manager = MemoryManager(
        working_capacity=8,
        episodic_path=tmp_path / "ep.sqlite",
        kg_path=tmp_path / "kg.json",
    )
    try:
        report = CMACompletenessAuditor(manager).audit()
    finally:
        manager.close()
    empty = CMAAuditReport()
    assert empty.get(CMAProperty.PERSISTENCE) is None
    assert report.get(CMAProperty.PERSISTENCE) is not None


@pytest.mark.parametrize("prop", list(CMAProperty))
def test_each_property_has_module_attribution(tmp_path, prop):
    """Every property result names the module(s) responsible."""
    manager = MemoryManager(
        working_capacity=8,
        episodic_path=tmp_path / "ep.sqlite",
        kg_path=tmp_path / "kg.json",
    )
    try:
        report = CMACompletenessAuditor(manager).audit()
    finally:
        manager.close()
    result = report.get(prop)
    assert result is not None
    assert result.module != ""
