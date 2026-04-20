"""Integration test: CMA audit + paging + associative linking + plan revision
working together on a synthetic research trajectory.

This verifies that the new modules interoperate with the existing
:class:`MemoryManager` and with each other. No network, no external processes.
"""

from __future__ import annotations

from research_pipeline.memory.associative import AssociativeLinker
from research_pipeline.memory.cma_audit import CMACompletenessAuditor
from research_pipeline.memory.manager import MemoryManager
from research_pipeline.memory.paging import PagedMemory
from research_pipeline.pipeline.plan_revision import PlanRevisionTracker


def test_end_to_end_memory_trajectory(tmp_path):
    manager = MemoryManager(
        working_capacity=3,
        episodic_path=tmp_path / "ep.sqlite",
        kg_path=tmp_path / "kg.json",
    )
    try:
        # 1. CMA audit must report full compliance on a healthy manager.
        report = CMACompletenessAuditor(manager).audit()
        assert report.passed is True

        # 2. Paging: faulting more items than capacity demotes to swap.
        paged = PagedMemory(manager)
        for i in range(5):
            paged.page_in(f"k{i}", f"value{i}", stage="search")
        assert paged.stats.page_out_count >= 2
        assert len(paged.swap) >= 2

        # 3. Associative linker builds a connectivity graph over items we
        #    just paged in (simulating cross-memory linking).
        linker = AssociativeLinker(top_k=2, min_weight=0.0)
        for i in range(5):
            linker.add(f"k{i}", f"topic alpha iteration {i}")
        # Every item after the first should have outbound links.
        assert all(linker.neighbors(f"k{i}") for i in range(1, 5)), (
            "associative linker failed to wire outbound edges"
        )

        # 4. Plan revision tracker detects plateau after repeated revisions.
        tracker = PlanRevisionTracker(
            plateau_tolerance=0.02, plateau_window=2, min_iterations=2
        )
        tracker.record("alpha beta", "alpha beta")
        tracker.record("alpha beta", "alpha beta")
        tracker.record("alpha beta", "alpha beta")
        assert tracker.should_stop() is True

        # 5. Swap can be drained for episodic consolidation.
        drained = paged.drain_swap()
        assert paged.swap == []
        assert len(drained) >= 2
    finally:
        manager.close()
