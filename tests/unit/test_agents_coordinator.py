"""Tests for multi-agent coordinator module."""

from __future__ import annotations

import time

from research_pipeline.agents.coordinator import (
    AgentTask,
    AggregatedResult,
    Conflict,
    ConflictSeverity,
    MasterAgent,
    SubAgent,
    TaskStatus,
    _default_worker,
    _detect_conflicts,
    _merge_results,
    run_parallel_analysis,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestTaskStatus:
    def test_values(self) -> None:
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.RUNNING == "running"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.SKIPPED == "skipped"


class TestConflictSeverity:
    def test_values(self) -> None:
        assert ConflictSeverity.LOW == "low"
        assert ConflictSeverity.MEDIUM == "medium"
        assert ConflictSeverity.HIGH == "high"


# ---------------------------------------------------------------------------
# AgentTask tests
# ---------------------------------------------------------------------------
class TestAgentTask:
    def test_creation(self) -> None:
        task = AgentTask(task_id="t1", task_type="analyze")
        assert task.task_id == "t1"
        assert task.status == TaskStatus.PENDING
        assert task.result is None
        assert task.error is None
        assert task.duration_seconds == 0.0

    def test_with_paper_id(self) -> None:
        task = AgentTask(task_id="t1", task_type="analyze", paper_id="2301.00001")
        assert task.paper_id == "2301.00001"

    def test_with_input_data(self) -> None:
        task = AgentTask(
            task_id="t1",
            task_type="eval",
            input_data={"title": "Test Paper", "score": 0.8},
        )
        assert task.input_data["title"] == "Test Paper"


# ---------------------------------------------------------------------------
# Conflict tests
# ---------------------------------------------------------------------------
class TestConflict:
    def test_creation(self) -> None:
        c = Conflict(
            field="sentiment",
            values=["positive", "negative"],
            task_ids=["t1", "t2"],
        )
        assert c.field == "sentiment"
        assert c.severity == ConflictSeverity.LOW

    def test_with_severity(self) -> None:
        c = Conflict(
            field="score",
            values=[0.9, 0.1],
            task_ids=["t1", "t2"],
            severity=ConflictSeverity.HIGH,
        )
        assert c.severity == ConflictSeverity.HIGH


# ---------------------------------------------------------------------------
# AggregatedResult tests
# ---------------------------------------------------------------------------
class TestAggregatedResult:
    def test_empty(self) -> None:
        r = AggregatedResult()
        assert r.success_rate == 0.0
        assert r.has_conflicts is False

    def test_success_rate(self) -> None:
        r = AggregatedResult(total_tasks=10, completed_tasks=8, failed_tasks=2)
        assert abs(r.success_rate - 0.8) < 1e-9

    def test_has_conflicts(self) -> None:
        r = AggregatedResult(
            conflicts=[Conflict(field="f", values=[1, 2], task_ids=["a", "b"])]
        )
        assert r.has_conflicts is True

    def test_no_conflicts(self) -> None:
        r = AggregatedResult(total_tasks=5, completed_tasks=5)
        assert r.has_conflicts is False


# ---------------------------------------------------------------------------
# Default worker tests
# ---------------------------------------------------------------------------
class TestDefaultWorker:
    def test_completes_task(self) -> None:
        task = AgentTask(
            task_id="t1",
            task_type="test",
            input_data={"key": "value"},
        )
        result = _default_worker(task)
        assert result.status == TaskStatus.COMPLETED
        assert result.result is not None
        assert result.result["processed"] is True
        assert result.result["key"] == "value"


# ---------------------------------------------------------------------------
# Conflict detection tests
# ---------------------------------------------------------------------------
class TestDetectConflicts:
    def test_no_tasks(self) -> None:
        assert _detect_conflicts([]) == []

    def test_single_task(self) -> None:
        task = AgentTask(task_id="t1", task_type="a")
        task.status = TaskStatus.COMPLETED
        task.result = {"score": 0.8}
        assert _detect_conflicts([task]) == []

    def test_no_conflict(self) -> None:
        t1 = AgentTask(task_id="t1", task_type="a")
        t1.status = TaskStatus.COMPLETED
        t1.result = {"score": 0.8}
        t2 = AgentTask(task_id="t2", task_type="a")
        t2.status = TaskStatus.COMPLETED
        t2.result = {"score": 0.8}
        conflicts = _detect_conflicts([t1, t2], ["score"])
        assert conflicts == []

    def test_conflict_detected(self) -> None:
        t1 = AgentTask(task_id="t1", task_type="a")
        t1.status = TaskStatus.COMPLETED
        t1.result = {"sentiment": "positive"}
        t2 = AgentTask(task_id="t2", task_type="a")
        t2.status = TaskStatus.COMPLETED
        t2.result = {"sentiment": "negative"}
        conflicts = _detect_conflicts([t1, t2], ["sentiment"])
        assert len(conflicts) == 1
        assert conflicts[0].field == "sentiment"

    def test_auto_detect_fields(self) -> None:
        t1 = AgentTask(task_id="t1", task_type="a")
        t1.status = TaskStatus.COMPLETED
        t1.result = {"a": 1, "b": "x"}
        t2 = AgentTask(task_id="t2", task_type="a")
        t2.status = TaskStatus.COMPLETED
        t2.result = {"a": 2, "b": "x"}
        conflicts = _detect_conflicts([t1, t2])
        # Only 'a' should conflict, 'b' is the same
        assert len(conflicts) == 1
        assert conflicts[0].field == "a"

    def test_skips_failed_tasks(self) -> None:
        t1 = AgentTask(task_id="t1", task_type="a")
        t1.status = TaskStatus.COMPLETED
        t1.result = {"x": 1}
        t2 = AgentTask(task_id="t2", task_type="a")
        t2.status = TaskStatus.FAILED
        t2.result = {"x": 2}
        # Only one completed task, no conflict
        assert _detect_conflicts([t1, t2]) == []

    def test_high_severity(self) -> None:
        tasks = []
        for i in range(4):
            t = AgentTask(task_id=f"t{i}", task_type="a")
            t.status = TaskStatus.COMPLETED
            t.result = {"val": i}  # All different
            tasks.append(t)
        conflicts = _detect_conflicts(tasks, ["val"])
        assert len(conflicts) == 1
        assert conflicts[0].severity == ConflictSeverity.HIGH


# ---------------------------------------------------------------------------
# Merge results tests
# ---------------------------------------------------------------------------
class TestMergeResults:
    def test_empty(self) -> None:
        assert _merge_results([]) == []

    def test_collect_strategy(self) -> None:
        t1 = AgentTask(task_id="t1", task_type="a", paper_id="p1")
        t1.status = TaskStatus.COMPLETED
        t1.result = {"score": 0.8, "summary": "Good paper"}
        results = _merge_results([t1], "collect")
        assert len(results) == 1
        assert results[0]["task_id"] == "t1"
        assert results[0]["result"]["score"] == 0.8

    def test_evidence_only_strategy(self) -> None:
        t1 = AgentTask(task_id="t1", task_type="a", paper_id="p1")
        t1.status = TaskStatus.COMPLETED
        t1.result = {
            "claims": ["claim1"],
            "evidence": ["ev1"],
            "extra": "ignored",
        }
        results = _merge_results([t1], "evidence_only")
        assert len(results) == 1
        assert "claims" in results[0]["evidence"]
        assert "evidence" in results[0]["evidence"]
        assert "extra" not in results[0]["evidence"]

    def test_skips_failed(self) -> None:
        t1 = AgentTask(task_id="t1", task_type="a")
        t1.status = TaskStatus.FAILED
        t1.result = {"score": 0.8}
        assert _merge_results([t1]) == []


# ---------------------------------------------------------------------------
# SubAgent tests
# ---------------------------------------------------------------------------
class TestSubAgent:
    def test_default_processing(self) -> None:
        agent = SubAgent("agent-0")
        task = AgentTask(task_id="t1", task_type="test")
        result = agent.process(task)
        assert result.status == TaskStatus.COMPLETED
        assert agent.tasks_processed == 1

    def test_custom_worker(self) -> None:
        def custom(task: AgentTask) -> AgentTask:
            task.status = TaskStatus.COMPLETED
            task.result = {"custom": True}
            return task

        agent = SubAgent("agent-0", custom)
        task = AgentTask(task_id="t1", task_type="test")
        result = agent.process(task)
        assert result.result == {"custom": True}

    def test_failed_worker(self) -> None:
        def failing(task: AgentTask) -> AgentTask:
            raise ValueError("Something went wrong")

        agent = SubAgent("agent-0", failing)
        task = AgentTask(task_id="t1", task_type="test")
        result = agent.process(task)
        assert result.status == TaskStatus.FAILED
        assert "Something went wrong" in (result.error or "")

    def test_duration_tracked(self) -> None:
        def slow(task: AgentTask) -> AgentTask:
            time.sleep(0.01)
            task.status = TaskStatus.COMPLETED
            task.result = {"done": True}
            return task

        agent = SubAgent("agent-0", slow)
        task = AgentTask(task_id="t1", task_type="test")
        result = agent.process(task)
        assert result.duration_seconds > 0


# ---------------------------------------------------------------------------
# MasterAgent tests
# ---------------------------------------------------------------------------
class TestMasterAgent:
    def test_create_tasks(self) -> None:
        master = MasterAgent()
        tasks = master.create_tasks(
            [{"paper_id": "p1"}, {"paper_id": "p2"}],
            task_type="analyze",
        )
        assert len(tasks) == 2
        assert tasks[0].task_id == "analyze-p1"
        assert tasks[1].paper_id == "p2"

    def test_create_tasks_with_id(self) -> None:
        master = MasterAgent()
        tasks = master.create_tasks([{"id": "abc"}], task_type="eval")
        assert tasks[0].task_id == "eval-abc"

    def test_create_tasks_auto_id(self) -> None:
        master = MasterAgent()
        tasks = master.create_tasks([{"data": "something"}], task_type="test")
        assert tasks[0].task_id == "test-item-0"

    def test_run_empty(self) -> None:
        master = MasterAgent()
        result = master.run([])
        assert result.total_tasks == 0
        assert result.success_rate == 0.0

    def test_run_single(self) -> None:
        master = MasterAgent(max_workers=1)
        tasks = master.create_tasks([{"paper_id": "p1", "title": "Test"}])
        result = master.run(tasks)
        assert result.total_tasks == 1
        assert result.completed_tasks == 1
        assert result.success_rate == 1.0

    def test_run_parallel(self) -> None:
        items = [{"paper_id": f"p{i}"} for i in range(5)]
        master = MasterAgent(max_workers=3)
        tasks = master.create_tasks(items)
        result = master.run(tasks)
        assert result.total_tasks == 5
        assert result.completed_tasks == 5

    def test_run_with_failures(self) -> None:
        call_count = 0

        def sometimes_fail(task: AgentTask) -> AgentTask:
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise ValueError("Even task fails")
            task.status = TaskStatus.COMPLETED
            task.result = {"ok": True}
            return task

        master = MasterAgent(max_workers=1, worker=sometimes_fail)
        tasks = master.create_tasks([{"paper_id": f"p{i}"} for i in range(4)])
        result = master.run(tasks)
        assert result.total_tasks == 4
        assert result.completed_tasks == 2
        assert result.failed_tasks == 2

    def test_conflict_detection(self) -> None:
        def scorer(task: AgentTask) -> AgentTask:
            # Different papers get different scores
            paper_id = task.paper_id or ""
            task.status = TaskStatus.COMPLETED
            task.result = {"quality": "high" if "1" in paper_id else "low"}
            return task

        master = MasterAgent(
            max_workers=1,
            worker=scorer,
            conflict_fields=["quality"],
        )
        tasks = master.create_tasks([{"paper_id": "p1"}, {"paper_id": "p2"}])
        result = master.run(tasks)
        assert result.has_conflicts is True
        assert result.conflicts[0].field == "quality"

    def test_evidence_only_merge(self) -> None:
        def evidence_worker(task: AgentTask) -> AgentTask:
            task.status = TaskStatus.COMPLETED
            task.result = {
                "claims": ["claim1"],
                "findings": ["finding1"],
                "metadata": "ignored",
            }
            return task

        master = MasterAgent(
            max_workers=1,
            worker=evidence_worker,
            merge_strategy="evidence_only",
        )
        tasks = master.create_tasks([{"paper_id": "p1"}])
        result = master.run(tasks)
        assert len(result.results) == 1
        assert "claims" in result.results[0]["evidence"]
        assert "metadata" not in result.results[0]["evidence"]

    def test_uses_stored_tasks(self) -> None:
        master = MasterAgent(max_workers=1)
        master.create_tasks([{"paper_id": "p1"}])
        result = master.run()
        assert result.total_tasks == 1


# ---------------------------------------------------------------------------
# run_parallel_analysis convenience function tests
# ---------------------------------------------------------------------------
class TestRunParallelAnalysis:
    def test_basic(self) -> None:
        items = [{"paper_id": f"p{i}"} for i in range(3)]
        result = run_parallel_analysis(items, max_workers=2)
        assert result.total_tasks == 3
        assert result.completed_tasks == 3

    def test_custom_worker(self) -> None:
        def custom(task: AgentTask) -> AgentTask:
            task.status = TaskStatus.COMPLETED
            task.result = {"analyzed": task.paper_id}
            return task

        result = run_parallel_analysis(
            [{"paper_id": "p1"}],
            worker=custom,
            max_workers=1,
        )
        assert result.completed_tasks == 1
        assert result.results[0]["result"]["analyzed"] == "p1"

    def test_empty_items(self) -> None:
        result = run_parallel_analysis([])
        assert result.total_tasks == 0

    def test_task_type(self) -> None:
        result = run_parallel_analysis(
            [{"paper_id": "p1"}],
            task_type="evaluate",
        )
        assert result.total_tasks == 1
