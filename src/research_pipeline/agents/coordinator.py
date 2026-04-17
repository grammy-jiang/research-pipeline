"""Multi-agent architecture: MasterAgent + SubAgent for parallel analysis.

Implements a lightweight agent coordination pattern:

- **MasterAgent**: orchestrates work distribution, aggregates results,
  resolves conflicts between sub-agent outputs
- **SubAgent**: processes individual papers or analysis tasks independently
- **AgentTask**: unit of work assigned to a sub-agent
- **AggregatedResult**: merged output from all sub-agents with conflict resolution

The architecture supports:
1. Parallel paper analysis (each sub-agent handles one paper)
2. Evidence-only merging (sub-agents produce evidence, master synthesizes)
3. Pre-commitment aggregation (results collected before final synthesis)
4. Conflict detection across sub-agent outputs
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class TaskStatus(StrEnum):
    """Status of an agent task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ConflictSeverity(StrEnum):
    """How severe a conflict between sub-agent outputs is."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class AgentTask:
    """Unit of work assigned to a sub-agent."""

    task_id: str
    task_type: str  # "analyze", "summarize", "evaluate", etc.
    paper_id: str | None = None
    input_data: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class Conflict:
    """Conflict detected between sub-agent outputs."""

    field: str
    values: list[Any]
    task_ids: list[str]
    severity: ConflictSeverity = ConflictSeverity.LOW
    resolution: str | None = None


@dataclass
class AggregatedResult:
    """Merged output from all sub-agents."""

    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    aggregation_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Fraction of tasks that completed successfully."""
        if self.total_tasks == 0:
            return 0.0
        return self.completed_tasks / self.total_tasks

    @property
    def has_conflicts(self) -> bool:
        """Whether any conflicts were detected."""
        return len(self.conflicts) > 0


# Type for a sub-agent worker function
SubAgentWorker = Callable[[AgentTask], AgentTask]


def _default_worker(task: AgentTask) -> AgentTask:
    """Default no-op worker that returns the task unchanged."""
    task.status = TaskStatus.COMPLETED
    task.result = {"processed": True, **task.input_data}
    return task


def _detect_conflicts(
    tasks: list[AgentTask],
    conflict_fields: list[str] | None = None,
) -> list[Conflict]:
    """Detect conflicts across completed sub-agent results.

    Compares specified fields across all results. If the same field
    has different values in different results, it's flagged as a conflict.

    Args:
        tasks: Completed tasks with results.
        conflict_fields: Fields to check for conflicts. If None, checks all.

    Returns:
        List of detected conflicts.
    """
    completed = [t for t in tasks if t.status == TaskStatus.COMPLETED and t.result]
    if len(completed) < 2:
        return []

    if conflict_fields is None:
        all_keys: set[str] = set()
        for t in completed:
            if t.result:
                all_keys.update(t.result.keys())
        conflict_fields = sorted(all_keys)

    conflicts: list[Conflict] = []
    for fld in conflict_fields:
        values: list[Any] = []
        task_ids: list[str] = []
        for t in completed:
            if t.result and fld in t.result:
                values.append(t.result[fld])
                task_ids.append(t.task_id)

        if len(values) < 2:
            continue

        # Check if all values are the same
        unique_vals: list[Any] = []
        for v in values:
            if v not in unique_vals:
                unique_vals.append(v)

        if len(unique_vals) > 1:
            severity = ConflictSeverity.LOW
            if len(unique_vals) > len(values) // 2:
                severity = ConflictSeverity.HIGH
            elif len(unique_vals) > 2:
                severity = ConflictSeverity.MEDIUM

            conflicts.append(
                Conflict(
                    field=fld,
                    values=unique_vals,
                    task_ids=task_ids,
                    severity=severity,
                )
            )

    return conflicts


def _merge_results(
    tasks: list[AgentTask],
    merge_strategy: str = "collect",
) -> list[dict[str, Any]]:
    """Merge results from completed sub-agents.

    Args:
        tasks: All tasks (completed and failed).
        merge_strategy: How to merge — "collect" (list all) or
            "evidence_only" (only include evidence fields).

    Returns:
        List of merged result dicts.
    """
    results: list[dict[str, Any]] = []
    for t in tasks:
        if t.status != TaskStatus.COMPLETED or t.result is None:
            continue
        entry = {
            "task_id": t.task_id,
            "paper_id": t.paper_id,
            "duration": t.duration_seconds,
        }
        if merge_strategy == "evidence_only":
            evidence_keys = {"claims", "evidence", "findings", "quotes", "citations"}
            entry["evidence"] = {
                k: v for k, v in t.result.items() if k in evidence_keys
            }
        else:
            entry["result"] = t.result
        results.append(entry)
    return results


class SubAgent:
    """Worker agent that processes individual tasks."""

    def __init__(
        self,
        agent_id: str,
        worker: SubAgentWorker | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.worker = worker or _default_worker
        self.tasks_processed = 0

    def process(self, task: AgentTask) -> AgentTask:
        """Process a single task.

        Args:
            task: The task to process.

        Returns:
            The task with updated status and result.
        """
        task.status = TaskStatus.RUNNING
        start = time.monotonic()
        try:
            result = self.worker(task)
            result.duration_seconds = time.monotonic() - start
            self.tasks_processed += 1
            logger.debug(
                "SubAgent %s completed task %s in %.2fs",
                self.agent_id,
                task.task_id,
                result.duration_seconds,
            )
            return result
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.error = str(exc)
            task.duration_seconds = time.monotonic() - start
            logger.warning(
                "SubAgent %s failed task %s: %s",
                self.agent_id,
                task.task_id,
                exc,
            )
            return task


class MasterAgent:
    """Orchestrator that distributes work across sub-agents.

    Manages task distribution, parallel execution, result aggregation,
    and conflict detection across sub-agent outputs.
    """

    def __init__(
        self,
        max_workers: int = 4,
        worker: SubAgentWorker | None = None,
        conflict_fields: list[str] | None = None,
        merge_strategy: str = "collect",
    ) -> None:
        """Initialize the master agent.

        Args:
            max_workers: Maximum parallel sub-agents.
            worker: Worker function for sub-agents. Defaults to no-op.
            conflict_fields: Fields to check for conflicts.
            merge_strategy: "collect" or "evidence_only".
        """
        self.max_workers = max_workers
        self.worker = worker
        self.conflict_fields = conflict_fields
        self.merge_strategy = merge_strategy
        self._sub_agents: list[SubAgent] = []
        self._tasks: list[AgentTask] = []

    def create_tasks(
        self,
        items: list[dict[str, Any]],
        task_type: str = "analyze",
    ) -> list[AgentTask]:
        """Create tasks from input items.

        Args:
            items: List of dicts with at least 'paper_id' or 'id'.
            task_type: Type of task to create.

        Returns:
            List of created AgentTask objects.
        """
        tasks: list[AgentTask] = []
        for i, item in enumerate(items):
            paper_id = item.get("paper_id") or item.get("id") or f"item-{i}"
            task = AgentTask(
                task_id=f"{task_type}-{paper_id}",
                task_type=task_type,
                paper_id=str(paper_id),
                input_data=item,
            )
            tasks.append(task)
        self._tasks = tasks
        return tasks

    def run(
        self,
        tasks: list[AgentTask] | None = None,
    ) -> AggregatedResult:
        """Execute all tasks using sub-agents.

        Distributes tasks across sub-agents in parallel using a thread pool.

        Args:
            tasks: Tasks to execute. Uses self._tasks if None.

        Returns:
            AggregatedResult with merged outputs and conflicts.
        """
        if tasks is None:
            tasks = self._tasks

        if not tasks:
            return AggregatedResult()

        # Create sub-agents
        num_agents = min(self.max_workers, len(tasks))
        self._sub_agents = [
            SubAgent(f"sub-{i}", self.worker) for i in range(num_agents)
        ]

        logger.info(
            "MasterAgent: dispatching %d tasks to %d sub-agents",
            len(tasks),
            num_agents,
        )

        completed_tasks: list[AgentTask] = []

        if num_agents == 1:
            # Single-threaded execution
            agent = self._sub_agents[0]
            for task in tasks:
                result = agent.process(task)
                completed_tasks.append(result)
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=num_agents) as executor:
                agent_cycle = iter(
                    self._sub_agents[i % num_agents] for i in range(len(tasks))
                )
                futures = {
                    executor.submit(agent.process, task): task
                    for task, agent in zip(tasks, agent_cycle, strict=False)
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        completed_tasks.append(result)
                    except Exception as exc:
                        original_task = futures[future]
                        original_task.status = TaskStatus.FAILED
                        original_task.error = str(exc)
                        completed_tasks.append(original_task)

        # Aggregate
        done = sum(1 for t in completed_tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in completed_tasks if t.status == TaskStatus.FAILED)

        conflicts = _detect_conflicts(completed_tasks, self.conflict_fields)
        merged = _merge_results(completed_tasks, self.merge_strategy)

        result = AggregatedResult(
            total_tasks=len(tasks),
            completed_tasks=done,
            failed_tasks=failed,
            results=merged,
            conflicts=conflicts,
            aggregation_metadata={
                "num_agents": num_agents,
                "merge_strategy": self.merge_strategy,
                "total_duration": sum(t.duration_seconds for t in completed_tasks),
            },
        )

        logger.info(
            "MasterAgent: %d/%d completed, %d failed, %d conflicts",
            done,
            len(tasks),
            failed,
            len(conflicts),
        )

        return result


def run_parallel_analysis(
    items: list[dict[str, Any]],
    worker: SubAgentWorker | None = None,
    max_workers: int = 4,
    task_type: str = "analyze",
    conflict_fields: list[str] | None = None,
    merge_strategy: str = "collect",
) -> AggregatedResult:
    """Convenience function to run parallel analysis on a list of items.

    Args:
        items: Items to analyze (each should have 'paper_id' or 'id').
        worker: Processing function for each task.
        max_workers: Maximum parallel workers.
        task_type: Type label for tasks.
        conflict_fields: Fields to check for cross-result conflicts.
        merge_strategy: "collect" or "evidence_only".

    Returns:
        AggregatedResult with merged outputs.
    """
    master = MasterAgent(
        max_workers=max_workers,
        worker=worker,
        conflict_fields=conflict_fields,
        merge_strategy=merge_strategy,
    )
    tasks = master.create_tasks(items, task_type)
    return master.run(tasks)
