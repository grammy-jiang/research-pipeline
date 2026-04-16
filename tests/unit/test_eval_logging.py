"""Tests for three-channel evaluation logging."""

from pathlib import Path

from research_pipeline.infra.eval_logging import (
    AuditDB,
    EvalLogger,
    ExecutionTracer,
    SnapshotManager,
    TraceLevel,
)

# ── ExecutionTracer ──────────────────────────────────────────────


class TestExecutionTracer:
    """Tests for Channel 1: Execution traces."""

    def test_emit_creates_jsonl(self, tmp_path: Path) -> None:
        tracer = ExecutionTracer(tmp_path)
        result = tracer.emit("test_event", stage="plan")
        assert result["event"] == "test_event"
        assert result["stage"] == "plan"
        assert tracer.log_path.exists()
        lines = tracer.log_path.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_emit_increments_count(self, tmp_path: Path) -> None:
        tracer = ExecutionTracer(tmp_path)
        assert tracer.trace_count == 0
        tracer.emit("e1")
        tracer.emit("e2")
        assert tracer.trace_count == 2

    def test_emit_includes_trace_id(self, tmp_path: Path) -> None:
        tracer = ExecutionTracer(tmp_path)
        result = tracer.emit("event")
        assert "trace_id" in result
        assert len(result["trace_id"]) == 12

    def test_emit_includes_timestamp(self, tmp_path: Path) -> None:
        tracer = ExecutionTracer(tmp_path)
        result = tracer.emit("event")
        assert "timestamp" in result

    def test_emit_with_parent_id(self, tmp_path: Path) -> None:
        tracer = ExecutionTracer(tmp_path)
        result = tracer.emit("child", parent_id="parent123")
        assert result["parent_id"] == "parent123"

    def test_emit_with_data(self, tmp_path: Path) -> None:
        tracer = ExecutionTracer(tmp_path)
        result = tracer.emit("event", data={"key": "value"})
        assert result["data"] == {"key": "value"}

    def test_emit_with_level(self, tmp_path: Path) -> None:
        tracer = ExecutionTracer(tmp_path)
        r1 = tracer.emit("warn", level=TraceLevel.WARNING)
        assert r1["level"] == "warning"
        r2 = tracer.emit("custom", level="custom_level")
        assert r2["level"] == "custom_level"

    def test_emit_disabled_is_noop(self, tmp_path: Path) -> None:
        tracer = ExecutionTracer(tmp_path, enabled=False)
        result = tracer.emit("event")
        assert result["event"] == "event"
        assert tracer.trace_count == 0
        assert not tracer.log_path.exists()

    def test_read_traces_empty(self, tmp_path: Path) -> None:
        tracer = ExecutionTracer(tmp_path)
        assert tracer.read_traces() == []

    def test_read_traces_returns_all(self, tmp_path: Path) -> None:
        tracer = ExecutionTracer(tmp_path)
        tracer.emit("e1", stage="plan")
        tracer.emit("e2", stage="search")
        tracer.emit("e3", stage="plan")
        traces = tracer.read_traces()
        assert len(traces) == 3

    def test_read_traces_filter_stage(self, tmp_path: Path) -> None:
        tracer = ExecutionTracer(tmp_path)
        tracer.emit("e1", stage="plan")
        tracer.emit("e2", stage="search")
        tracer.emit("e3", stage="plan")
        traces = tracer.read_traces(stage="plan")
        assert len(traces) == 2

    def test_read_traces_filter_level(self, tmp_path: Path) -> None:
        tracer = ExecutionTracer(tmp_path)
        tracer.emit("e1", level=TraceLevel.INFO)
        tracer.emit("e2", level=TraceLevel.ERROR)
        traces = tracer.read_traces(level="error")
        assert len(traces) == 1

    def test_optional_fields_omitted(self, tmp_path: Path) -> None:
        tracer = ExecutionTracer(tmp_path)
        result = tracer.emit("bare_event")
        assert "stage" not in result
        assert "parent_id" not in result
        assert "data" not in result

    def test_log_path_property(self, tmp_path: Path) -> None:
        tracer = ExecutionTracer(tmp_path)
        assert tracer.log_path == tmp_path / "logs" / "traces.jsonl"

    def test_enabled_property(self, tmp_path: Path) -> None:
        assert ExecutionTracer(tmp_path, enabled=True).enabled is True
        assert ExecutionTracer(tmp_path, enabled=False).enabled is False


# ── AuditDB ──────────────────────────────────────────────────────


class TestAuditDB:
    """Tests for Channel 2: SQLite audit database."""

    def test_record_creates_db(self, tmp_path: Path) -> None:
        db = AuditDB(tmp_path)
        row_id = db.record("plan", "started")
        assert row_id > 0
        assert db.db_path.exists()
        db.close()

    def test_record_stores_all_fields(self, tmp_path: Path) -> None:
        db = AuditDB(tmp_path)
        db.record(
            "screen",
            "filter",
            run_id="run123",
            input_hash="abc",
            output_hash="def",
            model="gpt-4",
            tokens_used=100,
            duration_ms=500,
            details={"key": "value"},
        )
        records = db.query(stage="screen")
        assert len(records) == 1
        r = records[0]
        assert r["stage"] == "screen"
        assert r["action"] == "filter"
        assert r["run_id"] == "run123"
        assert r["input_hash"] == "abc"
        assert r["output_hash"] == "def"
        assert r["model"] == "gpt-4"
        assert r["tokens_used"] == 100
        assert r["duration_ms"] == 500
        assert r["details"]["key"] == "value"
        db.close()

    def test_record_disabled_is_noop(self, tmp_path: Path) -> None:
        db = AuditDB(tmp_path, enabled=False)
        row_id = db.record("plan", "test")
        assert row_id == 0
        assert not db.db_path.exists()
        db.close()

    def test_query_empty(self, tmp_path: Path) -> None:
        db = AuditDB(tmp_path)
        # Force DB creation
        db.record("x", "y")
        records = db.query(stage="nonexistent")
        assert records == []
        db.close()

    def test_query_filter_stage(self, tmp_path: Path) -> None:
        db = AuditDB(tmp_path)
        db.record("plan", "start")
        db.record("search", "start")
        db.record("plan", "end")
        records = db.query(stage="plan")
        assert len(records) == 2
        db.close()

    def test_query_filter_action(self, tmp_path: Path) -> None:
        db = AuditDB(tmp_path)
        db.record("plan", "start")
        db.record("plan", "complete")
        records = db.query(action="start")
        assert len(records) == 1
        db.close()

    def test_query_filter_run_id(self, tmp_path: Path) -> None:
        db = AuditDB(tmp_path)
        db.record("plan", "x", run_id="run1")
        db.record("plan", "y", run_id="run2")
        records = db.query(run_id="run1")
        assert len(records) == 1
        db.close()

    def test_query_limit(self, tmp_path: Path) -> None:
        db = AuditDB(tmp_path)
        for i in range(10):
            db.record("plan", f"action_{i}")
        records = db.query(limit=3)
        assert len(records) == 3
        db.close()

    def test_count(self, tmp_path: Path) -> None:
        db = AuditDB(tmp_path)
        db.record("plan", "a")
        db.record("search", "b")
        db.record("plan", "c")
        assert db.count() == 3
        assert db.count(stage="plan") == 2
        db.close()

    def test_count_empty(self, tmp_path: Path) -> None:
        db = AuditDB(tmp_path)
        # Can't count on non-existent db without init
        db.record("x", "y")
        assert db.count(stage="nonexistent") == 0
        db.close()

    def test_close_and_reopen(self, tmp_path: Path) -> None:
        db = AuditDB(tmp_path)
        db.record("plan", "first")
        db.close()

        db2 = AuditDB(tmp_path)
        assert db2.count() == 1
        db2.close()

    def test_db_path_property(self, tmp_path: Path) -> None:
        db = AuditDB(tmp_path)
        assert db.db_path == tmp_path / "logs" / "audit.db"
        db.close()

    def test_enabled_property(self, tmp_path: Path) -> None:
        assert AuditDB(tmp_path, enabled=True).enabled is True
        assert AuditDB(tmp_path, enabled=False).enabled is False

    def test_details_parsed_as_dict(self, tmp_path: Path) -> None:
        db = AuditDB(tmp_path)
        db.record("s", "a", details={"nested": {"key": 1}})
        records = db.query()
        assert isinstance(records[0]["details"], dict)
        assert records[0]["details"]["nested"]["key"] == 1
        db.close()


# ── SnapshotManager ──────────────────────────────────────────────


class TestSnapshotManager:
    """Tests for Channel 3: Environment snapshots."""

    def test_capture_copies_files(self, tmp_path: Path) -> None:
        src = tmp_path / "stage_data"
        src.mkdir()
        (src / "file1.txt").write_text("hello")
        (src / "sub").mkdir()
        (src / "sub" / "file2.txt").write_text("world")

        mgr = SnapshotManager(tmp_path)
        meta = mgr.capture("screen", src)
        assert meta["stage"] == "screen"
        assert meta["file_count"] == 2
        assert mgr.snapshot_count == 1

        snap_dir = tmp_path / "snapshots" / "screen"
        assert (snap_dir / "file1.txt").exists()
        assert (snap_dir / "sub" / "file2.txt").exists()
        assert (snap_dir / "_snapshot_manifest.json").exists()

    def test_capture_with_label(self, tmp_path: Path) -> None:
        src = tmp_path / "data"
        src.mkdir()
        (src / "f.txt").write_text("test")

        mgr = SnapshotManager(tmp_path)
        meta = mgr.capture("plan", src, label="pre")
        assert meta["label"] == "pre"
        assert (tmp_path / "snapshots" / "plan-pre").exists()

    def test_capture_skips_large_files(self, tmp_path: Path) -> None:
        src = tmp_path / "data"
        src.mkdir()
        (src / "small.txt").write_text("ok")
        (src / "large.bin").write_bytes(b"x" * 200)

        mgr = SnapshotManager(tmp_path, max_file_size=100)
        meta = mgr.capture("test", src)
        files = meta["files"]
        small = next(f for f in files if f["path"] == "small.txt")
        large = next(f for f in files if f["path"] == "large.bin")
        assert small["captured"] is True
        assert large["captured"] is False
        assert large["reason"] == "exceeds_max_size"

    def test_capture_disabled_is_noop(self, tmp_path: Path) -> None:
        src = tmp_path / "data"
        src.mkdir()
        (src / "f.txt").write_text("test")

        mgr = SnapshotManager(tmp_path, enabled=False)
        meta = mgr.capture("test", src)
        assert meta["files"] == []
        assert mgr.snapshot_count == 0

    def test_capture_nonexistent_source(self, tmp_path: Path) -> None:
        mgr = SnapshotManager(tmp_path)
        meta = mgr.capture("test", tmp_path / "nonexistent")
        assert meta["error"] == "source_dir_not_found"

    def test_list_snapshots_empty(self, tmp_path: Path) -> None:
        mgr = SnapshotManager(tmp_path)
        assert mgr.list_snapshots() == []

    def test_list_snapshots(self, tmp_path: Path) -> None:
        src = tmp_path / "data"
        src.mkdir()
        (src / "f.txt").write_text("test")

        mgr = SnapshotManager(tmp_path)
        mgr.capture("plan", src)
        mgr.capture("screen", src)
        names = mgr.list_snapshots()
        assert "plan" in names
        assert "screen" in names

    def test_get_manifest(self, tmp_path: Path) -> None:
        src = tmp_path / "data"
        src.mkdir()
        (src / "f.txt").write_text("test")

        mgr = SnapshotManager(tmp_path)
        mgr.capture("plan", src)
        m = mgr.get_manifest("plan")
        assert m is not None
        assert m["stage"] == "plan"

    def test_get_manifest_nonexistent(self, tmp_path: Path) -> None:
        mgr = SnapshotManager(tmp_path)
        assert mgr.get_manifest("nope") is None

    def test_snapshot_dir_property(self, tmp_path: Path) -> None:
        mgr = SnapshotManager(tmp_path)
        assert mgr.snapshot_dir == tmp_path / "snapshots"

    def test_enabled_property(self, tmp_path: Path) -> None:
        assert SnapshotManager(tmp_path).enabled is True
        assert SnapshotManager(tmp_path, enabled=False).enabled is False


# ── EvalLogger ───────────────────────────────────────────────────


class TestEvalLogger:
    """Tests for the unified EvalLogger facade."""

    def test_trace_delegates(self, tmp_path: Path) -> None:
        el = EvalLogger(tmp_path)
        result = el.trace("test", stage="plan")
        assert result["event"] == "test"
        assert el.tracer.trace_count == 1
        el.close()

    def test_record_audit_delegates(self, tmp_path: Path) -> None:
        el = EvalLogger(tmp_path)
        row_id = el.record_audit("plan", "start", run_id="r1")
        assert row_id > 0
        el.close()

    def test_snapshot_delegates(self, tmp_path: Path) -> None:
        src = tmp_path / "data"
        src.mkdir()
        (src / "f.txt").write_text("t")

        el = EvalLogger(tmp_path)
        meta = el.snapshot("plan", src)
        assert meta["stage"] == "plan"
        el.close()

    def test_summary(self, tmp_path: Path) -> None:
        el = EvalLogger(tmp_path)
        el.trace("e1")
        el.record_audit("plan", "start")
        summary = el.summary()
        assert "traces" in summary
        assert "audit_db" in summary
        assert "snapshots" in summary
        assert summary["traces"]["count"] == 1
        assert summary["audit_db"]["total_records"] == 1
        el.close()

    def test_channel_override_flags(self, tmp_path: Path) -> None:
        el = EvalLogger(
            tmp_path,
            traces_enabled=False,
            audit_enabled=True,
            snapshots_enabled=False,
        )
        assert el.tracer.enabled is False
        assert el.audit.enabled is True
        assert el.snapshots.enabled is False
        el.close()

    def test_master_disabled(self, tmp_path: Path) -> None:
        el = EvalLogger(tmp_path, enabled=False)
        assert el.tracer.enabled is False
        assert el.audit.enabled is False
        assert el.snapshots.enabled is False
        el.close()

    def test_run_root_property(self, tmp_path: Path) -> None:
        el = EvalLogger(tmp_path)
        assert el.run_root == tmp_path
        el.close()

    def test_close_is_safe(self, tmp_path: Path) -> None:
        el = EvalLogger(tmp_path)
        el.close()
        el.close()  # Double close should not raise
