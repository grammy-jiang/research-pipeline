"""Structured JSONL logging setup."""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path


class JSONLFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_data"):
            entry["data"] = record.extra_data
        return json.dumps(entry, default=str)


def setup_logging(
    level: int = logging.INFO,
    log_dir: Path | None = None,
) -> None:
    """Configure structured logging for the package.

    Args:
        level: Logging level.
        log_dir: If provided, also write JSONL logs to a file in this directory.
    """
    root = logging.getLogger("research_pipeline")
    root.setLevel(level)

    if not root.handlers:
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(level)
        console.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "pipeline.jsonl"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(JSONLFormatter())
        root.addHandler(file_handler)


def enable_run_logging(run_root: Path) -> None:
    """Attach a file handler to the run's logs directory.

    Call this after ``init_run()`` so that all subsequent log messages
    are captured in ``<run_root>/logs/pipeline.jsonl``.

    Args:
        run_root: Root directory of the pipeline run.
    """
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.jsonl"

    root = logging.getLogger("research_pipeline")

    # Avoid adding duplicate file handlers to the same log file
    for handler in root.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == log_file.resolve()
        ):
            return

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(root.level)
    file_handler.setFormatter(JSONLFormatter())
    root.addHandler(file_handler)
