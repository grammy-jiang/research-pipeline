"""Persistent per-source request spacing and cooldowns for local processes."""

import contextlib
import hashlib
import json
import math
import os
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import requests


class SourceCooldown(requests.HTTPError):
    """A source must remain paused; retrying this operation cannot help yet."""

    retryable = False

    def __init__(
        self,
        source: str,
        not_before: float,
        response: requests.Response | None = None,
    ) -> None:
        self.not_before = not_before
        deadline = datetime.fromtimestamp(not_before, UTC).isoformat()
        super().__init__(
            f"Source {source} is cooling down until {deadline}", response=response
        )


class SharedRequestBudget:
    """Coordinate a source's full requests and preserve server deadlines.

    The state directory must be shared by processes that share a service
    budget. Separate hosts require a shared coordinator or external scheduling.
    """

    def __init__(
        self, source: str, min_interval: float, state_dir: Path | None = None
    ) -> None:
        self.source = source
        self.min_interval = max(0.0, min_interval)
        default_dir = os.environ.get(
            "RESEARCH_PIPELINE_REQUEST_STATE_DIR",
            "~/.cache/research-pipeline/request-budgets",
        )
        self.directory = (state_dir or Path(default_dir)).expanduser()
        self.key = hashlib.sha256(source.encode()).hexdigest()[:24]
        self._lock = threading.RLock()
        self._active: dict[str, float] | None = None

    @contextlib.contextmanager
    def _locked(self) -> Iterator[dict[str, float]]:
        # fcntl is intentionally imported at use time: pure parsing remains
        # available on platforms without POSIX locks.
        import fcntl

        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.directory / f"{self.key}.json"
        with self._lock, (self.directory / f"{self.key}.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                state = json.loads(path.read_text()) if path.exists() else {}
                if not isinstance(state, dict) or any(
                    not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in state.values()
                ):
                    raise ValueError("Invalid shared request-budget state")
                yield state
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _save(self, state: dict[str, float]) -> None:
        target = self.directory / f"{self.key}.json"
        temporary = target.with_suffix(".tmp")
        with temporary.open("w") as handle:
            handle.write(json.dumps(state))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)

    def _check(self, state: dict[str, float]) -> None:
        deadline = state.get("not_before", 0.0)
        if deadline > time.time():
            raise SourceCooldown(self.source, deadline)

    def check_available(self) -> None:
        """Fail before dispatch if a previous process left an active cooldown."""
        with self._locked() as state:
            self._check(state)

    @contextlib.contextmanager
    def slot(self) -> Iterator[None]:
        """Hold the source lock through the request, including its response."""
        with self._locked() as state:
            self._check(state)
            if "last_start" in state:
                wait = max(0.0, state["last_start"] + self.min_interval - time.time())
                if wait:
                    time.sleep(wait)
            state["last_start"] = time.time()
            self._save(state)
            self._active = state
            try:
                yield
            finally:
                self._save(state)
                self._active = None

    def defer(
        self, seconds: float, response: requests.Response | None = None
    ) -> SourceCooldown:
        """Persist a deadline while holding a request slot, then stop the source."""
        if self._active is None:
            raise RuntimeError("A cooldown must be recorded inside a request slot")
        deadline = time.time() + max(self.min_interval, seconds)
        self._active["not_before"] = max(self._active.get("not_before", 0.0), deadline)
        self._save(self._active)
        return SourceCooldown(self.source, self._active["not_before"], response)
