"""File-based cache with TTL support."""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "research-pipeline"


class FileCache:
    """Simple file-based cache keyed by string, with TTL.

    Cache entries are stored as JSON files. Each entry has a metadata sidecar
    with the creation timestamp for TTL enforcement.
    """

    def __init__(self, cache_dir: Path, ttl_hours: float = 24.0) -> None:
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_hours * 3600
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str) -> Path:
        from research_pipeline.infra.hashing import sha256_str

        hashed = sha256_str(key)
        return self.cache_dir / f"{hashed}.json"

    def _meta_path(self, key: str) -> Path:
        from research_pipeline.infra.hashing import sha256_str

        hashed = sha256_str(key)
        return self.cache_dir / f"{hashed}.meta.json"

    def get(self, key: str) -> str | None:
        """Retrieve a cached value if it exists and has not expired.

        Args:
            key: Cache key.

        Returns:
            Cached string value, or ``None`` if missing/expired.
        """
        path = self._key_path(key)
        meta = self._meta_path(key)
        if not path.exists() or not meta.exists():
            return None

        meta_data = json.loads(meta.read_text(encoding="utf-8"))
        created = meta_data.get("created_at", 0)
        if time.time() - created > self.ttl_seconds:
            logger.debug("Cache expired for key: %s", key[:80])
            path.unlink(missing_ok=True)
            meta.unlink(missing_ok=True)
            return None

        logger.debug("Cache hit for key: %s", key[:80])
        return path.read_text(encoding="utf-8")

    def put(self, key: str, value: str) -> None:
        """Store a value in the cache.

        Args:
            key: Cache key.
            value: String value to cache.
        """
        path = self._key_path(key)
        meta = self._meta_path(key)
        path.write_text(value, encoding="utf-8")
        meta.write_text(
            json.dumps({"key": key[:200], "created_at": time.time()}),
            encoding="utf-8",
        )
        logger.debug("Cached value for key: %s", key[:80])

    def has(self, key: str) -> bool:
        """Check if a non-expired entry exists for the given key.

        Args:
            key: Cache key.

        Returns:
            ``True`` if a valid cache entry exists.
        """
        return self.get(key) is not None

    def invalidate(self, key: str) -> None:
        """Remove a cache entry.

        Args:
            key: Cache key.
        """
        self._key_path(key).unlink(missing_ok=True)
        self._meta_path(key).unlink(missing_ok=True)
        logger.debug("Invalidated cache for key: %s", key[:80])
