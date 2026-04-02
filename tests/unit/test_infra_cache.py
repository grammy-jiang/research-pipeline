"""Unit tests for infra.cache module."""

import json
import time
from pathlib import Path

from research_pipeline.infra.cache import FileCache


class TestFileCache:
    def test_put_and_get(self, tmp_path: Path) -> None:
        cache = FileCache(tmp_path / "cache", ttl_hours=1.0)
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing_key(self, tmp_path: Path) -> None:
        cache = FileCache(tmp_path / "cache", ttl_hours=1.0)
        assert cache.get("nonexistent") is None

    def test_has_existing_key(self, tmp_path: Path) -> None:
        cache = FileCache(tmp_path / "cache", ttl_hours=1.0)
        cache.put("key1", "value1")
        assert cache.has("key1") is True

    def test_has_missing_key(self, tmp_path: Path) -> None:
        cache = FileCache(tmp_path / "cache", ttl_hours=1.0)
        assert cache.has("nonexistent") is False

    def test_ttl_expiry(self, tmp_path: Path) -> None:
        cache = FileCache(tmp_path / "cache", ttl_hours=0.0001)  # Very short TTL
        cache.put("key1", "value1")
        # Manually set metadata to expired time
        meta_path = cache._meta_path("key1")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["created_at"] = time.time() - 7200  # 2 hours ago
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        assert cache.get("key1") is None

    def test_invalidate(self, tmp_path: Path) -> None:
        cache = FileCache(tmp_path / "cache", ttl_hours=1.0)
        cache.put("key1", "value1")
        assert cache.has("key1")
        cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_invalidate_nonexistent(self, tmp_path: Path) -> None:
        cache = FileCache(tmp_path / "cache", ttl_hours=1.0)
        # Should not raise
        cache.invalidate("nonexistent")

    def test_overwrite_value(self, tmp_path: Path) -> None:
        cache = FileCache(tmp_path / "cache", ttl_hours=1.0)
        cache.put("key1", "value1")
        cache.put("key1", "value2")
        assert cache.get("key1") == "value2"

    def test_unicode_values(self, tmp_path: Path) -> None:
        cache = FileCache(tmp_path / "cache", ttl_hours=1.0)
        cache.put("unicode", "héllo wörld 🌍")
        assert cache.get("unicode") == "héllo wörld 🌍"

    def test_creates_cache_directory(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "deep" / "nested" / "cache"
        assert not cache_dir.exists()
        FileCache(cache_dir, ttl_hours=1.0)
        assert cache_dir.exists()

    def test_different_keys_independent(self, tmp_path: Path) -> None:
        cache = FileCache(tmp_path / "cache", ttl_hours=1.0)
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        cache.invalidate("key1")
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
