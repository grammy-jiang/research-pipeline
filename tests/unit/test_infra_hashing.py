"""Unit tests for infra.hashing module."""

from pathlib import Path

from arxiv_paper_pipeline.infra.hashing import sha256_bytes, sha256_file, sha256_str


class TestSha256Bytes:
    def test_empty_bytes(self) -> None:
        result = sha256_bytes(b"")
        assert (
            result == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

    def test_known_hash(self) -> None:
        result = sha256_bytes(b"hello")
        assert (
            result == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )

    def test_returns_lowercase_hex(self) -> None:
        result = sha256_bytes(b"test")
        assert result == result.lower()
        assert len(result) == 64


class TestSha256Str:
    def test_utf8_encoding(self) -> None:
        result = sha256_str("hello")
        assert result == sha256_bytes(b"hello")

    def test_unicode(self) -> None:
        result = sha256_str("héllo")
        assert result == sha256_bytes("héllo".encode())

    def test_empty_string(self) -> None:
        result = sha256_str("")
        assert result == sha256_bytes(b"")


class TestSha256File:
    def test_file_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        result = sha256_file(f)
        expected = sha256_bytes(b"hello world")
        assert result == expected

    def test_binary_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.bin"
        data = bytes(range(256))
        f.write_bytes(data)
        result = sha256_file(f)
        assert result == sha256_bytes(data)

    def test_large_file(self, tmp_path: Path) -> None:
        f = tmp_path / "large.txt"
        data = b"x" * 100_000
        f.write_bytes(data)
        result = sha256_file(f)
        assert result == sha256_bytes(data)
