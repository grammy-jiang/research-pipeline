"""Unit tests for MCP tool description integrity hashing."""

from mcp_server.integrity import compute_tool_hashes, verify_tool_integrity

SAMPLE_TOOLS = [
    {"name": "search", "description": "Search for academic papers"},
    {"name": "screen", "description": "Screen candidates by relevance"},
    {"name": "download", "description": "Download PDFs for selected papers"},
]


class TestComputeToolHashes:
    """Tests for compute_tool_hashes."""

    def test_returns_hash_per_tool(self) -> None:
        hashes = compute_tool_hashes(SAMPLE_TOOLS)
        assert len(hashes) == 3
        assert set(hashes.keys()) == {"search", "screen", "download"}

    def test_hashes_are_sha256(self) -> None:
        hashes = compute_tool_hashes(SAMPLE_TOOLS)
        for h in hashes.values():
            assert len(h) == 64  # SHA-256 hex digest
            assert all(c in "0123456789abcdef" for c in h)

    def test_same_description_same_hash(self) -> None:
        tools_a = [{"name": "t", "description": "hello world"}]
        tools_b = [{"name": "t", "description": "hello world"}]
        h_a = compute_tool_hashes(tools_a)
        h_b = compute_tool_hashes(tools_b)
        assert h_a["t"] == h_b["t"]

    def test_different_description_different_hash(self) -> None:
        tools_a = [{"name": "t", "description": "hello world"}]
        tools_b = [{"name": "t", "description": "hello world!"}]
        h_a = compute_tool_hashes(tools_a)
        h_b = compute_tool_hashes(tools_b)
        assert h_a["t"] != h_b["t"]

    def test_empty_tools_list(self) -> None:
        hashes = compute_tool_hashes([])
        assert hashes == {}

    def test_tool_without_description(self) -> None:
        """Tools missing description key get hash of empty string."""
        tools = [{"name": "empty"}]
        hashes = compute_tool_hashes(tools)
        assert "empty" in hashes
        assert len(hashes["empty"]) == 64


class TestVerifyToolIntegrity:
    """Tests for verify_tool_integrity."""

    def test_all_intact(self) -> None:
        hashes = compute_tool_hashes(SAMPLE_TOOLS)
        tampered = verify_tool_integrity(hashes, hashes)
        assert tampered == []

    def test_modified_description(self) -> None:
        original = compute_tool_hashes(SAMPLE_TOOLS)
        modified_tools = [
            {"name": "search", "description": "MODIFIED search description"},
            {"name": "screen", "description": "Screen candidates by relevance"},
            {"name": "download", "description": "Download PDFs for selected papers"},
        ]
        current = compute_tool_hashes(modified_tools)
        tampered = verify_tool_integrity(current, original)
        assert tampered == ["search"]

    def test_missing_tool(self) -> None:
        original = compute_tool_hashes(SAMPLE_TOOLS)
        current = compute_tool_hashes(SAMPLE_TOOLS[:2])
        tampered = verify_tool_integrity(current, original)
        assert "download" in tampered

    def test_empty_reference(self) -> None:
        current = compute_tool_hashes(SAMPLE_TOOLS)
        tampered = verify_tool_integrity(current, {})
        assert tampered == []

    def test_multiple_tampered(self) -> None:
        original = compute_tool_hashes(SAMPLE_TOOLS)
        all_modified = [
            {"name": "search", "description": "changed"},
            {"name": "screen", "description": "also changed"},
            {"name": "download", "description": "Download PDFs for selected papers"},
        ]
        current = compute_tool_hashes(all_modified)
        tampered = verify_tool_integrity(current, original)
        assert "search" in tampered
        assert "screen" in tampered
        assert "download" not in tampered
