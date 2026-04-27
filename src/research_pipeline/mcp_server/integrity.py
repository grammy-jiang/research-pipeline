"""Tool description integrity verification for the MCP server.

Computes SHA-256 hashes of tool descriptions at startup so agents can
verify that tool definitions have not been tampered with (prompt
injection defense via K^n amplification — an attacker must compromise
N tools simultaneously).
"""

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_tool_hashes(tools: list[dict[str, Any]]) -> dict[str, str]:
    """Compute SHA-256 hashes for each tool's description.

    Args:
        tools: List of tool definition dicts, each containing at least
            ``name`` and ``description`` keys.

    Returns:
        Mapping of tool name → SHA-256 hex digest of its description.
    """
    hashes: dict[str, str] = {}
    for tool in tools:
        name = tool.get("name", "")
        desc = tool.get("description", "")
        digest = hashlib.sha256(desc.encode("utf-8")).hexdigest()
        hashes[name] = digest
    return hashes


def verify_tool_integrity(
    current_hashes: dict[str, str],
    reference_hashes: dict[str, str],
) -> list[str]:
    """Verify tool descriptions against reference hashes.

    Args:
        current_hashes: Tool name → current SHA-256 digest.
        reference_hashes: Tool name → expected SHA-256 digest.

    Returns:
        List of tool names whose descriptions have been modified.
        Empty list means all tools are intact.
    """
    tampered: list[str] = []

    for name, expected in reference_hashes.items():
        current = current_hashes.get(name)
        if current is None:
            logger.warning("Tool %r missing from current server", name)
            tampered.append(name)
        elif current != expected:
            logger.warning("Tool %r description hash mismatch", name)
            tampered.append(name)

    return tampered
