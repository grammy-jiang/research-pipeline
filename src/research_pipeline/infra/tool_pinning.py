"""Hash-pinned tool definitions for MCP integrity verification.

Computes SHA-256 hashes of tool definitions at registration time and
verifies them at invocation to detect unauthorized modifications
(rug-pull attacks).  This is part of the MCP zero-trust security model.

References:
    Deep-research report Theme 16 (Zero-Trust Tool Verification) and
    Theme 18 (Tool Integrity & Provenance).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from research_pipeline.infra.hashing import sha256_str

logger = logging.getLogger(__name__)


class ToolIntegrityError(Exception):
    """Raised when a tool definition hash does not match its pinned value."""

    def __init__(self, tool_name: str, expected: str, actual: str) -> None:
        self.tool_name = tool_name
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Tool '{tool_name}' integrity check failed: "
            f"expected {expected[:16]}…, got {actual[:16]}…"
        )


@dataclass(frozen=True)
class PinnedTool:
    """A registered tool with its pinned hash.

    Attributes:
        name: Tool name (unique identifier).
        definition_hash: SHA-256 hex digest of the canonical definition.
        version: Optional tool version string.
    """

    name: str
    definition_hash: str
    version: str = ""


@dataclass
class ToolRegistry:
    """Registry of hash-pinned tool definitions.

    Register tools at startup; verify before each invocation.
    """

    _tools: dict[str, PinnedTool] = field(default_factory=dict)
    _definitions: dict[str, dict[str, Any]] = field(default_factory=dict)
    strict: bool = True

    def register(
        self,
        name: str,
        definition: dict[str, Any],
        version: str = "",
    ) -> PinnedTool:
        """Register a tool and pin its definition hash.

        Args:
            name: Unique tool name.
            definition: Tool definition dict (schema, description, etc.).
            version: Optional version string.

        Returns:
            The ``PinnedTool`` created for this registration.

        Raises:
            ValueError: If a tool with this name is already registered.
        """
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")

        canonical = _canonicalize(definition)
        digest = sha256_str(canonical)

        pinned = PinnedTool(name=name, definition_hash=digest, version=version)
        self._tools[name] = pinned
        self._definitions[name] = definition

        logger.debug(
            "Registered tool '%s' with hash %s (version=%s)",
            name,
            digest[:16],
            version or "none",
        )
        return pinned

    def verify(self, name: str, definition: dict[str, Any] | None = None) -> bool:
        """Verify a tool's definition matches its pinned hash.

        Args:
            name: Tool name to verify.
            definition: Current definition to check.  If ``None``,
                re-hashes the originally stored definition (self-check).

        Returns:
            ``True`` if the hash matches.

        Raises:
            KeyError: If the tool is not registered.
            ToolIntegrityError: In strict mode, if the hash does not match.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")

        pinned = self._tools[name]
        check_def = definition if definition is not None else self._definitions[name]
        canonical = _canonicalize(check_def)
        actual_hash = sha256_str(canonical)

        if actual_hash == pinned.definition_hash:
            return True

        logger.warning(
            "Tool '%s' integrity mismatch: expected %s, got %s",
            name,
            pinned.definition_hash[:16],
            actual_hash[:16],
        )

        if self.strict:
            raise ToolIntegrityError(name, pinned.definition_hash, actual_hash)

        return False

    def verify_all(self) -> dict[str, bool]:
        """Verify all registered tools against their pinned hashes.

        Returns:
            Dict mapping tool name to verification result.
        """
        results: dict[str, bool] = {}
        for name in self._tools:
            try:
                results[name] = self.verify(name)
            except ToolIntegrityError:
                results[name] = False
        return results

    def get(self, name: str) -> PinnedTool | None:
        """Get a registered tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[PinnedTool]:
        """List all registered tools."""
        return list(self._tools.values())

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry.

        Args:
            name: Tool name to remove.

        Raises:
            KeyError: If the tool is not registered.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")
        del self._tools[name]
        del self._definitions[name]
        logger.debug("Unregistered tool '%s'", name)

    @property
    def count(self) -> int:
        """Number of registered tools."""
        return len(self._tools)


def _canonicalize(definition: dict[str, Any]) -> str:
    """Produce a canonical JSON string for hashing.

    Keys are sorted, no whitespace, deterministic output.
    """
    return json.dumps(definition, sort_keys=True, separators=(",", ":"))


def compute_definition_hash(definition: dict[str, Any]) -> str:
    """Compute the SHA-256 hash of a tool definition.

    Convenience function for external callers.
    """
    return sha256_str(_canonicalize(definition))
