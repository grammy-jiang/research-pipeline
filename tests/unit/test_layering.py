"""Layering guard: Core packages must not import the CLI layer (#109).

The pipeline orchestrator used to import stage logic back out of ``cli/cmd_*``,
inverting the documented dependency direction and forming a real import cycle.
This guard fails if any module under ``pipeline/`` imports ``research_pipeline.cli``
again — at module level or inside a function.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src" / "research_pipeline"


def _is_cli(module: str) -> bool:
    return module == "research_pipeline.cli" or module.startswith(
        "research_pipeline.cli."
    )


def _cli_imports(py: Path) -> list[str]:
    """Return every research_pipeline.cli import in *py* (module-level or local)."""
    tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and _is_cli(node.module):
            hits.append(f"{py.name}:{node.lineno} from {node.module}")
        elif isinstance(node, ast.Import):
            hits.extend(
                f"{py.name}:{node.lineno} import {a.name}"
                for a in node.names
                if _is_cli(a.name)
            )
    return hits


@pytest.mark.parametrize("package", ["pipeline", "mcp_server"])
def test_core_package_does_not_import_cli(package: str) -> None:
    """Neither Core surface may import the CLI layer (#109).

    ``pipeline/`` is Core (the orchestrator cycle); ``mcp_server/`` is the other
    presentation surface, which must depend on Core rather than reach into
    ``cli/`` privates. Both are pinned so the boundary can't erode again.
    """
    offenders: list[str] = []
    for py in sorted((_SRC / package).rglob("*.py")):
        offenders.extend(_cli_imports(py))
    assert not offenders, (
        f"{package}/ must not import the CLI layer (#109 layering); found: {offenders}"
    )
