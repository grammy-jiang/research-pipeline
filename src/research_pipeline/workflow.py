"""Entry point for the bundled, manifest-governed research workflow."""

import runpy
from pathlib import Path
from typing import cast


def main() -> int:
    runner = Path(__file__).parent / "skill_data/research-pipeline/runners/runner.py"
    namespace = runpy.run_path(str(runner))
    return cast(int, namespace["main"]())
