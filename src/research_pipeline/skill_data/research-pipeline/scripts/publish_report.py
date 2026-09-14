#!/usr/bin/env python3
"""Publish exactly the draft whose content passed validation."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def publish(draft: Path, validation: Path, destination: Path) -> None:
    data = draft.read_bytes()
    verdict = json.loads(validation.read_text())
    if verdict.get("passed") is not True:
        raise ValueError("Report validation did not pass")
    if verdict.get("report_sha256") != hashlib.sha256(data).hexdigest():
        raise ValueError("Report validation hash does not match the draft")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as fh:
        temporary = Path(fh.name)
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    publish(args.draft, args.validation, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
