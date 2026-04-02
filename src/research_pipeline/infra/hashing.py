"""Content hashing utilities."""

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes.

    Args:
        data: Bytes to hash.

    Returns:
        Lowercase hex digest string.
    """
    return hashlib.sha256(data).hexdigest()


def sha256_str(text: str) -> str:
    """Compute SHA-256 hex digest of a string (UTF-8 encoded).

    Args:
        text: String to hash.

    Returns:
        Lowercase hex digest string.
    """
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA-256 hex digest of a file.

    Args:
        path: Path to the file.
        chunk_size: Read buffer size.

    Returns:
        Lowercase hex digest string.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    digest = h.hexdigest()
    logger.debug("SHA-256 of %s: %s", path, digest)
    return digest
