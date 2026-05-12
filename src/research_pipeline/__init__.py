"""research-pipeline: multi-source academic paper search and analysis."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("research-pipeline")
except PackageNotFoundError:
    __version__ = "unknown"
