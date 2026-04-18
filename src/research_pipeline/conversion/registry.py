"""Backend registry for PDF-to-Markdown converters."""

from __future__ import annotations

import logging
from typing import Any

from research_pipeline.conversion.base import ConverterBackend

logger = logging.getLogger(__name__)

# Maps backend name → backend class
_BACKEND_REGISTRY: dict[str, type[ConverterBackend]] = {}


def register_backend(name: str) -> Any:
    """Class decorator that registers a converter backend under *name*.

    Args:
        name: Unique name for the backend (e.g. ``"docling"``).

    Returns:
        The original class, unchanged.
    """

    def _decorator(cls: type[ConverterBackend]) -> type[ConverterBackend]:
        if name in _BACKEND_REGISTRY:
            raise ValueError(
                f"Backend {name!r} is already registered "
                f"({_BACKEND_REGISTRY[name].__name__})"
            )
        _BACKEND_REGISTRY[name] = cls
        return cls

    return _decorator


def get_backend(name: str, **kwargs: Any) -> ConverterBackend:
    """Instantiate a registered converter backend by name.

    Args:
        name: Registered backend name.
        **kwargs: Passed to the backend constructor.

    Returns:
        An instance of the requested backend.

    Raises:
        ValueError: If the name is not registered.
    """
    if name not in _BACKEND_REGISTRY:
        available = ", ".join(sorted(_BACKEND_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown conversion backend {name!r}. Available: {available}")
    cls = _BACKEND_REGISTRY[name]
    logger.info("Creating converter backend: %s", name)
    return cls(**kwargs)


def list_backends() -> list[str]:
    """Return sorted list of registered backend names."""
    return sorted(_BACKEND_REGISTRY)


def _ensure_builtins_registered() -> None:
    """Import builtin backend modules so their ``@register_backend`` decorators run."""
    # Local backends
    # Online/cloud backends
    import research_pipeline.conversion.datalab_backend
    import research_pipeline.conversion.docling_backend
    import research_pipeline.conversion.llamaparse_backend
    import research_pipeline.conversion.marker_backend
    import research_pipeline.conversion.mathpix_backend
    import research_pipeline.conversion.mineru_backend
    import research_pipeline.conversion.mistral_ocr_backend
    import research_pipeline.conversion.openai_vision_backend
    import research_pipeline.conversion.pymupdf4llm_backend  # noqa: F401
