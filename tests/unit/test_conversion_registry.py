"""Unit tests for the conversion backend registry."""

from __future__ import annotations

import pytest

from research_pipeline.conversion.base import ConverterBackend
from research_pipeline.conversion.registry import (
    _BACKEND_REGISTRY,
    KNOWN_BACKENDS,
    ensure_builtins_registered,
    get_backend,
    list_backends,
    register_backend,
)


@pytest.fixture(autouse=True)
def _clean_test_backends() -> object:
    """Drop any ``_test_*`` backends a test registers, unconditionally (#117).

    These tests register throwaway backends via ``@register_backend`` into the
    process-global ``_BACKEND_REGISTRY``. Teardown here runs even if an assert
    fails, so a test backend can no longer leak into every later test — while
    leaving the builtins (registered once via a cached import) in place.
    """
    yield
    for name in [n for n in _BACKEND_REGISTRY if n.startswith("_test")]:
        del _BACKEND_REGISTRY[name]


def test_known_backends_matches_registry() -> None:
    """KNOWN_BACKENDS must stay in lock-step with the @register_backend decorators.

    Guards #120: the canonical list feeds the MCP schema descriptions and the
    completions fallback, so it must not drift from the live registry.
    """
    ensure_builtins_registered()
    assert set(KNOWN_BACKENDS) == set(list_backends())


class TestRegisterBackend:
    """Tests for the @register_backend decorator."""

    def test_register_and_retrieve(self) -> None:
        """A decorated class appears in the registry."""

        @register_backend("_test_dummy")
        class _DummyBackend(ConverterBackend):
            def convert(self, pdf_path, output_dir, *, force=False):  # type: ignore[override]
                ...

            def fingerprint(self) -> str:
                return "dummy/0/000"

        assert "_test_dummy" in _BACKEND_REGISTRY
        assert _BACKEND_REGISTRY["_test_dummy"] is _DummyBackend

    def test_duplicate_registration_raises(self) -> None:
        """Registering the same name twice raises ValueError."""

        @register_backend("_test_dup")
        class _First(ConverterBackend):
            def convert(self, pdf_path, output_dir, *, force=False):  # type: ignore[override]
                ...

            def fingerprint(self) -> str:
                return "first/0/000"

        with pytest.raises(ValueError, match="already registered"):

            @register_backend("_test_dup")
            class _Second(ConverterBackend):
                def convert(self, pdf_path, output_dir, *, force=False):  # type: ignore[override]
                    ...

                def fingerprint(self) -> str:
                    return "second/0/000"


class TestGetBackend:
    """Tests for get_backend factory."""

    def test_unknown_backend_raises(self) -> None:
        """Requesting an unregistered backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown conversion backend"):
            get_backend("nonexistent_backend_xyz")

    def test_get_backend_passes_kwargs(self) -> None:
        """Constructor kwargs are forwarded to the backend class."""

        @register_backend("_test_kwargs")
        class _KwargsBackend(ConverterBackend):
            def __init__(self, timeout: int = 100) -> None:
                self.timeout = timeout

            def convert(self, pdf_path, output_dir, *, force=False):  # type: ignore[override]
                ...

            def fingerprint(self) -> str:
                return "kwargs/0/000"

        backend = get_backend("_test_kwargs", timeout=42)
        assert backend.timeout == 42  # type: ignore[attr-defined]


class TestListBackends:
    """Tests for list_backends."""

    def test_returns_sorted_names(self) -> None:
        """list_backends returns names in sorted order."""
        ensure_builtins_registered()
        names = list_backends()
        assert names == sorted(names)
        assert isinstance(names, list)


class TestEnsureBuiltinsRegistered:
    """Tests for ensure_builtins_registered."""

    def test_builtins_are_registered(self) -> None:
        """All three builtin backends are registered after ensure call."""
        ensure_builtins_registered()
        names = list_backends()
        assert "docling" in names
        assert "marker" in names
        assert "pymupdf4llm" in names

    def test_get_docling_backend(self) -> None:
        """DoclingBackend can be instantiated via registry."""
        ensure_builtins_registered()
        backend = get_backend("docling")
        assert backend.fingerprint().startswith("docling/")

    def test_get_marker_backend(self) -> None:
        """MarkerBackend can be instantiated via registry."""
        ensure_builtins_registered()
        backend = get_backend("marker")
        assert backend.fingerprint().startswith("marker/")

    def test_get_pymupdf4llm_backend(self) -> None:
        """PyMuPDF4LLMBackend can be instantiated via registry."""
        ensure_builtins_registered()
        backend = get_backend("pymupdf4llm")
        assert backend.fingerprint().startswith("pymupdf4llm/")

    def test_marker_backend_with_llm_kwargs(self) -> None:
        """MarkerBackend accepts LLM configuration kwargs."""
        ensure_builtins_registered()
        backend = get_backend(
            "marker",
            force_ocr=True,
            use_llm=True,
            llm_service="gemini",
        )
        assert "marker/" in backend.fingerprint()
