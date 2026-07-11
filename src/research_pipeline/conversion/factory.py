"""Converter-backend factory (#109).

Builds a :class:`ConverterBackend` from pipeline configuration — including
multi-account fan-out and cross-service fallback — as shared Core logic. The CLI
(``cli/cmd_convert*.py``), the MCP server, and the pipeline orchestrator all
import from here, so the construction rule lives in one place instead of being
duplicated (and drifting) across the presentation layers. Prior to this the
orchestrator carried its own simplified copy that silently ignored
``fallback_backends`` and multi-account config.
"""

from __future__ import annotations

from research_pipeline.config.models import PipelineConfig
from research_pipeline.conversion.base import ConverterBackend
from research_pipeline.conversion.fallback import FallbackConverter
from research_pipeline.conversion.registry import (
    ensure_builtins_registered,
    get_backend,
)


def backend_kwargs_list(
    backend_name: str,
    config: PipelineConfig,
) -> list[dict[str, object]]:
    """Build constructor kwargs for each account of a converter backend.

    Returns a list of kwargs dicts — one per account. If no multi-account
    config is set, returns a single-element list with the default credentials.
    """
    if backend_name == "docling":
        return [{"timeout_seconds": config.conversion.timeout_seconds}]
    if backend_name == "marker":
        mc = config.conversion.marker
        kwargs: dict[str, object] = {"force_ocr": mc.force_ocr}
        if mc.use_llm:
            kwargs["use_llm"] = True
            if mc.llm_service:
                kwargs["llm_service"] = mc.llm_service
            if mc.llm_api_key:
                kwargs["llm_api_key"] = mc.llm_api_key
        return [kwargs]
    if backend_name == "mathpix":
        mp = config.conversion.mathpix
        if mp.accounts:
            return [
                {"app_id": acct.app_id, "app_key": acct.app_key} for acct in mp.accounts
            ]
        return [{"app_id": mp.app_id, "app_key": mp.app_key}]
    if backend_name == "datalab":
        dl = config.conversion.datalab
        if dl.accounts:
            return [
                {"api_key": acct.api_key, "mode": acct.mode} for acct in dl.accounts
            ]
        return [{"api_key": dl.api_key, "mode": dl.mode}]
    if backend_name == "llamaparse":
        lp = config.conversion.llamaparse
        if lp.accounts:
            return [
                {"api_key": acct.api_key, "tier": acct.tier} for acct in lp.accounts
            ]
        return [{"api_key": lp.api_key, "tier": lp.tier}]
    if backend_name == "mistral_ocr":
        mo = config.conversion.mistral_ocr
        if mo.accounts:
            return [
                {"api_key": acct.api_key, "model": acct.model} for acct in mo.accounts
            ]
        return [{"api_key": mo.api_key, "model": mo.model}]
    if backend_name == "openai_vision":
        ov = config.conversion.openai_vision
        if ov.accounts:
            return [
                {"api_key": acct.api_key, "model": acct.model} for acct in ov.accounts
            ]
        return [{"api_key": ov.api_key, "model": ov.model}]
    if backend_name == "mineru":
        mn = config.conversion.mineru
        return [
            {
                "parse_method": mn.parse_method,
                "timeout_seconds": mn.timeout_seconds,
            }
        ]
    # pymupdf4llm and others: no special kwargs
    return [{}]


def create_converter(config: PipelineConfig) -> ConverterBackend:
    """Create a converter backend from pipeline config.

    If ``fallback_backends`` is configured, creates a FallbackConverter wrapping
    all backends (primary + fallbacks) with all their accounts. Otherwise,
    creates the primary backend only (with multi-account fallback if configured).
    """
    ensure_builtins_registered()

    primary = config.conversion.backend
    backend_names = [primary, *list(config.conversion.fallback_backends)]

    all_backends: list[ConverterBackend] = []
    for name in backend_names:
        kwargs_list = backend_kwargs_list(name, config)
        for kwargs in kwargs_list:
            all_backends.append(get_backend(name, **kwargs))

    if len(all_backends) == 1:
        return all_backends[0]
    return FallbackConverter(all_backends)
