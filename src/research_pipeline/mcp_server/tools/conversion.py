from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from research_pipeline.mcp_server.schemas import (
    ConvertFileInput,
    ConvertFineInput,
    ConvertPdfsInput,
    ConvertRoughInput,
    ListBackendsInput,
    ToolResult,
)
from research_pipeline.mcp_server.tools._common import (
    _backend_kwargs,
    _log_info,
    _raise_tool_error,
    _report_progress,
    _resolve_run_id,
    _resolve_workspace,
    logger,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context


def convert_pdfs(params: ConvertPdfsInput, ctx: Context | None = None) -> ToolResult:
    """Convert downloaded PDFs to Markdown using FallbackConverter."""
    try:
        from research_pipeline.config.loader import load_config
        from research_pipeline.conversion.factory import create_converter
        from research_pipeline.models.download import DownloadManifestEntry
        from research_pipeline.storage.manifests import read_jsonl, write_jsonl
        from research_pipeline.storage.workspace import get_stage_dir, init_run

        config = load_config()
        if params.backend:
            config.conversion.backend = params.backend

        ws = _resolve_workspace(params.workspace)
        rid = _resolve_run_id(params.run_id)
        _rid, run_root = init_run(ws, rid)

        dl_manifest_path = (
            get_stage_dir(run_root, "download_root") / "download_manifest.jsonl"
        )
        if not dl_manifest_path.exists():
            return ToolResult(
                success=False,
                message="No download manifest found. Run tool_download_pdfs first.",
            )

        raw = read_jsonl(dl_manifest_path)
        entries = [DownloadManifestEntry.model_validate(d) for d in raw]

        converter = create_converter(config)
        convert_dir = get_stage_dir(run_root, "convert")
        convert_dir.mkdir(parents=True, exist_ok=True)

        eligible = [e for e in entries if e.status in ("downloaded", "skipped_exists")]
        _log_info(ctx, f"Starting conversion of {len(eligible)} PDFs")

        results = []
        total = len(eligible)
        for i, entry in enumerate(eligible):
            pdf_path = Path(entry.local_path)
            if not pdf_path.exists():
                logger.warning("PDF not found: %s", pdf_path)
                continue
            _report_progress(ctx, i, total, f"Converting {pdf_path.name}")
            result = converter.convert(pdf_path, convert_dir, force=params.force)
            results.append(result)
        _report_progress(ctx, total, total, "Conversion complete")

        conv_manifest = (
            get_stage_dir(run_root, "convert_root") / "convert_manifest.jsonl"
        )
        write_jsonl(conv_manifest, [r.model_dump(mode="json") for r in results])

        converted = sum(1 for r in results if r.status == "converted")
        skipped = sum(1 for r in results if r.status == "skipped_exists")
        failed = sum(1 for r in results if r.status == "failed")
        logger.info(
            "Convert stage: %d converted, %d skipped, %d failed",
            converted,
            skipped,
            failed,
        )
        return ToolResult(
            success=True,
            message=(
                f"Converted {converted}/{len(results)} PDFs to Markdown, "
                f"{skipped} skipped, {failed} failed "
                f"(backend={config.conversion.backend})."
            ),
            artifacts={
                "manifest": str(conv_manifest),
                "convert_dir": str(convert_dir),
                "converted": converted,
                "skipped": skipped,
                "failed": failed,
                "backend": config.conversion.backend,
            },
        )
    except ImportError as exc:
        return ToolResult(
            success=False,
            message=(
                f"Converter backend is not installed: {exc}. "
                "Install the corresponding extra."
            ),
        )
    except Exception as exc:
        _raise_tool_error("convert_pdfs", exc)


def convert_file(params: ConvertFileInput, ctx: Context | None = None) -> ToolResult:
    """Convert a single PDF file to Markdown (standalone, no workspace needed)."""
    try:
        from research_pipeline.config.loader import load_config
        from research_pipeline.conversion.registry import (
            ensure_builtins_registered,
            get_backend,
        )

        ensure_builtins_registered()

        pdf_path = Path(params.pdf_path).expanduser().resolve()
        if not pdf_path.exists():
            return ToolResult(
                success=False,
                message=f"PDF file not found: {pdf_path}",
            )

        output_dir = Path(params.output_dir) if params.output_dir else pdf_path.parent
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        config = load_config()
        backend_name = params.backend or config.conversion.backend
        kwargs = _backend_kwargs(backend_name, config)
        backend = get_backend(backend_name, **kwargs)

        result = backend.convert(pdf_path, output_dir)

        md_path = output_dir / f"{pdf_path.stem}.md"
        logger.info(
            "Converted %s → %s (status=%s, backend=%s)",
            pdf_path,
            md_path,
            result.status,
            backend_name,
        )
        return ToolResult(
            success=result.status in ("converted", "skipped_exists"),
            message=(
                f"Conversion {result.status}: {pdf_path.name} → {md_path.name} "
                f"(backend={backend_name})"
            ),
            artifacts={
                "markdown_path": str(md_path) if md_path.exists() else "",
                "status": result.status,
                "backend": backend_name,
            },
        )
    except ImportError as exc:
        return ToolResult(
            success=False,
            message=(
                f"Converter backend is not installed: {exc}. "
                "Install the corresponding extra."
            ),
        )
    except Exception as exc:
        _raise_tool_error("convert_file", exc)


def list_backends(params: ListBackendsInput, ctx: Context | None = None) -> ToolResult:
    """List available converter backends."""
    try:
        from research_pipeline.conversion.registry import (
            ensure_builtins_registered,
        )
        from research_pipeline.conversion.registry import list_backends as _list

        ensure_builtins_registered()
        backends = _list()
        return ToolResult(
            success=True,
            message=f"Available backends: {', '.join(backends)}",
            artifacts={"backends": backends},
        )
    except Exception as exc:
        _raise_tool_error("list_backends", exc)


def convert_rough(params: ConvertRoughInput, ctx: Context | None = None) -> ToolResult:
    """Fast Tier 2 conversion of all downloaded PDFs via pymupdf4llm."""
    try:
        from research_pipeline.conversion.registry import (
            ensure_builtins_registered,
            get_backend,
        )
        from research_pipeline.models.download import DownloadManifestEntry
        from research_pipeline.storage.manifests import read_jsonl, write_jsonl
        from research_pipeline.storage.workspace import get_stage_dir, init_run

        ws = _resolve_workspace(params.workspace)
        rid = _resolve_run_id(params.run_id)
        _rid, run_root = init_run(ws, rid)

        dl_manifest_path = (
            get_stage_dir(run_root, "download_root") / "download_manifest.jsonl"
        )
        if not dl_manifest_path.exists():
            return ToolResult(
                success=False,
                message="No download manifest found. Run tool_download_pdfs first.",
            )

        raw = read_jsonl(dl_manifest_path)
        entries = [DownloadManifestEntry.model_validate(d) for d in raw]

        ensure_builtins_registered()
        converter = get_backend("pymupdf4llm")

        rough_dir = get_stage_dir(run_root, "convert_rough")
        rough_dir.mkdir(parents=True, exist_ok=True)

        _log_info(ctx, f"Starting rough conversion of {len(entries)} entries")

        results = []
        eligible = [e for e in entries if e.status in ("downloaded", "skipped_exists")]
        total = len(eligible)
        for i, entry in enumerate(eligible):
            pdf_path = Path(entry.local_path)
            if not pdf_path.exists():
                logger.warning("PDF not found: %s", pdf_path)
                continue
            _report_progress(ctx, i, total, f"Converting {pdf_path.name}")
            result = converter.convert(pdf_path, rough_dir, force=params.force)
            results.append(result)
        _report_progress(ctx, total, total, "Rough conversion complete")

        manifest_path = rough_dir / "convert_rough_manifest.jsonl"
        records = [r.model_dump(mode="json") for r in results]
        for rec in records:
            rec["tier"] = "rough"
        write_jsonl(manifest_path, records)

        converted = sum(1 for r in results if r.status == "converted")
        skipped = sum(1 for r in results if r.status == "skipped_exists")
        failed = sum(1 for r in results if r.status == "failed")

        logger.info(
            "Rough conversion: %d converted, %d skipped, %d failed",
            converted,
            skipped,
            failed,
        )
        return ToolResult(
            success=True,
            message=(
                f"Rough conversion: {converted} converted, "
                f"{skipped} skipped, {failed} failed."
            ),
            artifacts={
                "manifest": str(manifest_path),
                "run_id": _rid,
                "converted": converted,
                "skipped": skipped,
                "failed": failed,
            },
        )
    except Exception as exc:
        _raise_tool_error("convert_rough", exc)


def convert_fine(params: ConvertFineInput, ctx: Context | None = None) -> ToolResult:
    """High-quality Tier 3 conversion of selected PDFs."""
    try:
        from research_pipeline.config.loader import load_config
        from research_pipeline.conversion.factory import create_converter
        from research_pipeline.models.download import DownloadManifestEntry
        from research_pipeline.storage.manifests import read_jsonl, write_jsonl
        from research_pipeline.storage.workspace import get_stage_dir, init_run

        if not params.paper_ids:
            return ToolResult(success=False, message="No paper IDs provided.")

        config = load_config()
        if params.backend:
            config.conversion.backend = params.backend
        ws = _resolve_workspace(params.workspace)
        rid = _resolve_run_id(params.run_id)
        _rid, run_root = init_run(ws, rid)

        dl_manifest_path = (
            get_stage_dir(run_root, "download_root") / "download_manifest.jsonl"
        )
        if not dl_manifest_path.exists():
            return ToolResult(
                success=False,
                message="No download manifest found. Run tool_download_pdfs first.",
            )

        raw = read_jsonl(dl_manifest_path)
        entries = [DownloadManifestEntry.model_validate(d) for d in raw]

        id_set = set(params.paper_ids)
        selected = [
            e
            for e in entries
            if e.arxiv_id in id_set and e.status in ("downloaded", "skipped_exists")
        ]

        if not selected:
            return ToolResult(
                success=False,
                message="No matching downloaded papers found for given IDs.",
            )

        converter = create_converter(config)

        fine_dir = get_stage_dir(run_root, "convert_fine")
        fine_dir.mkdir(parents=True, exist_ok=True)

        _log_info(ctx, f"Starting fine conversion of {len(selected)} papers")

        results = []
        total = len(selected)
        for i, entry in enumerate(selected):
            pdf_path = Path(entry.local_path)
            if not pdf_path.exists():
                logger.warning("PDF not found: %s", pdf_path)
                continue
            _report_progress(ctx, i, total, f"Converting {pdf_path.name}")
            result = converter.convert(pdf_path, fine_dir, force=params.force)
            results.append(result)
        _report_progress(ctx, total, total, "Fine conversion complete")

        manifest_path = fine_dir / "convert_fine_manifest.jsonl"
        records = [r.model_dump(mode="json") for r in results]
        for rec in records:
            rec["tier"] = "fine"
        write_jsonl(manifest_path, records)

        converted = sum(1 for r in results if r.status == "converted")
        skipped = sum(1 for r in results if r.status == "skipped_exists")
        failed = sum(1 for r in results if r.status == "failed")

        logger.info(
            "Fine conversion: %d converted, %d skipped, %d failed",
            converted,
            skipped,
            failed,
        )
        return ToolResult(
            success=True,
            message=(
                f"Fine conversion: {converted} converted, "
                f"{skipped} skipped, {failed} failed."
            ),
            artifacts={
                "manifest": str(manifest_path),
                "run_id": _rid,
                "converted": converted,
                "skipped": skipped,
                "failed": failed,
            },
        )
    except Exception as exc:
        _raise_tool_error("convert_fine", exc)
