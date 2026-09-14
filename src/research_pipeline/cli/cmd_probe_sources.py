"""CLI adapter for public source access diagnostics."""

import logging
from pathlib import Path

import typer

from research_pipeline.config.loader import load_config
from research_pipeline.infra.logging import setup_logging
from research_pipeline.models.source_probe import ProbeSource
from research_pipeline.sources.source_probe import probe_sources

logger = logging.getLogger(__name__)


def run_probe_sources(
    source: ProbeSource = ProbeSource.ALL,
    config_path: Path | None = None,
    json_output: bool = False,
    output: Path | None = None,
) -> None:
    """Report search endpoint usability; exit 1 when any selected source is unusable."""
    setup_logging(level=logging.INFO)
    try:
        config = load_config(config_path)
        report = probe_sources(source, config=config)
        serialized = report.model_dump_json(indent=2)
        if output is not None:
            output.write_text(serialized + "\n", encoding="utf-8")
    except Exception as exc:
        # Provider/proxy/config errors can contain credentials. Retain only type.
        logger.error("Source probe failed (%s)", type(exc).__name__)
        typer.echo(f"Source probe failed ({type(exc).__name__}).", err=True)
        raise typer.Exit(2) from None
    if json_output:
        typer.echo(serialized)
    else:
        typer.echo("Source\tHTTP\tStatus\tLocal cooldown until (UTC)")
        for item in report.results:
            deadline = item.cooldown_until.isoformat() if item.cooldown_until else "-"
            typer.echo(
                f"{item.source.value}\t{item.http_status or '-'}\t"
                f"{item.status}\t{deadline}"
            )
        typer.echo(
            "Anonymous search endpoints only. An HTTP response does not imply "
            "usable search; "
            "429 does not establish an IP-specific ban."
        )
    raise typer.Exit(0 if report.all_available else 1)
