"""CLI command for query-adaptive retrieval stopping evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

logger = logging.getLogger(__name__)


def adaptive_stopping_command(
    scores_file: Path = typer.Argument(
        ...,
        help="JSON file with retrieval scores (list of lists, one per batch).",
        exists=True,
        readable=True,
    ),
    query: str = typer.Option(
        "",
        "--query",
        "-q",
        help="Original query for auto-classifying stopping strategy.",
    ),
    query_type: str = typer.Option(
        "auto",
        "--query-type",
        "-t",
        help="Query type: recall, precision, judgment, or auto.",
    ),
    min_results: int = typer.Option(
        5, "--min-results", help="Minimum results before stopping is considered."
    ),
    max_budget: int = typer.Option(
        500, "--max-budget", help="Hard budget limit on total results."
    ),
    relevance_threshold: float = typer.Option(
        0.5,
        "--relevance-threshold",
        help="Score threshold for a result to count as relevant.",
    ),
    output: Path = typer.Option(
        None, "--output", "-o", help="Output JSON path (default: stdout)."
    ),
) -> None:
    """Evaluate adaptive retrieval stopping criteria.

    Reads batch scores from a JSON file and evaluates whether retrieval
    should stop based on the query type and convergence signals.

    The scores file should contain a JSON array of arrays, where each
    inner array is the scores from one retrieval batch.
    """
    from research_pipeline.screening.adaptive_stopping import (
        BatchScores,
        QueryType,
        StoppingState,
        evaluate_stopping,
    )

    raw = json.loads(scores_file.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        typer.echo("Error: scores file must contain a JSON array", err=True)
        raise typer.Exit(1)

    # Parse query type
    try:
        qtype = QueryType(query_type.lower())
    except ValueError:
        typer.echo(
            f"Error: invalid query type '{query_type}'. "
            f"Use: recall, precision, judgment, auto",
            err=True,
        )
        raise typer.Exit(1) from None

    # Build state
    state = StoppingState(
        query_type=qtype,
        max_budget=max_budget,
        min_results=min_results,
        relevance_threshold=relevance_threshold,
    )
    for i, batch in enumerate(raw):
        if isinstance(batch, list):
            state.batches.append(BatchScores(i, [float(s) for s in batch]))
        else:
            typer.echo(f"Warning: batch {i} is not a list, skipping", err=True)

    decision = evaluate_stopping(state, query=query or None)

    result = {
        "should_stop": decision.should_stop,
        "reason": decision.reason.value,
        "details": decision.details,
        "batches_processed": decision.batches_processed,
        "total_results": decision.total_results,
        "current_score": decision.current_score,
        "query_type_used": qtype.value,
    }

    out_text = json.dumps(result, indent=2)
    if output:
        output.write_text(out_text, encoding="utf-8")
        logger.info("Stopping decision written to %s", output)
    else:
        typer.echo(out_text)
