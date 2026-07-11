# Getting Started

A five-minute, hands-on tutorial: install research-pipeline and produce your
first literature-review report. Follow it top to bottom — every step builds on
the previous one, and you will finish with a real synthesis report on disk.

> This is a **tutorial** (one guided happy path). For the full command and
> configuration reference see the [User Manual](user-manual.md) and the
> [Configuration reference](user-guide.md); for every CLI command and MCP tool
> see the [API Reference](api-reference.md).

## Before you start

You need **Python 3.12 or newer** and network access to arXiv. That is all —
this tutorial uses only the free arXiv source and a local PDF converter, so no
API keys are required.

## Step 1 — Install

Install the package with the recommended local PDF-conversion backend:

```bash
pip install "research-pipeline[docling]"
```

Verify the CLI is on your path:

```bash
research-pipeline --version
```

You should see a version number printed. If the command is not found, make sure
your Python scripts directory is on your `PATH`.

## Step 2 — Run your first pipeline

Give the pipeline a research topic in plain English and let it run end to end:

```bash
research-pipeline run "attention mechanisms in transformer models"
```

Behind the scenes it runs seven stages in order:

1. **plan** — turn your topic into structured search queries
2. **search** — query arXiv (and any other enabled sources)
3. **screen** — score and shortlist the most relevant papers
4. **download** — fetch PDFs for the top candidates
5. **convert** — turn those PDFs into Markdown
6. **extract** — pull out the key sections
7. **summarize** — write per-paper summaries and a cross-paper synthesis

Progress prints to the terminal as each stage completes. A typical first run
takes **2–5 minutes**. When it finishes, note the **run ID** it prints — a
timestamp like `20260711T190500Z`.

## Step 3 — Find and read your results

List your runs and inspect the one you just created:

```bash
research-pipeline inspect                       # list all runs
research-pipeline inspect --run-id <RUN_ID>     # details for one run
```

Everything for a run lives under `./runs/<RUN_ID>/`. The files you care about
first:

| Path | Contents |
|------|----------|
| `runs/<ID>/synthesis.md` | **Main output** — your cross-paper synthesis report |
| `runs/<ID>/summaries/<ID>.md` | One summary per paper |
| `runs/<ID>/screened.jsonl` | The shortlisted papers and their relevance scores |
| `runs/<ID>/run_manifest.json` | Which stages completed, with artifact hashes |

Open `synthesis.md` in any Markdown viewer to read your literature review.

Prefer a browser or a citation manager? Export the report:

```bash
research-pipeline export-html --run-id <RUN_ID>     # → runs/<ID>/synthesis.html
research-pipeline export-bibtex --run-id <RUN_ID>   # → runs/<ID>/screened.bib
```

## What next

You just ran the whole pipeline with one command. From here you can go deeper:

- Run stages individually, or re-run just one — see the [User Manual](user-manual.md).
- Add more sources, tune screening, or wire up cloud converters and API keys —
  see the [Configuration reference](user-guide.md).
- Drive the pipeline from an MCP client — see the [API Reference](api-reference.md).
