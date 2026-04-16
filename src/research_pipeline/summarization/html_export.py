"""HTML report export using Jinja2 templates.

Converts a SynthesisReport (or raw Markdown synthesis report) into a
self-contained HTML document with navigation, citation links,
confidence badges, and responsive styling.

References:
    - Deep Research Report §A5: Jinja2 HTML synthesis report
"""

from __future__ import annotations

import html
import logging
import re
from pathlib import Path

from jinja2 import BaseLoader, Environment
from markupsafe import Markup

from research_pipeline.models.summary import SynthesisReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confidence badge mapping
# ---------------------------------------------------------------------------

CONFIDENCE_COLORS: dict[str, str] = {
    "high": "#22c55e",
    "medium": "#eab308",
    "low": "#ef4444",
    "unknown": "#94a3b8",
}

CONFIDENCE_LABELS: dict[str, str] = {
    "high": "High Confidence",
    "medium": "Medium Confidence",
    "low": "Low Confidence",
    "unknown": "Unrated",
}


def _detect_confidence(text: str) -> str:
    """Detect confidence level from text markers.

    Args:
        text: Text to scan for confidence markers.

    Returns:
        Confidence level string.
    """
    lower = text.lower()
    if any(w in lower for w in ["strong evidence", "well-established", "robust"]):
        return "high"
    if any(w in lower for w in ["some evidence", "moderate", "preliminary"]):
        return "medium"
    if any(w in lower for w in ["limited evidence", "unclear", "speculative"]):
        return "low"
    return "unknown"


# ---------------------------------------------------------------------------
# Citation link helpers
# ---------------------------------------------------------------------------

_ARXIV_PATTERN = re.compile(r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b")


def _linkify_arxiv_ids(text: str) -> str:
    """Convert arXiv IDs in text to clickable links.

    Args:
        text: Input text potentially containing arXiv IDs.

    Returns:
        Text with arXiv IDs wrapped in anchor tags.
    """

    def _replace(match: re.Match[str]) -> str:
        aid = match.group(1)
        return (
            f'<a href="https://arxiv.org/abs/{aid}" '
            f'target="_blank" class="arxiv-link">{aid}</a>'
        )

    return _ARXIV_PATTERN.sub(_replace, text)


def _escape(text: str) -> str:
    """HTML-escape text then linkify arXiv IDs.

    Args:
        text: Raw text.

    Returns:
        Safe HTML string with linked arXiv IDs.
    """
    escaped = html.escape(text)
    return _linkify_arxiv_ids(escaped)


# ---------------------------------------------------------------------------
# Jinja2 HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title | e }} — Research Report</title>
<style>
  :root {
    --bg: #ffffff; --fg: #1e293b; --muted: #64748b;
    --border: #e2e8f0; --accent: #3b82f6; --surface: #f8fafc;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f172a; --fg: #e2e8f0; --muted: #94a3b8;
      --border: #334155; --accent: #60a5fa; --surface: #1e293b;
    }
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--fg);
    line-height: 1.6; max-width: 960px; margin: 0 auto; padding: 2rem 1rem;
  }
  h1 { font-size: 1.75rem; margin-bottom: 0.5rem; }
  h2 { font-size: 1.35rem; margin: 2rem 0 0.75rem; border-bottom: 2px solid var(--border); padding-bottom: 0.25rem; }
  h3 { font-size: 1.1rem; margin: 1.5rem 0 0.5rem; }
  nav { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; margin: 1.5rem 0; }
  nav ul { list-style: none; columns: 2; }
  nav li { margin: 0.25rem 0; }
  nav a { color: var(--accent); text-decoration: none; }
  nav a:hover { text-decoration: underline; }
  .meta { color: var(--muted); font-size: 0.9rem; margin-bottom: 1rem; }
  .badge {
    display: inline-block; font-size: 0.75rem; font-weight: 600;
    padding: 0.15rem 0.5rem; border-radius: 9999px; color: #fff;
    vertical-align: middle; margin-left: 0.4rem;
  }
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 1rem; margin: 1rem 0;
  }
  .card h3 { margin-top: 0; }
  ul, ol { padding-left: 1.5rem; margin: 0.5rem 0; }
  li { margin: 0.25rem 0; }
  .arxiv-link { color: var(--accent); text-decoration: none; font-family: monospace; font-size: 0.9em; }
  .arxiv-link:hover { text-decoration: underline; }
  .disagreement { border-left: 3px solid #ef4444; }
  .agreement { border-left: 3px solid #22c55e; }
  .open-q { border-left: 3px solid #eab308; }
  footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); color: var(--muted); font-size: 0.8rem; }
  @media (max-width: 600px) { nav ul { columns: 1; } body { padding: 1rem 0.5rem; } }
</style>
</head>
<body>
<h1>{{ title | e }}</h1>
<p class="meta">{{ paper_count }} papers · Generated by research-pipeline</p>

<nav>
<strong>Contents</strong>
<ul>
{% if agreements %}<li><a href="#agreements">Agreements ({{ agreements | length }})</a></li>{% endif %}
{% if disagreements %}<li><a href="#disagreements">Disagreements ({{ disagreements | length }})</a></li>{% endif %}
{% if open_questions %}<li><a href="#open-questions">Open Questions ({{ open_questions | length }})</a></li>{% endif %}
{% if paper_summaries %}<li><a href="#papers">Paper Summaries ({{ paper_summaries | length }})</a></li>{% endif %}
</ul>
</nav>

{% if agreements %}
<h2 id="agreements">Agreements</h2>
{% for a in agreements %}
<div class="card agreement">
<p><strong>{{ a.claim | e }}</strong>
<span class="badge" style="background:{{ confidence_color(a.claim) }}">{{ confidence_label(a.claim) }}</span>
</p>
<p class="meta">Supported by: {{ a.supporting_papers | map('arxiv_link') | join(', ') }}</p>
</div>
{% endfor %}
{% endif %}

{% if disagreements %}
<h2 id="disagreements">Disagreements</h2>
{% for d in disagreements %}
<div class="card disagreement">
<h3>{{ d.topic | e }}</h3>
<ul>
{% for pid, pos in d.positions.items() %}
<li><strong>{{ arxiv_link(pid) }}</strong>: {{ pos | e }}</li>
{% endfor %}
</ul>
</div>
{% endfor %}
{% endif %}

{% if open_questions %}
<h2 id="open-questions">Open Questions</h2>
{% for q in open_questions %}
<div class="card open-q">
<p>{{ q | e }}</p>
</div>
{% endfor %}
{% endif %}

{% if paper_summaries %}
<h2 id="papers">Paper Summaries</h2>
{% for p in paper_summaries %}
<div class="card" id="paper-{{ p.arxiv_id | replace('.', '-') }}">
<h3>{{ arxiv_link(p.arxiv_id) }} — {{ p.title | e }}</h3>
<p><strong>Objective:</strong> {{ p.objective | e }}</p>
<p><strong>Methodology:</strong> {{ p.methodology | e }}</p>
{% if p.findings %}
<p><strong>Key Findings:</strong></p>
<ul>{% for f in p.findings %}<li>{{ f | e }}</li>{% endfor %}</ul>
{% endif %}
{% if p.limitations %}
<p><strong>Limitations:</strong></p>
<ul>{% for l in p.limitations %}<li>{{ l | e }}</li>{% endfor %}</ul>
{% endif %}
</div>
{% endfor %}
{% endif %}

<footer>
Generated by <a href="https://pypi.org/project/research-pipeline/">research-pipeline</a>
</footer>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_html_report(
    report: SynthesisReport,
    output_path: Path | None = None,
) -> str:
    """Render a SynthesisReport as a self-contained HTML document.

    Args:
        report: The synthesis report to render.
        output_path: If provided, write HTML to this file.

    Returns:
        HTML string.
    """

    def _arxiv_link(arxiv_id: str) -> Markup:
        safe_id = html.escape(arxiv_id)
        return Markup(  # nosec B704
            f'<a href="https://arxiv.org/abs/{safe_id}" '
            f'target="_blank" class="arxiv-link">{safe_id}</a>'
        )

    def _conf_color(text: str) -> str:
        level = _detect_confidence(text)
        return CONFIDENCE_COLORS.get(level, CONFIDENCE_COLORS["unknown"])

    def _conf_label(text: str) -> str:
        level = _detect_confidence(text)
        return CONFIDENCE_LABELS.get(level, CONFIDENCE_LABELS["unknown"])

    env = Environment(loader=BaseLoader(), autoescape=True)
    env.filters["arxiv_link"] = _arxiv_link
    env.globals["arxiv_link"] = _arxiv_link
    env.globals["confidence_color"] = _conf_color
    env.globals["confidence_label"] = _conf_label

    template = env.from_string(_HTML_TEMPLATE)

    html_str = template.render(
        title=report.topic,
        paper_count=report.paper_count,
        agreements=[a.model_dump() for a in report.agreements],
        disagreements=[d.model_dump() for d in report.disagreements],
        open_questions=report.open_questions,
        paper_summaries=[p.model_dump() for p in report.paper_summaries],
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_str, encoding="utf-8")
        logger.info("HTML report written to %s", output_path)

    return html_str


def render_html_from_markdown(
    markdown_path: Path,
    output_path: Path | None = None,
    title: str = "Research Report",
) -> str:
    """Convert a Markdown synthesis report to HTML.

    Uses a simpler approach: wraps Markdown content in styled HTML.
    For full structured rendering, use render_html_report() with
    a SynthesisReport object.

    Args:
        markdown_path: Path to Markdown report file.
        output_path: If provided, write HTML to this file.
        title: Report title for the HTML page.

    Returns:
        HTML string.
    """
    md_text = markdown_path.read_text(encoding="utf-8")

    # Convert basic Markdown to HTML
    html_body = _markdown_to_html(md_text)
    html_body = _linkify_arxiv_ids(html_body)

    html_str = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} — Research Report</title>
<style>
  :root {{
    --bg: #ffffff; --fg: #1e293b; --muted: #64748b;
    --border: #e2e8f0; --accent: #3b82f6; --surface: #f8fafc;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #0f172a; --fg: #e2e8f0; --muted: #94a3b8;
      --border: #334155; --accent: #60a5fa; --surface: #1e293b;
    }}
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--fg);
    line-height: 1.6; max-width: 960px; margin: 0 auto; padding: 2rem 1rem;
  }}
  h1 {{ font-size: 1.75rem; margin: 1.5rem 0 0.5rem; }}
  h2 {{ font-size: 1.35rem; margin: 2rem 0 0.75rem; border-bottom: 2px solid var(--border); padding-bottom: 0.25rem; }}
  h3 {{ font-size: 1.1rem; margin: 1.5rem 0 0.5rem; }}
  p {{ margin: 0.5rem 0; }}
  ul, ol {{ padding-left: 1.5rem; margin: 0.5rem 0; }}
  li {{ margin: 0.25rem 0; }}
  .arxiv-link {{ color: var(--accent); text-decoration: none; font-family: monospace; }}
  .arxiv-link:hover {{ text-decoration: underline; }}
  code {{ background: var(--surface); padding: 0.15rem 0.3rem; border-radius: 3px; font-size: 0.9em; }}
  pre {{ background: var(--surface); padding: 1rem; border-radius: 8px; overflow-x: auto; margin: 1rem 0; }}
  blockquote {{ border-left: 3px solid var(--accent); padding-left: 1rem; color: var(--muted); margin: 0.75rem 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid var(--border); padding: 0.5rem; text-align: left; }}
  th {{ background: var(--surface); }}
  footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); color: var(--muted); font-size: 0.8rem; }}
</style>
</head>
<body>
{html_body}
<footer>
Generated by <a href="https://pypi.org/project/research-pipeline/">research-pipeline</a>
</footer>
</body>
</html>"""

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_str, encoding="utf-8")
        logger.info("HTML report written to %s", output_path)

    return html_str


def _markdown_to_html(md: str) -> str:
    """Convert basic Markdown to HTML without external dependencies.

    Handles: headings, bold, italic, code blocks, inline code,
    unordered lists, ordered lists, blockquotes, horizontal rules,
    tables, and paragraphs.

    Args:
        md: Markdown text.

    Returns:
        HTML string.
    """
    lines = md.split("\n")
    html_lines: list[str] = []
    in_code_block = False
    in_list = False
    in_ol = False
    in_table = False

    for line in lines:
        # Code blocks
        if line.strip().startswith("```"):
            if in_code_block:
                html_lines.append("</code></pre>")
                in_code_block = False
            else:
                lang = line.strip()[3:].strip()
                html_lines.append(
                    f'<pre><code class="language-{html.escape(lang)}">'
                    if lang
                    else "<pre><code>"
                )
                in_code_block = True
            continue

        if in_code_block:
            html_lines.append(html.escape(line))
            continue

        # Close open lists if the line is not a list item
        stripped = line.strip()
        if in_list and not stripped.startswith(("- ", "* ", "+ ")):
            html_lines.append("</ul>")
            in_list = False
        if in_ol and not re.match(r"^\d+\.\s", stripped):
            html_lines.append("</ol>")
            in_ol = False
        if in_table and not stripped.startswith("|"):
            html_lines.append("</table>")
            in_table = False

        # Blank line
        if not stripped:
            html_lines.append("")
            continue

        # Headings
        heading_match = re.match(r"^(#{1,6})\s+(.+)", line)
        if heading_match:
            level = len(heading_match.group(1))
            text = _inline_format(heading_match.group(2))
            slug = re.sub(r"[^a-z0-9]+", "-", heading_match.group(2).lower()).strip("-")
            html_lines.append(f'<h{level} id="{slug}">{text}</h{level}>')
            continue

        # Horizontal rule
        if re.match(r"^[-*_]{3,}\s*$", stripped):
            html_lines.append("<hr>")
            continue

        # Blockquote
        if stripped.startswith("> "):
            html_lines.append(
                f"<blockquote><p>{_inline_format(stripped[2:])}</p></blockquote>"
            )
            continue

        # Unordered list
        if stripped.startswith(("- ", "* ", "+ ")):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            item_text = _inline_format(stripped[2:])
            html_lines.append(f"<li>{item_text}</li>")
            continue

        # Ordered list
        ol_match = re.match(r"^(\d+)\.\s+(.+)", stripped)
        if ol_match:
            if not in_ol:
                html_lines.append("<ol>")
                in_ol = True
            item_text = _inline_format(ol_match.group(2))
            html_lines.append(f"<li>{item_text}</li>")
            continue

        # Table
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if all(re.match(r"^[-:]+$", c) for c in cells):
                continue  # separator row
            if not in_table:
                html_lines.append("<table>")
                in_table = True
                tag = "th"
            else:
                tag = "td"
            row = "".join(f"<{tag}>{_inline_format(c)}</{tag}>" for c in cells)
            html_lines.append(f"<tr>{row}</tr>")
            continue

        # Paragraph
        html_lines.append(f"<p>{_inline_format(stripped)}</p>")

    # Close any open blocks
    if in_list:
        html_lines.append("</ul>")
    if in_ol:
        html_lines.append("</ol>")
    if in_table:
        html_lines.append("</table>")
    if in_code_block:
        html_lines.append("</code></pre>")

    return "\n".join(html_lines)


def _inline_format(text: str) -> str:
    """Apply inline Markdown formatting (bold, italic, code, links).

    Args:
        text: Inline text to format.

    Returns:
        HTML-formatted string.
    """
    # Escape HTML first
    text = html.escape(text)
    # Bold + italic
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", text)
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Italic
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    # Inline code
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    # Links [text](url)
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r'<a href="\2" target="_blank">\1</a>',
        text,
    )
    return text
