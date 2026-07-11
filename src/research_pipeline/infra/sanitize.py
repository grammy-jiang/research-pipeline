"""Content sanitization for untrusted paper text.

Strips potential prompt injection patterns from paper abstracts, titles,
and other user-provided content before it reaches LLM prompts or MCP
tool descriptions.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Patterns that could be prompt injection attempts
_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # XML/HTML-like tool/system tags
    (
        re.compile(
            r"</?(?:tool_call|tool_result|system|assistant|user|function_call|invoke)[^>]*>",
            re.IGNORECASE,
        ),
        "",
    ),
    # System/instruction markers
    (
        re.compile(
            r"(?:^|\n)\s*(?:SYSTEM|INSTRUCTION|ADMIN|OVERRIDE)\s*:",
            re.IGNORECASE,
        ),
        "\n[FILTERED]:",
    ),
    # Jinja/template injection
    (re.compile(r"\{\{.*?\}\}", re.DOTALL), "[TEMPLATE_FILTERED]"),
    (re.compile(r"\{%.*?%\}", re.DOTALL), "[TEMPLATE_FILTERED]"),
    # Markdown code fences that might contain executable payloads
    (
        re.compile(r"```(?:python|javascript|bash|sh|shell|exec)\b", re.IGNORECASE),
        "```text",
    ),
    # Consecutive backticks (trying to escape code blocks)
    (re.compile(r"`{4,}"), "```"),
    # Data URI schemes
    (re.compile(r"data:\w+/\w+;base64,", re.IGNORECASE), "[DATA_URI_FILTERED]"),
    # JSON-like function calls
    (
        re.compile(r'"function_call"\s*:\s*\{', re.IGNORECASE),
        '"[FILTERED]": {',
    ),
]

# Maximum allowed length for sanitized content (chars)
_MAX_CONTENT_LENGTH = 50_000

# Credential-shaped substrings that must never reach a log line, a ToolResult
# error, or any stored artefact (#125, HC6). Two classes: a value that follows a
# secret label (``api_key=…``), and self-identifying token prefixes.
_LABELED_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|secret|token|password|passwd|access[_-]?key)"
    r"(\"?\s*[:=]\s*\"?)"
    r"([^\s\"'&]{6,})"
)
_PREFIXED_SECRET_RE = re.compile(
    r"\b("
    r"sk-[A-Za-z0-9]{16,}"  # OpenAI-style
    r"|gh[posru]_[A-Za-z0-9]{20,}"  # GitHub tokens
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"  # Slack
    r"|AKIA[0-9A-Z]{16}"  # AWS access key id
    r")\b"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{8,}")


def redact_secrets(text: str) -> str:
    """Redact likely credentials (API keys, tokens, bearer) from *text* (#125).

    Conservative by design: it targets labeled ``key=value`` secrets and
    self-identifying token prefixes, not arbitrary high-entropy strings, so it
    does not mangle ordinary log messages. Returns *text* with each match
    replaced by ``[REDACTED]``.
    """
    if not text:
        return text
    result = _LABELED_SECRET_RE.sub(r"\1\2[REDACTED]", text)
    result = _PREFIXED_SECRET_RE.sub("[REDACTED]", result)
    result = _BEARER_RE.sub("Bearer [REDACTED]", result)
    return result


def sanitize_text(text: str, max_length: int = _MAX_CONTENT_LENGTH) -> str:
    """Sanitize untrusted text by removing potential injection patterns.

    Args:
        text: Raw untrusted text (e.g., paper abstract or title).
        max_length: Maximum allowed output length.

    Returns:
        Sanitized text with injection patterns removed.
    """
    if not text:
        return text

    result = text
    patterns_matched = 0
    for pattern, replacement in _INJECTION_PATTERNS:
        new_result = pattern.sub(replacement, result)
        if new_result != result:
            patterns_matched += 1
        result = new_result

    # Truncate if too long
    if len(result) > max_length:
        result = result[:max_length] + "... [TRUNCATED]"
        logger.warning("Content truncated from %d to %d chars", len(text), max_length)

    if patterns_matched > 0:
        logger.warning(
            "Sanitized %d injection pattern(s) from content (len=%d)",
            patterns_matched,
            len(text),
        )

    return result


def sanitize_candidate_fields(
    title: str,
    abstract: str,
    max_title_length: int = 500,
    max_abstract_length: int = 10_000,
) -> tuple[str, str]:
    """Sanitize paper title and abstract together.

    Args:
        title: Paper title.
        abstract: Paper abstract.
        max_title_length: Maximum title length.
        max_abstract_length: Maximum abstract length.

    Returns:
        Tuple of (sanitized_title, sanitized_abstract).
    """
    return (
        sanitize_text(title, max_length=max_title_length),
        sanitize_text(abstract, max_length=max_abstract_length),
    )


def is_suspicious(text: str) -> bool:
    """Check if text contains suspicious injection patterns.

    Does NOT modify the text — use ``sanitize_text()`` for that.

    Args:
        text: Text to check.

    Returns:
        True if any injection pattern is detected.
    """
    if not text:
        return False
    return any(pattern.search(text) for pattern, _ in _INJECTION_PATTERNS)
