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
