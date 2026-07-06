"""Range sanity checks for extracted statistics (#33).

Sub-agent extraction of numeric statistics is exactly where hallucinated or
garbled values leak into final reports — e.g. a recorded Spearman rho of
-1.8 (impossible: rho in [-1, 1]) surviving verbatim into a draft. This module
scans analyzer/synthesis JSON for recognizable statistics and flags values that
violate their hard mathematical bounds, so report generation can footnote or
drop them instead of presenting them as fact.

Only *hard* violations are flagged to avoid false positives on metrics that
legitimately use more than one scale (e.g. accuracy as 0-1 or 0-100).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

# (key-substring pattern, low, high, human label). Conservative: only stats
# with unambiguous hard bounds.
_RULES: list[tuple[re.Pattern[str], float, float, str]] = [
    (
        re.compile(r"spearman|pearson|kendall|\brho\b|correlation|corr_coef"),
        -1.0,
        1.0,
        "correlation coefficient",
    ),
    (
        re.compile(r"probability|\bprob\b|p_?values?|likelihood"),
        0.0,
        1.0,
        "probability",
    ),
    (re.compile(r"\bbleu\b|\bchrf\b|\bmeteor\b|\bter\b"), 0.0, 100.0, "MT metric"),
    (re.compile(r"percent|\bpct\b|percentage"), 0.0, 100.0, "percentage"),
]


def _match_rule(key: str) -> tuple[float, float, str] | None:
    lowered = key.lower()
    for pattern, low, high, label in _RULES:
        if pattern.search(lowered):
            return low, high, label
    return None


def _numeric_values(value: object) -> list[float]:
    """Numeric scalars directly under a key (a scalar or a flat list)."""
    if isinstance(value, bool):  # bool is an int subclass; never a statistic
        return []
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [
            float(v)
            for v in value
            if isinstance(v, int | float) and not isinstance(v, bool)
        ]
    return []


def find_stat_anomalies(obj: object, path: str = "") -> list[str]:
    """Return human-readable messages for out-of-range statistics in *obj*.

    Recursively walks mappings/sequences; a numeric value under a key that
    names a bounded statistic is flagged when it falls outside those bounds.
    """
    anomalies: list[str] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            key_path = f"{path}.{key}" if path else str(key)
            rule = _match_rule(str(key))
            if rule is not None:
                low, high, label = rule
                for number in _numeric_values(value):
                    if not low <= number <= high:
                        anomalies.append(
                            f"{key_path}={number} is outside [{low}, {high}] "
                            f"for a {label}"
                        )
            anomalies.extend(find_stat_anomalies(value, key_path))
    elif isinstance(obj, Sequence) and not isinstance(obj, str | bytes):
        for index, item in enumerate(obj):
            anomalies.extend(find_stat_anomalies(item, f"{path}[{index}]"))
    return anomalies
