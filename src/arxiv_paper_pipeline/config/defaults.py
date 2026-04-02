"""Default configuration values."""

DEFAULTS: dict[str, dict[str, object]] = {
    "arxiv": {
        "base_url": "https://export.arxiv.org/api/query",
        "min_interval_seconds": 5.0,
        "single_connection": True,
        "default_page_size": 100,
        "max_page_size": 500,
        "daily_query_cache": True,
    },
    "search": {
        "primary_months": 6,
        "fallback_months": 12,
        "max_query_variants": 5,
        "min_candidates": 40,
        "min_highscore": 10,
        "min_downloads": 5,
    },
    "screen": {
        "cheap_top_k": 50,
        "download_top_n": 8,
        "final_score_threshold": 0.70,
        "llm_score_threshold": 0.60,
    },
    "download": {
        "max_per_run": 20,
    },
    "conversion": {
        "backend": "docling",
        "timeout_seconds": 300,
    },
    "llm": {
        "enabled": False,
        "temperature": 0,
        "profile": "default",
    },
    "cache": {
        "enabled": True,
        "search_snapshot_ttl_hours": 24,
        "cache_dir": "~/.cache/arxiv-paper-pipeline",
    },
}
