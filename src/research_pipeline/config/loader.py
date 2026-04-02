"""TOML and environment-variable configuration loader."""

import logging
import os
import tomllib
from pathlib import Path

from research_pipeline.config.models import PipelineConfig

logger = logging.getLogger(__name__)

_ENV_PREFIX = "RESEARCH_PIPELINE_"


def _load_toml(path: Path) -> dict:  # type: ignore[type-arg]
    """Load a TOML file.

    Args:
        path: Path to the TOML file.

    Returns:
        Parsed TOML as a dictionary.
    """
    with path.open("rb") as f:
        return tomllib.load(f)


def _apply_env_overrides(data: dict) -> dict:  # type: ignore[type-arg]
    """Apply environment variable overrides to config data.

    Environment variables follow the pattern:
    ``RESEARCH_PIPELINE_<SECTION>_<KEY>`` (uppercase, underscore-separated).

    Known overrides:
    - RESEARCH_PIPELINE_CONFIG: config file path (handled upstream)
    - RESEARCH_PIPELINE_CACHE_DIR: override cache directory
    - RESEARCH_PIPELINE_WORKSPACE: override workspace directory
    - RESEARCH_PIPELINE_ALLOW_LIVE: enable live arXiv access
    - RESEARCH_PIPELINE_DISABLE_LLM: force LLM off
    - RESEARCH_PIPELINE_LLM_PROFILE: LLM model selection
    """
    env_map = {
        f"{_ENV_PREFIX}CACHE_DIR": ("cache", "cache_dir"),
        f"{_ENV_PREFIX}WORKSPACE": (None, "workspace"),
        f"{_ENV_PREFIX}DISABLE_LLM": ("llm", "enabled"),
        f"{_ENV_PREFIX}LLM_PROFILE": ("llm", "profile"),
    }

    for env_key, (section, key) in env_map.items():
        value = os.environ.get(env_key)
        if value is None:
            continue

        if section is not None:
            if section not in data:
                data[section] = {}
            if env_key.endswith("DISABLE_LLM"):
                data[section][key] = value.lower() not in ("1", "true", "yes")
            else:
                data[section][key] = value
        else:
            data[key] = value

        logger.info("Config override from env: %s", env_key)

    return data


def load_config(config_path: Path | None = None) -> PipelineConfig:
    """Load pipeline configuration from TOML file and environment.

    Precedence: environment variables > TOML file > defaults.

    Args:
        config_path: Path to a TOML config file. If None, checks
            ``RESEARCH_PIPELINE_CONFIG`` env var, then ``config.toml``
            in the current directory.

    Returns:
        Validated PipelineConfig.
    """
    data: dict = {}  # type: ignore[type-arg]

    if config_path is None:
        env_path = os.environ.get(f"{_ENV_PREFIX}CONFIG")
        if env_path:
            config_path = Path(env_path)
        elif Path("config.toml").exists():
            config_path = Path("config.toml")

    if config_path is not None and config_path.exists():
        logger.info("Loading config from %s", config_path)
        data = _load_toml(config_path)
    else:
        logger.info("No config file found; using defaults")

    data = _apply_env_overrides(data)
    config = PipelineConfig.model_validate(data)
    logger.info("Configuration loaded successfully")
    return config
