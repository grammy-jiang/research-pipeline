"""Explicit --config path handling in the config loader (#21)."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_pipeline.config.loader import load_config


def test_explicit_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(tmp_path / "nope.toml")


def test_env_missing_config_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RESEARCH_PIPELINE_CONFIG", str(tmp_path / "nope.toml"))
    with pytest.raises(FileNotFoundError):
        load_config()


def test_no_config_uses_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RESEARCH_PIPELINE_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)  # no ./config.toml here
    config = load_config()
    assert config is not None


def test_explicit_existing_config_loads(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('contact_email = "a@b.c"\n')
    config = load_config(cfg_file)
    assert config.contact_email == "a@b.c"
