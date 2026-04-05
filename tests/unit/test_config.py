"""Unit tests for config.loader module."""

from pathlib import Path

from research_pipeline.config.loader import load_config
from research_pipeline.config.models import PipelineConfig


class TestLoadConfig:
    def test_defaults_without_file(self, tmp_path: Path, monkeypatch: object) -> None:
        monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
        # Ensure no env var override
        monkeypatch.delenv("RESEARCH_PIPELINE_CONFIG", raising=False)  # type: ignore[attr-defined]
        config = load_config()
        assert isinstance(config, PipelineConfig)
        assert config.arxiv.min_interval_seconds == 5.0
        assert config.arxiv.base_url == "https://export.arxiv.org/api/query"

    def test_load_from_toml(self, tmp_path: Path, monkeypatch: object) -> None:
        monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
        toml_content = """\
[arxiv]
min_interval_seconds = 7.0

[download]
max_per_run = 5
"""
        toml_path = tmp_path / "custom.toml"
        toml_path.write_text(toml_content, encoding="utf-8")
        config = load_config(config_path=toml_path)
        assert config.arxiv.min_interval_seconds == 7.0
        assert config.download.max_per_run == 5

    def test_auto_detect_config_toml(self, tmp_path: Path, monkeypatch: object) -> None:
        monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
        monkeypatch.delenv("RESEARCH_PIPELINE_CONFIG", raising=False)  # type: ignore[attr-defined]
        toml_content = """\
[download]
max_per_run = 3
"""
        (tmp_path / "config.toml").write_text(toml_content, encoding="utf-8")
        config = load_config()
        assert config.download.max_per_run == 3

    def test_env_var_override_workspace(
        self, tmp_path: Path, monkeypatch: object
    ) -> None:
        monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
        monkeypatch.setenv("RESEARCH_PIPELINE_WORKSPACE", "/tmp/custom_ws")  # type: ignore[attr-defined]
        config = load_config()
        assert config.workspace == "/tmp/custom_ws"
        monkeypatch.delenv("RESEARCH_PIPELINE_WORKSPACE")  # type: ignore[attr-defined]

    def test_env_var_disable_llm(self, tmp_path: Path, monkeypatch: object) -> None:
        monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
        monkeypatch.setenv("RESEARCH_PIPELINE_DISABLE_LLM", "1")  # type: ignore[attr-defined]
        config = load_config()
        assert config.llm.enabled is False
        monkeypatch.delenv("RESEARCH_PIPELINE_DISABLE_LLM")  # type: ignore[attr-defined]

    def test_env_var_config_path(self, tmp_path: Path, monkeypatch: object) -> None:
        monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
        toml_content = """\
[download]
max_per_run = 99
"""
        toml_path = tmp_path / "env-config.toml"
        toml_path.write_text(toml_content, encoding="utf-8")
        monkeypatch.setenv("RESEARCH_PIPELINE_CONFIG", str(toml_path))  # type: ignore[attr-defined]
        config = load_config()
        assert config.download.max_per_run == 99
        monkeypatch.delenv("RESEARCH_PIPELINE_CONFIG")  # type: ignore[attr-defined]


class TestPipelineConfigDefaults:
    def test_all_sections_present(self) -> None:
        config = PipelineConfig()
        assert config.arxiv is not None
        assert config.search is not None
        assert config.screen is not None
        assert config.download is not None
        assert config.conversion is not None
        assert config.llm is not None
        assert config.cache is not None

    def test_arxiv_hard_floor_respected(self) -> None:
        config = PipelineConfig()
        assert config.arxiv.min_interval_seconds >= 3.0

    def test_default_workspace(self) -> None:
        config = PipelineConfig()
        assert config.workspace == "runs"

    def test_conversion_has_marker_config(self) -> None:
        config = PipelineConfig()
        assert config.conversion.marker is not None
        assert config.conversion.marker.force_ocr is False
        assert config.conversion.marker.use_llm is False
        assert config.conversion.marker.llm_service == ""
        assert config.conversion.marker.llm_api_key == ""

    def test_conversion_default_backend(self) -> None:
        config = PipelineConfig()
        assert config.conversion.backend == "docling"

    def test_marker_config_from_toml(self, tmp_path: Path, monkeypatch: object) -> None:
        monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
        toml_content = """\
[conversion]
backend = "marker"

[conversion.marker]
force_ocr = true
use_llm = true
llm_service = "gemini"
"""
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(toml_content, encoding="utf-8")
        config = load_config(config_path=toml_path)
        assert config.conversion.backend == "marker"
        assert config.conversion.marker.force_ocr is True
        assert config.conversion.marker.use_llm is True
        assert config.conversion.marker.llm_service == "gemini"
