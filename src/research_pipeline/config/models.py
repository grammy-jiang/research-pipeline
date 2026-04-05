"""Pydantic configuration schemas."""

from pydantic import BaseModel, Field

from research_pipeline.config.defaults import DEFAULTS


class ArxivConfig(BaseModel):
    """arXiv API access configuration."""

    base_url: str = str(DEFAULTS["arxiv"]["base_url"])
    min_interval_seconds: float = float(DEFAULTS["arxiv"]["min_interval_seconds"])
    single_connection: bool = bool(DEFAULTS["arxiv"]["single_connection"])
    default_page_size: int = int(DEFAULTS["arxiv"]["default_page_size"])
    max_page_size: int = int(DEFAULTS["arxiv"]["max_page_size"])
    daily_query_cache: bool = bool(DEFAULTS["arxiv"]["daily_query_cache"])
    request_timeout_seconds: int = int(DEFAULTS["arxiv"]["request_timeout_seconds"])


class SearchConfig(BaseModel):
    """Search parameters."""

    primary_months: int = int(DEFAULTS["search"]["primary_months"])
    fallback_months: int = int(DEFAULTS["search"]["fallback_months"])
    max_query_variants: int = int(DEFAULTS["search"]["max_query_variants"])
    min_candidates: int = int(DEFAULTS["search"]["min_candidates"])
    min_highscore: int = int(DEFAULTS["search"]["min_highscore"])
    min_downloads: int = int(DEFAULTS["search"]["min_downloads"])


class ScreenConfig(BaseModel):
    """Screening parameters."""

    cheap_top_k: int = int(DEFAULTS["screen"]["cheap_top_k"])
    download_top_n: int = int(DEFAULTS["screen"]["download_top_n"])
    final_score_threshold: float = float(DEFAULTS["screen"]["final_score_threshold"])
    llm_score_threshold: float = float(DEFAULTS["screen"]["llm_score_threshold"])
    use_semantic_reranking: bool = False
    embedding_model: str = "allenai/specter2"
    embedding_batch_size: int = 32


class DownloadConfig(BaseModel):
    """Download parameters."""

    max_per_run: int = int(DEFAULTS["download"]["max_per_run"])


class MarkerConfig(BaseModel):
    """Marker-specific conversion parameters."""

    force_ocr: bool = False
    use_llm: bool = False
    llm_service: str = ""
    llm_api_key: str = ""


class MathpixAccount(BaseModel):
    """Single Mathpix account credentials."""

    app_id: str
    app_key: str


class MathpixConfig(BaseModel):
    """Mathpix cloud OCR parameters."""

    app_id: str = ""
    app_key: str = ""
    accounts: list[MathpixAccount] = Field(default_factory=list)


class DatalabAccount(BaseModel):
    """Single Datalab account credentials."""

    api_key: str
    mode: str = "balanced"


class DatalabConfig(BaseModel):
    """Datalab (hosted Marker) cloud parameters."""

    api_key: str = ""
    mode: str = "balanced"
    accounts: list[DatalabAccount] = Field(default_factory=list)


class LlamaParseAccount(BaseModel):
    """Single LlamaParse account credentials."""

    api_key: str
    tier: str = "agentic"


class LlamaParseConfig(BaseModel):
    """LlamaParse cloud parameters."""

    api_key: str = ""
    tier: str = "agentic"
    accounts: list[LlamaParseAccount] = Field(default_factory=list)


class MistralOcrAccount(BaseModel):
    """Single Mistral OCR account credentials."""

    api_key: str
    model: str = "mistral-ocr-latest"


class MistralOcrConfig(BaseModel):
    """Mistral OCR cloud parameters."""

    api_key: str = ""
    model: str = "mistral-ocr-latest"
    accounts: list[MistralOcrAccount] = Field(default_factory=list)


class OpenAIVisionAccount(BaseModel):
    """Single OpenAI Vision account credentials."""

    api_key: str
    model: str = "gpt-4o"


class OpenAIVisionConfig(BaseModel):
    """OpenAI GPT-4o vision parameters."""

    api_key: str = ""
    model: str = "gpt-4o"
    accounts: list[OpenAIVisionAccount] = Field(default_factory=list)


class ConversionConfig(BaseModel):
    """Conversion parameters."""

    backend: str = str(DEFAULTS["conversion"]["backend"])
    fallback_backends: list[str] = Field(default_factory=list)
    timeout_seconds: int = int(DEFAULTS["conversion"]["timeout_seconds"])
    marker: MarkerConfig = Field(default_factory=MarkerConfig)
    mathpix: MathpixConfig = Field(default_factory=MathpixConfig)
    datalab: DatalabConfig = Field(default_factory=DatalabConfig)
    llamaparse: LlamaParseConfig = Field(default_factory=LlamaParseConfig)
    mistral_ocr: MistralOcrConfig = Field(default_factory=MistralOcrConfig)
    openai_vision: OpenAIVisionConfig = Field(default_factory=OpenAIVisionConfig)


class LLMConfig(BaseModel):
    """LLM integration configuration."""

    enabled: bool = bool(DEFAULTS["llm"]["enabled"])
    temperature: float = float(DEFAULTS["llm"]["temperature"])
    profile: str = str(DEFAULTS["llm"]["profile"])


class CacheConfig(BaseModel):
    """Cache configuration."""

    enabled: bool = bool(DEFAULTS["cache"]["enabled"])
    search_snapshot_ttl_hours: int = int(DEFAULTS["cache"]["search_snapshot_ttl_hours"])
    cache_dir: str = str(DEFAULTS["cache"]["cache_dir"])


class SourcesConfig(BaseModel):
    """Multi-source search configuration."""

    enabled: list[str] = list(DEFAULTS["sources"]["enabled"])  # type: ignore[arg-type]
    scholar_backend: str = str(DEFAULTS["sources"]["scholar_backend"])
    scholar_min_interval: float = float(DEFAULTS["sources"]["scholar_min_interval"])
    serpapi_key: str = str(DEFAULTS["sources"]["serpapi_key"])
    serpapi_min_interval: float = float(DEFAULTS["sources"]["serpapi_min_interval"])

    # Semantic Scholar
    semantic_scholar_enabled: bool = False
    semantic_scholar_api_key: str = ""
    semantic_scholar_min_interval: float = 1.0

    # OpenAlex
    openalex_enabled: bool = False
    openalex_api_key: str = ""
    openalex_min_interval: float = 0.2

    # DBLP
    dblp_enabled: bool = False
    dblp_min_interval: float = 2.0


class QualityConfig(BaseModel):
    """Quality evaluation parameters."""

    enabled: bool = False
    citation_weight: float = 0.35
    venue_weight: float = 0.25
    author_weight: float = 0.25
    recency_weight: float = 0.15
    venue_data_path: str = ""
    author_cache_ttl_hours: float = 168.0
    min_quality_score: float = 0.0


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    arxiv: ArxivConfig = Field(default_factory=ArxivConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    screen: ScreenConfig = Field(default_factory=ScreenConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    conversion: ConversionConfig = Field(default_factory=ConversionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    workspace: str = "runs"
    contact_email: str = ""
