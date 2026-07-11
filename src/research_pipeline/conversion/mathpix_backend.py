"""Mathpix cloud OCR backend for PDF-to-Markdown conversion.

Uses the Mathpix REST API (https://docs.mathpix.com) for best-in-class
LaTeX equation extraction. Requires ``app_id`` and ``app_key`` from
https://console.mathpix.com.
Free tier: 1,000 pages/month.

Keywords: multi-account, account rotation, quota rotation.
"""

from __future__ import annotations

import logging
import time

import requests

from research_pipeline.conversion.base import (
    PROGRAMMING_ERRORS,
    ConversionContext,
    ConverterBackend,
)
from research_pipeline.conversion.registry import register_backend

logger = logging.getLogger(__name__)

_API_BASE = "https://api.mathpix.com/v3"
_POLL_INTERVAL = 3  # seconds
_POLL_TIMEOUT = 600  # 10 minutes max


@register_backend("mathpix")
class MathpixBackend(ConverterBackend):
    """Mathpix cloud PDF-to-Markdown converter.

    Best-in-class LaTeX/equation extraction.
    ~1-3 s per page, ~$0.01/page (1K free/month).
    """

    def __init__(
        self,
        *,
        app_id: str = "",
        app_key: str = "",
    ) -> None:
        if not app_id or not app_key:
            raise ValueError(
                "Mathpix backend requires 'app_id' and 'app_key'. "
                "Set RESEARCH_PIPELINE_MATHPIX_APP_ID and "
                "RESEARCH_PIPELINE_MATHPIX_APP_KEY, or configure in config.toml."
            )
        self.app_id = app_id
        self.app_key = app_key

    @property
    def converter_name(self) -> str:
        return "mathpix"

    @property
    def converter_version(self) -> str:
        return "cloud"

    def _config_string(self) -> str:
        return f"app_id={self.app_id}"

    def _headers(self) -> dict[str, str]:
        return {
            "app_id": self.app_id,
            "app_key": self.app_key,
        }

    def _run(self, ctx: ConversionContext) -> tuple[str, list[str]]:
        # 1. Upload PDF
        logger.info("Uploading %s to Mathpix API...", ctx.pdf_path.name)
        with ctx.pdf_path.open("rb") as f:
            resp = requests.post(
                f"{_API_BASE}/pdf",
                headers=self._headers(),
                files={"file": (ctx.pdf_path.name, f, "application/pdf")},
                data={"conversion_formats": '{"md": true}'},
                timeout=60,
            )
        resp.raise_for_status()
        pdf_id = resp.json()["pdf_id"]
        logger.info("Mathpix PDF ID: %s", pdf_id)

        # 2. Poll until complete
        elapsed = 0.0
        while elapsed < _POLL_TIMEOUT:
            time.sleep(_POLL_INTERVAL)
            elapsed += _POLL_INTERVAL
            status_resp = requests.get(
                f"{_API_BASE}/pdf/{pdf_id}",
                headers=self._headers(),
                timeout=30,
            )
            status_resp.raise_for_status()
            status_data = status_resp.json()
            status = status_data.get("status", "")
            pct = status_data.get("percent_done", 0)
            logger.debug("Mathpix status: %s (%.0f%%)", status, pct)
            if status == "completed":
                break
            if status == "error":
                raise RuntimeError(f"Mathpix processing error: {status_data}")
        else:
            raise TimeoutError(f"Mathpix processing timed out after {_POLL_TIMEOUT}s")

        # 3. Download Markdown
        md_resp = requests.get(
            f"{_API_BASE}/pdf/{pdf_id}.md",
            headers=self._headers(),
            timeout=60,
        )
        md_resp.raise_for_status()
        markdown_text = md_resp.text

        ctx.md_path.write_text(markdown_text, encoding="utf-8")
        logger.info(
            "Converted %s → %s via Mathpix", ctx.pdf_path.name, ctx.md_path.name
        )

        # 4. Clean up remote
        try:
            requests.delete(
                f"{_API_BASE}/pdf/{pdf_id}",
                headers=self._headers(),
                timeout=10,
            )
        except Exception as exc:
            if isinstance(exc, PROGRAMMING_ERRORS):
                raise
            logger.debug("Failed to delete Mathpix PDF %s: %s", pdf_id, exc)

        return markdown_text, []
