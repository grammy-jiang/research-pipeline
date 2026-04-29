"""Unit tests for the X (Twitter) policy stub."""

from __future__ import annotations

import pytest

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingSourceConfig,
    SourceClass,
)
from research_pipeline.briefing.sources.x_api import XApiPolicyError, XApiSource


def _source(**overrides: object) -> BriefingSourceConfig:
    base: dict[str, object] = {
        "source_id": "x-ai",
        "source_name": "X AI list",
        "source_class": SourceClass.SOCIAL_SIGNAL,
        "access_method": AccessMethod.X_API,
        "enabled": False,
        "auth_required": False,
        "tags": (),
    }
    base.update(overrides)
    return BriefingSourceConfig(**base)  # type: ignore[arg-type]


class TestXApiPolicyStub:
    def test_disabled_by_default_refuses(self) -> None:
        with pytest.raises(XApiPolicyError, match="disabled"):
            XApiSource(_source()).poll()

    def test_missing_auth_required_refuses(self) -> None:
        cfg = _source(
            enabled=True,
            auth_required=False,
            last_reviewed_at="2026-04-30",
            tags=("policy_gate_passed",),
        )
        with pytest.raises(XApiPolicyError, match="auth_required"):
            XApiSource(cfg).poll()

    def test_missing_review_refuses(self) -> None:
        cfg = _source(
            enabled=True,
            auth_required=True,
            last_reviewed_at=None,
            tags=("policy_gate_passed",),
        )
        with pytest.raises(XApiPolicyError, match="last_reviewed_at"):
            XApiSource(cfg).poll()

    def test_missing_policy_tag_refuses(self) -> None:
        cfg = _source(
            enabled=True,
            auth_required=True,
            last_reviewed_at="2026-04-30",
            tags=(),
        )
        with pytest.raises(XApiPolicyError, match="policy_gate_passed"):
            XApiSource(cfg).poll()

    def test_all_gates_pass_returns_empty(self) -> None:
        cfg = _source(
            enabled=True,
            auth_required=True,
            last_reviewed_at="2026-04-30",
            tags=("policy_gate_passed",),
        )
        assert XApiSource(cfg).poll() == []

    def test_wrong_access_method_rejected(self) -> None:
        bad = _source().model_copy(
            update={
                "access_method": AccessMethod.RSS_ATOM,
                "feed_url": "https://example.com/feed.xml",
            }
        )
        with pytest.raises(ValueError, match="X_API"):
            XApiSource(bad)
