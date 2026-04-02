"""Unit tests for infra.http module."""

from research_pipeline.infra.http import create_session


class TestCreateSession:
    def test_returns_session(self) -> None:
        session = create_session()
        assert session is not None

    def test_has_user_agent(self) -> None:
        session = create_session()
        assert "User-Agent" in session.headers

    def test_user_agent_contains_package_name(self) -> None:
        session = create_session()
        ua = session.headers["User-Agent"]
        assert "research-pipeline" in ua
