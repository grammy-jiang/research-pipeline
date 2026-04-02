"""Unit tests for infra.clock module."""

from datetime import UTC, datetime

from arxiv_paper_pipeline.infra.clock import date_window, format_arxiv_date, utc_now


class TestUtcNow:
    def test_returns_utc_aware_datetime(self) -> None:
        result = utc_now()
        assert result.tzinfo is not None
        assert result.tzinfo == UTC

    def test_is_roughly_current(self) -> None:
        before = datetime.now(UTC)
        result = utc_now()
        after = datetime.now(UTC)
        assert before <= result <= after


class TestFormatArxivDate:
    def test_basic_formatting(self) -> None:
        dt = datetime(2024, 1, 15, 9, 30, tzinfo=UTC)
        assert format_arxiv_date(dt) == "202401150930"

    def test_midnight(self) -> None:
        dt = datetime(2024, 6, 1, 0, 0, tzinfo=UTC)
        assert format_arxiv_date(dt) == "202406010000"

    def test_end_of_day(self) -> None:
        dt = datetime(2024, 12, 31, 23, 59, tzinfo=UTC)
        assert format_arxiv_date(dt) == "202412312359"


class TestDateWindow:
    def test_basic_window(self) -> None:
        ref = datetime(2024, 7, 1, 12, 0, tzinfo=UTC)
        from_date, to_date = date_window(6, reference=ref)
        assert to_date == "202407011200"
        # 6 months * 30 days = 180 days back
        expected_start = datetime(2024, 1, 3, 12, 0, tzinfo=UTC)
        assert from_date == format_arxiv_date(expected_start)

    def test_zero_months(self) -> None:
        ref = datetime(2024, 7, 1, 12, 0, tzinfo=UTC)
        from_date, to_date = date_window(0, reference=ref)
        assert from_date == to_date

    def test_twelve_months(self) -> None:
        ref = datetime(2024, 7, 1, 0, 0, tzinfo=UTC)
        from_date, to_date = date_window(12, reference=ref)
        assert to_date == "202407010000"
        # from_date should be approximately 360 days back
        assert from_date < to_date

    def test_default_reference_is_now(self) -> None:
        from_date, to_date = date_window(1)
        # to_date should be close to now
        now_str = format_arxiv_date(utc_now())
        assert to_date[:8] == now_str[:8]  # Same date
