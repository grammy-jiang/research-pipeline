"""Unit tests for arxiv.parser module."""

from datetime import UTC, datetime

from arxiv_paper_pipeline.arxiv.parser import (
    parse_atom_response,
    parse_total_results,
)

SAMPLE_ATOM_RESPONSE = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>2</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <opensearch:itemsPerPage>10</opensearch:itemsPerPage>
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <updated>2024-02-15T10:30:00Z</updated>
    <published>2024-01-20T08:00:00Z</published>
    <title>Neural Information Retrieval: A Comprehensive Survey</title>
    <summary>This paper surveys neural approaches to information retrieval,
    covering dense and sparse methods.</summary>
    <author><name>Alice Researcher</name></author>
    <author><name>Bob Scientist</name></author>
    <category term="cs.IR" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <arxiv:primary_category term="cs.IR" scheme="http://arxiv.org/schemas/atom"/>
    <link href="http://arxiv.org/abs/2401.12345v2" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2401.12345v2" title="pdf" type="application/pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2403.67890v1</id>
    <updated>2024-03-10T14:00:00Z</updated>
    <published>2024-03-10T14:00:00Z</published>
    <title>Lightweight Embedding Models</title>
    <summary>We propose a lightweight embedding approach.</summary>
    <author><name>Carol Expert</name></author>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
    <arxiv:primary_category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
    <link href="http://arxiv.org/abs/2403.67890v1" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2403.67890v1" title="pdf" type="application/pdf"/>
  </entry>
</feed>
"""

EMPTY_ATOM_RESPONSE = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>0</opensearch:totalResults>
</feed>
"""


class TestParseAtomResponse:
    def test_parses_two_entries(self) -> None:
        results = parse_atom_response(SAMPLE_ATOM_RESPONSE)
        assert len(results) == 2

    def test_first_entry_fields(self) -> None:
        results = parse_atom_response(SAMPLE_ATOM_RESPONSE)
        entry = results[0]
        assert entry.arxiv_id == "2401.12345"
        assert entry.version == "v2"
        assert entry.title == "Neural Information Retrieval: A Comprehensive Survey"
        assert entry.authors == ["Alice Researcher", "Bob Scientist"]
        assert entry.categories == ["cs.IR", "cs.CL"]
        assert entry.primary_category == "cs.IR"
        assert "surveys neural approaches" in entry.abstract

    def test_published_datetime(self) -> None:
        results = parse_atom_response(SAMPLE_ATOM_RESPONSE)
        entry = results[0]
        assert entry.published == datetime(2024, 1, 20, 8, 0, tzinfo=UTC)

    def test_updated_datetime(self) -> None:
        results = parse_atom_response(SAMPLE_ATOM_RESPONSE)
        entry = results[0]
        assert entry.updated == datetime(2024, 2, 15, 10, 30, tzinfo=UTC)

    def test_urls(self) -> None:
        results = parse_atom_response(SAMPLE_ATOM_RESPONSE)
        entry = results[0]
        assert "2401.12345" in entry.abs_url
        assert "2401.12345" in entry.pdf_url

    def test_second_entry(self) -> None:
        results = parse_atom_response(SAMPLE_ATOM_RESPONSE)
        entry = results[1]
        assert entry.arxiv_id == "2403.67890"
        assert entry.version == "v1"
        assert entry.primary_category == "cs.AI"

    def test_empty_response(self) -> None:
        results = parse_atom_response(EMPTY_ATOM_RESPONSE)
        assert len(results) == 0

    def test_whitespace_in_title_normalized(self) -> None:
        results = parse_atom_response(SAMPLE_ATOM_RESPONSE)
        entry = results[0]
        assert "\n" not in entry.title
        assert "  " not in entry.title


class TestParseTotalResults:
    def test_total_results(self) -> None:
        count = parse_total_results(SAMPLE_ATOM_RESPONSE)
        assert count == 2

    def test_empty_response(self) -> None:
        count = parse_total_results(EMPTY_ATOM_RESPONSE)
        assert count == 0
