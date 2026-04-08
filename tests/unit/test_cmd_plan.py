"""Tests for stop-word filtering in query plan generation."""

from research_pipeline.cli.cmd_plan import _filter_stop_words, _split_topic_terms


class TestFilterStopWords:
    """Test stop-word removal from search terms."""

    def test_removes_common_stop_words(self) -> None:
        terms = ["local", "memory", "for", "ai", "agents"]
        result = _filter_stop_words(terms)
        assert "for" not in result
        assert "local" in result
        assert "memory" in result

    def test_removes_multiple_stop_words(self) -> None:
        terms = ["the", "role", "of", "memory", "in", "agent", "systems"]
        result = _filter_stop_words(terms)
        assert "the" not in result
        assert "of" not in result
        assert "in" not in result
        assert "role" in result
        assert "memory" in result

    def test_preserves_non_stop_words(self) -> None:
        terms = ["neural", "network", "retrieval"]
        result = _filter_stop_words(terms)
        assert result == ["neural", "network", "retrieval"]

    def test_empty_input(self) -> None:
        assert _filter_stop_words([]) == []

    def test_all_stop_words(self) -> None:
        terms = ["the", "a", "an", "of", "for", "in"]
        result = _filter_stop_words(terms)
        assert result == []

    def test_case_insensitive(self) -> None:
        terms = ["The", "Memory", "FOR", "agents"]
        result = _filter_stop_words(terms)
        assert "The" not in result
        assert "FOR" not in result
        assert "Memory" in result
        assert "agents" in result


class TestSplitTopicTerms:
    """Test smart splitting of topic into must_terms and nice_terms."""

    def test_short_topic_all_must(self) -> None:
        must, nice = _split_topic_terms("memory agents")
        assert must == ["memory", "agents"]
        assert nice == []

    def test_long_topic_splits(self) -> None:
        must, nice = _split_topic_terms("local memory system for ai agents")
        # "for" is a stop word, should be removed
        # First 3 non-stop terms become must, rest become nice
        assert len(must) <= 3
        assert "for" not in must
        assert "for" not in nice

    def test_caps_must_at_three(self) -> None:
        must, nice = _split_topic_terms(
            "neural network retrieval augmented generation optimization"
        )
        assert len(must) <= 3
        # Overflow goes to nice_terms
        assert len(nice) > 0

    def test_single_word_topic(self) -> None:
        must, nice = _split_topic_terms("transformers")
        assert must == ["transformers"]
        assert nice == []

    def test_stop_words_removed_before_split(self) -> None:
        must, nice = _split_topic_terms("the role of transformers in nlp")
        assert "the" not in must + nice
        assert "of" not in must + nice
        assert "in" not in must + nice
        # Remaining: "role", "transformers", "nlp"
        assert "role" in must + nice
        assert "transformers" in must + nice
        assert "nlp" in must + nice
