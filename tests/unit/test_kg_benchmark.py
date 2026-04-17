"""Tests for quality.kg_benchmark — Scientific KG Benchmark."""

from __future__ import annotations

import pytest

from research_pipeline.quality.kg_benchmark import (
    BenchmarkMetrics,
    EntityMetrics,
    ExtractionResult,
    ExtractedTriple,
    FullBenchmarkReport,
    GoldDataset,
    GoldTriple,
    KGBenchmarkRunner,
    NormalizationStrategy,
    PredicateMetrics,
    PredefinedSeeds,
    _normalize_lemma_like,
    _normalize_lowercase,
)


# ---------------------------------------------------------------------------
# GoldTriple / GoldDataset
# ---------------------------------------------------------------------------


class TestGoldTriple:
    def test_as_tuple(self) -> None:
        t = GoldTriple("A", "rel", "B")
        assert t.as_tuple() == ("A", "rel", "B")

    def test_to_dict(self) -> None:
        t = GoldTriple("X", "y", "Z", source_paper_id="p1", confidence=0.9)
        d = t.to_dict()
        assert d["subject"] == "X"
        assert d["object"] == "Z"
        assert d["confidence"] == 0.9

    def test_frozen(self) -> None:
        t = GoldTriple("A", "r", "B")
        with pytest.raises(AttributeError):
            t.subject = "C"  # type: ignore[misc]


class TestGoldDataset:
    def test_properties(self) -> None:
        ds = GoldDataset(
            name="test",
            domain="test",
            triples=[
                GoldTriple("A", "r1", "B"),
                GoldTriple("A", "r2", "C"),
            ],
        )
        assert ds.num_triples == 2
        assert ds.entities == {"A", "B", "C"}
        assert ds.predicates == {"r1", "r2"}

    def test_to_dict(self) -> None:
        ds = GoldDataset(name="d", domain="dom", triples=[GoldTriple("X", "y", "Z")])
        d = ds.to_dict()
        assert d["name"] == "d"
        assert d["num_triples"] == 1


# ---------------------------------------------------------------------------
# ExtractedTriple / ExtractionResult
# ---------------------------------------------------------------------------


class TestExtractedTriple:
    def test_as_tuple(self) -> None:
        t = ExtractedTriple("A", "r", "B")
        assert t.as_tuple() == ("A", "r", "B")

    def test_to_dict(self) -> None:
        t = ExtractedTriple("A", "r", "B", extraction_confidence=0.8)
        d = t.to_dict()
        assert d["extraction_confidence"] == 0.8


class TestExtractionResult:
    def test_num_triples(self) -> None:
        er = ExtractionResult(
            system_name="test", triples=[ExtractedTriple("A", "r", "B")]
        )
        assert er.num_triples == 1


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_lowercase(self) -> None:
        assert _normalize_lowercase("  Hello World  ") == "hello world"

    def test_lemma_like_strips_plural(self) -> None:
        assert _normalize_lemma_like("transformers") == "transformer"

    def test_lemma_like_short_word(self) -> None:
        assert _normalize_lemma_like("is") == "is"

    def test_lemma_like_collapses_whitespace(self) -> None:
        assert _normalize_lemma_like("  multiple   spaces  ") == "multiple space"


# ---------------------------------------------------------------------------
# BenchmarkMetrics / to_dict
# ---------------------------------------------------------------------------


class TestBenchmarkMetrics:
    def test_to_dict(self) -> None:
        m = BenchmarkMetrics(
            precision=0.8,
            recall=0.6,
            f1=0.6857,
            hallucination_rate=0.2,
            missing_rate=0.4,
            true_positives=6,
            false_positives=2,
            false_negatives=4,
            total_gold=10,
            total_extracted=8,
        )
        d = m.to_dict()
        assert d["precision"] == 0.8
        assert d["true_positives"] == 6


class TestEntityMetrics:
    def test_to_dict(self) -> None:
        m = EntityMetrics(
            precision=0.9, recall=0.8, f1=0.85, gold_entities=10,
            extracted_entities=9, matched=8
        )
        d = m.to_dict()
        assert d["matched"] == 8


class TestPredicateMetrics:
    def test_to_dict(self) -> None:
        m = PredicateMetrics(
            precision=1.0, recall=0.5, f1=0.6667, gold_predicates=4,
            extracted_predicates=2, matched=2
        )
        d = m.to_dict()
        assert d["gold_predicates"] == 4


# ---------------------------------------------------------------------------
# KGBenchmarkRunner
# ---------------------------------------------------------------------------


def _make_gold() -> GoldDataset:
    return GoldDataset(
        name="test",
        domain="test",
        triples=[
            GoldTriple("BERT", "is_a", "language model"),
            GoldTriple("GPT", "is_a", "language model"),
            GoldTriple("BERT", "uses", "transformer"),
            GoldTriple("attention", "is_component_of", "transformer"),
        ],
    )


def _make_extraction(triples: list[tuple[str, str, str]]) -> ExtractionResult:
    return ExtractionResult(
        system_name="test_sys",
        triples=[ExtractedTriple(s, p, o) for s, p, o in triples],
    )


class TestKGBenchmarkRunner:
    def test_perfect_match(self) -> None:
        gold = _make_gold()
        ext = _make_extraction([
            ("bert", "is_a", "language model"),
            ("gpt", "is_a", "language model"),
            ("bert", "uses", "transformer"),
            ("attention", "is_component_of", "transformer"),
        ])
        runner = KGBenchmarkRunner(strategy=NormalizationStrategy.LOWERCASE)
        m = runner.evaluate_triples(gold, ext)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.hallucination_rate == 0.0

    def test_partial_match(self) -> None:
        gold = _make_gold()
        ext = _make_extraction([
            ("bert", "is_a", "language model"),
            ("gpt", "is_a", "language model"),
            # missing 2 gold triples, add 1 hallucinated
            ("gpt", "beats", "bert"),
        ])
        runner = KGBenchmarkRunner(strategy=NormalizationStrategy.LOWERCASE)
        m = runner.evaluate_triples(gold, ext)
        assert m.true_positives == 2
        assert m.false_positives == 1
        assert m.false_negatives == 2

    def test_exact_strategy(self) -> None:
        gold = _make_gold()
        ext = _make_extraction([
            ("bert", "is_a", "language model"),  # case mismatch
        ])
        runner = KGBenchmarkRunner(strategy=NormalizationStrategy.EXACT)
        m = runner.evaluate_triples(gold, ext)
        assert m.true_positives == 0  # "bert" != "BERT"

    def test_lemma_like_strategy(self) -> None:
        gold = GoldDataset(
            name="t",
            domain="t",
            triples=[GoldTriple("transformers", "are", "models")],
        )
        ext = _make_extraction([("transformer", "are", "model")])
        runner = KGBenchmarkRunner(strategy=NormalizationStrategy.LEMMA_LIKE)
        m = runner.evaluate_triples(gold, ext)
        assert m.true_positives == 1

    def test_alias_strategy(self) -> None:
        gold = GoldDataset(
            name="t",
            domain="t",
            triples=[GoldTriple("SVM", "is_a", "classifier")],
            entity_aliases={"SVM": ["Support Vector Machine"]},
        )
        ext = _make_extraction([("support vector machine", "is_a", "classifier")])
        runner = KGBenchmarkRunner(
            strategy=NormalizationStrategy.ALIAS,
            alias_map=gold.entity_aliases,
        )
        m = runner.evaluate_triples(gold, ext)
        assert m.true_positives == 1

    def test_entity_metrics(self) -> None:
        gold = _make_gold()
        ext = _make_extraction([
            ("bert", "is_a", "language model"),
        ])
        runner = KGBenchmarkRunner(strategy=NormalizationStrategy.LOWERCASE)
        em = runner.evaluate_entities(gold, ext)
        assert em.matched >= 1
        assert em.gold_entities == 5  # bert, gpt, language model, transformer, attention

    def test_predicate_metrics(self) -> None:
        gold = _make_gold()
        ext = _make_extraction([("bert", "is_a", "language model")])
        runner = KGBenchmarkRunner(strategy=NormalizationStrategy.LOWERCASE)
        pm = runner.evaluate_predicates(gold, ext)
        assert pm.matched == 1
        assert pm.gold_predicates == 3

    def test_per_predicate_f1(self) -> None:
        gold = _make_gold()
        ext = _make_extraction([
            ("bert", "is_a", "language model"),
            ("gpt", "is_a", "language model"),
        ])
        runner = KGBenchmarkRunner(strategy=NormalizationStrategy.LOWERCASE)
        ppf1 = runner.per_predicate_f1(gold, ext)
        assert ppf1["is_a"] == 1.0
        assert ppf1.get("uses", 0.0) == 0.0

    def test_find_hallucinated(self) -> None:
        gold = _make_gold()
        ext = _make_extraction([
            ("bert", "is_a", "language model"),
            ("gpt", "beats", "bert"),
        ])
        runner = KGBenchmarkRunner(strategy=NormalizationStrategy.LOWERCASE)
        hall = runner.find_hallucinated(gold, ext)
        assert len(hall) == 1
        assert hall[0].predicate == "beats"

    def test_find_missing(self) -> None:
        gold = _make_gold()
        ext = _make_extraction([("bert", "is_a", "language model")])
        runner = KGBenchmarkRunner(strategy=NormalizationStrategy.LOWERCASE)
        missing = runner.find_missing(gold, ext)
        assert len(missing) == 3

    def test_full_report(self) -> None:
        gold = _make_gold()
        ext = _make_extraction([
            ("bert", "is_a", "language model"),
            ("gpt", "is_a", "language model"),
            ("bert", "uses", "transformer"),
        ])
        runner = KGBenchmarkRunner(strategy=NormalizationStrategy.LOWERCASE)
        report = runner.full_report(gold, ext)
        assert isinstance(report, FullBenchmarkReport)
        assert report.triple_metrics.true_positives == 3
        d = report.to_dict()
        assert d["dataset_name"] == "test"
        assert d["system_name"] == "test_sys"

    def test_empty_gold(self) -> None:
        gold = GoldDataset(name="empty", domain="t")
        ext = _make_extraction([("a", "b", "c")])
        runner = KGBenchmarkRunner()
        m = runner.evaluate_triples(gold, ext)
        assert m.recall == 0.0
        assert m.hallucination_rate == 1.0

    def test_empty_extraction(self) -> None:
        gold = _make_gold()
        ext = ExtractionResult(system_name="empty")
        runner = KGBenchmarkRunner()
        m = runner.evaluate_triples(gold, ext)
        assert m.precision == 0.0
        assert m.missing_rate == 1.0

    def test_both_empty(self) -> None:
        gold = GoldDataset(name="e", domain="d")
        ext = ExtractionResult(system_name="e")
        runner = KGBenchmarkRunner()
        m = runner.evaluate_triples(gold, ext)
        assert m.f1 == 0.0


# ---------------------------------------------------------------------------
# Predefined seeds
# ---------------------------------------------------------------------------


class TestPredefinedSeeds:
    def test_nlp_basics(self) -> None:
        ds = PredefinedSeeds.nlp_basics()
        assert ds.num_triples == 10
        assert ds.domain == "natural language processing"

    def test_ml_methods(self) -> None:
        ds = PredefinedSeeds.ml_methods()
        assert ds.num_triples == 8

    def test_list_datasets(self) -> None:
        names = PredefinedSeeds.list_datasets()
        assert "nlp_basics" in names
        assert "ml_methods" in names

    def test_get_valid(self) -> None:
        ds = PredefinedSeeds.get("nlp_basics")
        assert ds.name == "nlp_basics"

    def test_get_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown dataset"):
            PredefinedSeeds.get("nonexistent")

    def test_smoke_benchmark_on_seeds(self) -> None:
        """Smoke test: run benchmark runner on predefined seed datasets."""
        ds = PredefinedSeeds.nlp_basics()
        ext = ExtractionResult(
            system_name="test",
            triples=[
                ExtractedTriple("BERT", "is_a", "language model"),
                ExtractedTriple("BERT", "uses", "transformer"),
            ],
        )
        runner = KGBenchmarkRunner(
            strategy=NormalizationStrategy.ALIAS,
            alias_map=ds.entity_aliases,
        )
        report = runner.full_report(ds, ext)
        assert report.triple_metrics.true_positives == 2
        assert report.triple_metrics.recall == pytest.approx(0.2, abs=0.01)
