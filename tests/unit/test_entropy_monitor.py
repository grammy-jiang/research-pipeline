"""Unit tests for :mod:`research_pipeline.infra.entropy_monitor`."""

from __future__ import annotations

from research_pipeline.infra.entropy_monitor import (
    EntropyMonitor,
    shannon_entropy,
    tokenize,
)


def test_shannon_entropy_zero_for_empty():
    assert shannon_entropy([]) == 0.0


def test_shannon_entropy_uniform_vs_skewed():
    uniform = shannon_entropy(["a", "b", "c", "d"])
    skewed = shannon_entropy(["a", "a", "a", "b"])
    assert uniform > skewed


def test_tokenize_lowercases_alphanumeric():
    assert tokenize("Hello, World! 123") == ["hello", "world", "123"]


def test_monitor_observes_and_records():
    m = EntropyMonitor(window_size=3, threshold=0.0, monotonic_drop_count=99)
    m.observe("alpha beta gamma")
    m.observe("delta epsilon zeta")
    assert len(m.readings) == 2
    assert m.recent_entropy() is not None


def test_low_entropy_triggers_alarm():
    m = EntropyMonitor(window_size=3, threshold=5.0, monotonic_drop_count=99)
    reading = m.observe("same same same same same same")
    assert reading.alarm is True
    assert m.alarm_count() == 1


def test_monotonic_drop_triggers_alarm():
    m = EntropyMonitor(
        window_size=1,  # each observe reflects only the latest sample
        threshold=-1.0,  # disable the threshold alarm
        monotonic_drop_count=3,
    )
    # Strictly decreasing entropy sequence
    m.observe("a b c d e f g h i j")  # 10 unique tokens → ~3.32 bits
    m.observe("a b c d e f")  # 6 unique → ~2.58 bits
    r = m.observe("a a a b")  # 2 unique → lower
    assert r.alarm is True


def test_reset_clears_state():
    m = EntropyMonitor(window_size=3, threshold=5.0)
    m.observe("same same same")
    m.reset()
    assert m.readings == []
    assert m.recent_entropy() is None
