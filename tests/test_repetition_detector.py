"""Tests for src.repetition_detector — the streaming collapse guard."""

from src.repetition_detector import RepetitionCollapseDetector


def test_normal_text_does_not_trigger():
    d = RepetitionCollapseDetector()
    for word in "The quick brown fox jumps over the lazy dog and keeps going".split():
        assert d.feed(word + " ") is False
    assert not d.collapsed


def test_single_token_runaway_triggers():
    d = RepetitionCollapseDetector(check_every_chars=10)
    d.feed("The sunset was beautiful. ")
    d.feed("own " * 20)
    assert d.collapsed
    assert d.trigger_phrase == "own"


def test_phrase_repetition_triggers():
    d = RepetitionCollapseDetector(check_every_chars=10)
    d.feed("Start of text. ")
    d.feed("the beach the beach the beach the beach the beach the beach the beach the beach the beach the beach the beach the beach ")
    assert d.collapsed
    assert "beach" in d.trigger_phrase


def test_short_text_does_not_false_positive():
    d = RepetitionCollapseDetector(check_every_chars=1)
    d.feed("yes yes yes")
    assert not d.collapsed


def test_gemma_style_collapse():
    """Simulate the exact pattern from gemma#622: word doubles then collapses."""
    d = RepetitionCollapseDetector(check_every_chars=20)
    d.feed("The contemplative contemplative atmosphere of the beach. ")
    assert not d.collapsed
    d.feed("own own own own own own own own own own own own own own own own own own ")
    assert d.collapsed
    assert d.trigger_phrase == "own"


def test_exotic_fragment_collapse():
    """MoE variant produces exotic repeated fragments."""
    d = RepetitionCollapseDetector(check_every_chars=10)
    d.feed("visually-cent, " * 20)
    assert d.collapsed


def test_reset_clears_state():
    d = RepetitionCollapseDetector(check_every_chars=10)
    d.feed("own " * 20)
    assert d.collapsed
    d.reset()
    assert not d.collapsed
    assert d.trigger_phrase is None
    d.feed("normal text here ")
    assert not d.collapsed


def test_collapse_stays_true_after_more_feed():
    d = RepetitionCollapseDetector(check_every_chars=10)
    d.feed("own " * 20)
    assert d.collapsed
    assert d.feed("more text") is True


def test_multi_word_phrase_repetition():
    d = RepetitionCollapseDetector(check_every_chars=10)
    d.feed("the same beach " * 15)
    assert d.collapsed


def test_legit_repetitive_content_ok():
    """Reasonable repetition (e.g. bullet points) should not trigger."""
    d = RepetitionCollapseDetector(check_every_chars=10)
    items = [f"Item {i}: description of item number {i} with unique content. " for i in range(30)]
    for item in items:
        d.feed(item)
    assert not d.collapsed
