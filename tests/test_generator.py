"""Unit tests for prompt building and span re-reading.

The model itself is never loaded here: these cover the parts that decide what
the model sees, which is where grounding is won or lost.
"""

from pathlib import Path

from src.generator import (
    build_context,
    build_messages,
    read_span,
    _strip_thinking,
)
from src.models import MinimalSource


def make_corpus(root: Path) -> None:
    """Two small corpus files under `root`."""
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "a.md").write_text(
        "HEADER\nLoRA adapters are loaded with enable_lora.\nFOOTER\n",
        encoding="utf-8",
        newline="",
    )
    (root / "data" / "raw" / "b.md").write_text(
        "Quantization reduces precision.\n", encoding="utf-8", newline=""
    )


def source(path: str, first: int, last: int) -> MinimalSource:
    """Shorthand for building a source in these tests."""
    return MinimalSource(
        file_path=path, first_character_index=first, last_character_index=last
    )


def test_read_span_returns_exactly_the_span(tmp_path: Path) -> None:
    make_corpus(tmp_path)
    span = read_span(tmp_path, source("data/raw/a.md", 7, 50))
    assert span == "LoRA adapters are loaded with enable_lora.\n"


def test_read_span_of_a_missing_file_is_empty(tmp_path: Path) -> None:
    assert read_span(tmp_path, source("data/raw/gone.md", 0, 10)) == ""


def test_build_context_labels_each_span_with_its_file(tmp_path: Path) -> None:
    make_corpus(tmp_path)
    context = build_context(
        tmp_path,
        [source("data/raw/a.md", 7, 50), source("data/raw/b.md", 0, 31)],
    )
    assert "--- data/raw/a.md ---" in context
    assert "--- data/raw/b.md ---" in context
    assert "enable_lora" in context
    assert "Quantization" in context


def test_build_context_respects_the_character_budget(tmp_path: Path) -> None:
    make_corpus(tmp_path)
    context = build_context(
        tmp_path,
        [source("data/raw/a.md", 7, 50), source("data/raw/b.md", 0, 31)],
        max_chars=70,
    )
    assert "enable_lora" in context
    assert "Quantization" not in context


def test_build_context_skips_missing_and_blank_spans(tmp_path: Path) -> None:
    make_corpus(tmp_path)
    context = build_context(
        tmp_path,
        [source("data/raw/gone.md", 0, 10), source("data/raw/a.md", 6, 7)],
    )
    assert context == ""


def test_build_context_of_no_sources_is_empty(tmp_path: Path) -> None:
    assert build_context(tmp_path, []) == ""


def test_build_messages_carries_question_and_context() -> None:
    messages = build_messages("How?", "--- a.md ---\nbody")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "How?" in messages[1]["content"]
    assert "body" in messages[1]["content"]


def test_build_messages_says_so_when_nothing_was_retrieved() -> None:
    messages = build_messages("How?", "")
    assert "No context was retrieved" in messages[1]["content"]


def test_strip_thinking_drops_a_reasoning_block() -> None:
    assert _strip_thinking("<think>hmm</think>The answer.") == "The answer."
    assert _strip_thinking("The answer.") == "The answer."
