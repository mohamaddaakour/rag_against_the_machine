"""The offset invariant and the two structure-aware chunking strategies."""

from pathlib import Path
from typing import List

import pytest

from src.chunking import chunk_file, chunk_fixed, chunk_markdown, chunk_python
from src.models import Chunk

SAMPLE = (
    "# Title\n\nSome prose about serving models.\n\n"
    "## Section\n\ndef get_model_config(name: str) -> Config:\n"
    "    return Config(name)\n" * 12
)

PYTHON_SOURCE = (
    "import os\n"
    "\n"
    "CONSTANT = 1\n"
    "\n"
    "\n"
    "@decorator\n"
    "def first(value):\n"
    "    return value\n"
    "\n"
    "\n"
    "class Second:\n"
    "    def method(self):\n"
    "        return 2\n"
)

MARKDOWN_SOURCE = (
    "Intro paragraph.\n"
    "\n"
    "# Title\n"
    "\n"
    "Body of the title section.\n"
    "\n"
    "## Subsection\n"
    "\n"
    "Body of the subsection.\n"
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def assert_offsets_hold(chunks: List[Chunk], text: str, cap: int) -> None:
    """Every chunk reproduces its span and respects the width cap."""
    for chunk in chunks:
        span = text[chunk.first_character_index:chunk.last_character_index]
        assert span == chunk.text
        assert 0 < len(chunk.text) <= cap


def test_span_reproduces_text() -> None:
    for chunk in chunk_fixed("data/raw/x.md", SAMPLE, 200):
        sliced = SAMPLE[
            chunk.first_character_index : chunk.last_character_index
        ]
        assert sliced == chunk.text


def test_no_chunk_exceeds_the_cap() -> None:
    for chunk in chunk_fixed("data/raw/x.md", SAMPLE, 200):
        assert chunk.last_character_index - chunk.first_character_index <= 200


def test_chunks_cover_every_character() -> None:
    chunks = chunk_fixed("data/raw/x.md", SAMPLE, 200)
    assert chunks[0].first_character_index == 0
    assert chunks[-1].last_character_index == len(SAMPLE)


def test_empty_and_blank_files_produce_nothing() -> None:
    assert chunk_file("data/raw/x.md", "", 200) == []
    assert chunk_file("data/raw/x.md", "   \n\n  ", 200) == []
    assert chunk_file("data/raw/x.py", "", 200) == []


def test_rejects_non_positive_chunk_size() -> None:
    with pytest.raises(ValueError):
        chunk_fixed("data/raw/x.md", SAMPLE, 0)
    with pytest.raises(ValueError):
        chunk_python("data/raw/x.py", PYTHON_SOURCE, 0)
    with pytest.raises(ValueError):
        chunk_markdown("data/raw/x.md", MARKDOWN_SOURCE, 0)


def test_python_cuts_on_definition_boundaries() -> None:
    chunks = chunk_python("data/raw/x.py", PYTHON_SOURCE, 60)
    assert_offsets_hold(chunks, PYTHON_SOURCE, 60)
    # The decorator stays attached to the function it decorates.
    assert any(c.text.startswith("@decorator\ndef first") for c in chunks)
    assert any(c.text.startswith("class Second:") for c in chunks)
    # Module-level code before the first definition is its own chunk.
    assert chunks[0].text.startswith("import os")


def test_python_falls_back_when_the_file_does_not_parse() -> None:
    broken = "def oops(:\n    pass\n" * 30
    chunks = chunk_python("data/raw/x.py", broken, 100)
    assert chunks
    assert_offsets_hold(chunks, broken, 100)


def test_markdown_cuts_on_headings() -> None:
    chunks = chunk_markdown("data/raw/x.md", MARKDOWN_SOURCE, 40)
    assert_offsets_hold(chunks, MARKDOWN_SOURCE, 40)
    assert any(c.text.startswith("# Title") for c in chunks)
    assert any(c.text.startswith("## Subsection") for c in chunks)


def test_dispatch_picks_a_strategy_per_extension() -> None:
    assert chunk_file("a/b.py", PYTHON_SOURCE, 60) == chunk_python(
        "a/b.py", PYTHON_SOURCE, 60
    )
    assert chunk_file("a/b.md", MARKDOWN_SOURCE, 40) == chunk_markdown(
        "a/b.md", MARKDOWN_SOURCE, 40
    )
    assert chunk_file("a/b.yaml", SAMPLE, 200) == chunk_fixed(
        "a/b.yaml", SAMPLE, 200
    )


@pytest.mark.parametrize(
    "relative",
    [
        "data/raw/vllm-0.10.1/docs/features/lora.md",
        "data/raw/vllm-0.10.1/vllm/entrypoints/openai/serving_models.py",
    ],
)
def test_offsets_hold_on_real_corpus_files(relative: str) -> None:
    path = REPO_ROOT / relative
    if not path.is_file():
        pytest.skip(f"{relative} not present")
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        text = handle.read()
    chunks = chunk_file(relative, text, 2000)
    assert chunks
    assert_offsets_hold(chunks, text, 2000)


def test_a_long_unstructured_file_is_covered_end_to_end() -> None:
    """A file with no def/heading must not be truncated at the first chunk."""
    text = "word " * 4000
    for name in ("data/raw/x.py", "data/raw/x.md", "data/raw/x.yaml"):
        chunks = chunk_file(name, text, 2000)
        assert_offsets_hold(chunks, text, 2000)
        assert chunks[0].first_character_index == 0
        assert chunks[-1].last_character_index == len(text)
