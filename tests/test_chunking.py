"""The offset invariant: a chunk span must reproduce the chunk text."""

import pytest

from src.chunking import chunk_file, chunk_fixed

SAMPLE = (
    "# Title\n\nSome prose about serving models.\n\n"
    "## Section\n\ndef get_model_config(name: str) -> Config:\n"
    "    return Config(name)\n" * 12
)


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


def test_rejects_non_positive_chunk_size() -> None:
    with pytest.raises(ValueError):
        chunk_fixed("data/raw/x.md", SAMPLE, 0)
