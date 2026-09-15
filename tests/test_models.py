"""The graded JSON shape, pinned."""

import json

import pytest
from pydantic import ValidationError

from src.models import Chunk, MinimalSource, ScoredSource


def test_minimal_source_serializes_exactly_three_fields() -> None:
    source = MinimalSource(
        file_path="data/raw/vllm-0.10.1/docs/features/lora.md",
        first_character_index=4695,
        last_character_index=6098,
    )
    assert set(json.loads(source.model_dump_json())) == {
        "file_path",
        "first_character_index",
        "last_character_index",
    }
    assert source.width == 1403


def test_scored_source_is_not_a_minimal_source_on_the_wire() -> None:
    scored = ScoredSource(
        file_path="a.md",
        first_character_index=0,
        last_character_index=5,
        score=0.42,
    )
    assert "score" in json.loads(scored.model_dump_json())


def test_reversed_span_is_rejected() -> None:
    with pytest.raises(ValidationError):
        MinimalSource(
            file_path="a.md",
            first_character_index=10,
            last_character_index=3,
        )


def test_chunk_search_text_defaults_to_text() -> None:
    chunk = Chunk(
        file_path="a.md",
        first_character_index=0,
        last_character_index=3,
        text="abc",
    )
    assert chunk.search_text == "abc"
    enriched = chunk.model_copy(update={"indexed_text": "docs lora abc"})
    assert enriched.search_text == "docs lora abc"
