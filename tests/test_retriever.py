"""Unit tests for Retriever.load and Retriever.search."""

from pathlib import Path

import pytest

from src.indexer import Indexer
from src.retriever import Retriever


@pytest.fixture
def retriever(tmp_path: Path) -> Retriever:
    """A retriever over a three-file corpus with distinct vocabularies."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "lora.md").write_text(
        "LoRA adapters are loaded with enable_lora.\n" * 5, encoding="utf-8"
    )
    (raw_dir / "quant.md").write_text(
        "Quantization reduces weight precision to fp8.\n" * 5, encoding="utf-8"
    )
    (raw_dir / "server.py").write_text(
        "def start_openai_server(port):\n    return port\n" * 5,
        encoding="utf-8",
    )
    processed_dir = tmp_path / "data" / "processed"
    indexer = Indexer(max_chunk_size=200)
    indexer.build(raw_dir, tmp_path)
    indexer.save(processed_dir)
    return Retriever.load(processed_dir)


def test_load_raises_when_index_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Retriever.load(tmp_path / "no_index_here")


def test_search_ranks_the_matching_file_first(retriever: Retriever) -> None:
    results = retriever.search("lora adapters", k=3)
    assert results[0].file_path == "data/raw/lora.md"


def test_search_returns_at_most_k_sources(retriever: Retriever) -> None:
    assert len(retriever.search("lora quantization server", k=2)) <= 2


def test_search_results_are_sorted_best_first(retriever: Retriever) -> None:
    scores = [s.score for s in retriever.search("lora fp8 server", k=10)]
    assert scores == sorted(scores, reverse=True)
    assert all(score > 0.0 for score in scores)


def test_search_sources_respect_the_width_limit(retriever: Retriever) -> None:
    for source in retriever.search("lora fp8 server port", k=10):
        assert 0 < source.width <= 2000


def test_search_degenerate_inputs_return_empty(retriever: Retriever) -> None:
    assert retriever.search("", k=5) == []
    assert retriever.search("   ", k=5) == []
    assert retriever.search("lora", k=0) == []
    assert retriever.search("lora", k=-1) == []
    assert retriever.search("zzzzunknownzzzz", k=5) == []


def test_search_many_matches_search_one_by_one(retriever: Retriever) -> None:
    queries = ["lora adapters", "fp8 quantization", "openai server port"]
    batched = retriever.search_many(queries, k=3)
    assert batched == [retriever.search(q, k=3) for q in queries]


def test_search_many_keeps_order_and_length_with_empty_queries(
    retriever: Retriever,
) -> None:
    queries = ["", "lora adapters", "zzzzunknownzzzz", "fp8 quantization"]
    batched = retriever.search_many(queries, k=3)
    assert len(batched) == 4
    assert batched[0] == [] and batched[2] == []
    assert batched[1][0].file_path == "data/raw/lora.md"
    assert batched[3][0].file_path == "data/raw/quant.md"


def test_search_many_crosses_a_batch_boundary(
    retriever: Retriever, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.retriever.BATCH_SIZE", 2)
    queries = ["lora adapters", "fp8 quantization", "openai server port"] * 3
    batched = retriever.search_many(queries, k=1)
    assert [r[0].file_path for r in batched] == [
        "data/raw/lora.md", "data/raw/quant.md", "data/raw/server.py"
    ] * 3


def test_repeated_query_is_served_from_the_cache(retriever: Retriever) -> None:
    first = retriever.search("lora adapters", k=3)
    second = retriever.search("lora adapters", k=3)
    assert second == first
    assert (retriever.cache_hits, retriever.cache_misses) == (1, 1)


def test_cache_is_keyed_on_k(retriever: Retriever) -> None:
    retriever.search("lora adapters", k=1)
    assert len(retriever.search("lora adapters", k=3)) >= 1
    assert retriever.cache_misses == 2


def test_cached_result_cannot_be_corrupted_by_the_caller(
    retriever: Retriever,
) -> None:
    retriever.search("lora adapters", k=3).clear()
    assert retriever.search("lora adapters", k=3) != []


def test_load_cached_reuses_and_reloads_on_rebuild(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "a.md").write_text("alpha beta\n", encoding="utf-8")
    processed = tmp_path / "processed"
    indexer = Indexer(max_chunk_size=200)
    indexer.build(raw, tmp_path)
    indexer.save(processed)

    first = Retriever.load_cached(processed)
    assert Retriever.load_cached(processed) is first

    (raw / "b.md").write_text("gamma delta\n", encoding="utf-8")
    indexer.build(raw, tmp_path)
    indexer.save(processed)
    assert Retriever.load_cached(processed) is not first
