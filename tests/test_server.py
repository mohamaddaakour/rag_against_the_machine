"""Tests for the local HTTP API."""

from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
from fastapi.testclient import TestClient

from src.indexer import Indexer
from src.server import RagService, create_app


class FakeGenerator:
    """Counts calls instead of running a model."""

    def __init__(self) -> None:
        self.calls = 0

    def answer(self, question: str, context: str, max_new_tokens: int) -> str:
        self.calls += 1
        return f"answer#{self.calls} ctx={bool(context)}"


@pytest.fixture
def api(tmp_path: Path) -> Tuple[TestClient, FakeGenerator]:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "lora.md").write_text(
        "LoRA adapters are loaded with enable_lora.\n" * 5, encoding="utf-8"
    )
    (raw / "quant.md").write_text(
        "Quantization reduces weight precision to fp8.\n" * 5,
        encoding="utf-8",
    )
    processed = tmp_path / "data" / "processed"
    indexer = Indexer(max_chunk_size=200)
    indexer.build(raw, tmp_path)
    indexer.save(processed)

    generator = FakeGenerator()
    service = RagService(
        tmp_path, processed, generator_factory=lambda: generator
    )
    return TestClient(create_app(service)), generator


def test_health(api: Tuple[TestClient, FakeGenerator]) -> None:
    client, _ = api
    assert client.get("/health").json() == {"status": "ok"}


def test_search_returns_ranked_sources(
    api: Tuple[TestClient, FakeGenerator],
) -> None:
    client, _ = api
    response = client.post("/search", json={"query": "lora adapters", "k": 2})
    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["file_path"] == "data/raw/lora.md"
    assert len(results) <= 2


def test_search_degenerate_inputs_are_not_errors(
    api: Tuple[TestClient, FakeGenerator],
) -> None:
    client, _ = api
    empty = client.post("/search", json={"query": "", "k": 5})
    zero = client.post("/search", json={"query": "lora", "k": 0})
    assert empty.json()["results"] == []
    assert zero.json()["results"] == []


@pytest.mark.parametrize(
    "body",
    [{}, {"query": 3}, {"query": "x", "k": "5"}, {"query": "x", "k": True}],
)
def test_bad_fields_are_rejected(
    api: Tuple[TestClient, FakeGenerator], body: Dict[str, Any]
) -> None:
    client, _ = api
    assert client.post("/search", json=body).status_code == 422


def test_malformed_json_is_rejected(
    api: Tuple[TestClient, FakeGenerator],
) -> None:
    client, _ = api
    response = client.post(
        "/search",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_unknown_path_and_wrong_method(
    api: Tuple[TestClient, FakeGenerator],
) -> None:
    client, _ = api
    assert client.get("/nope").status_code == 404
    assert client.get("/search").status_code == 405


def test_answer_is_grounded_and_cached(
    api: Tuple[TestClient, FakeGenerator],
) -> None:
    client, generator = api
    body = {"query": "lora adapters", "k": 2}
    first = client.post("/answer", json=body).json()
    second = client.post("/answer", json=body).json()
    assert first["answer"] == "answer#1 ctx=True"
    assert first["cached"] is False and second["cached"] is True
    assert second["answer"] == first["answer"]
    assert generator.calls == 1


def test_answer_rejects_an_absurd_token_budget(
    api: Tuple[TestClient, FakeGenerator],
) -> None:
    client, _ = api
    body = {"query": "lora", "max_new_tokens": 0}
    assert client.post("/answer", json=body).status_code == 400


def test_stats_show_query_cache_hits(
    api: Tuple[TestClient, FakeGenerator],
) -> None:
    client, _ = api
    client.post("/search", json={"query": "lora", "k": 3})
    client.post("/search", json={"query": "lora", "k": 3})
    stats = client.get("/stats").json()
    assert stats["query_cache_hits"] == 1
    assert stats["query_cache_misses"] == 1


def test_missing_index_is_a_503(tmp_path: Path) -> None:
    client = TestClient(create_app(RagService(tmp_path, tmp_path / "none")))
    assert client.post("/search", json={"query": "x"}).status_code == 503
