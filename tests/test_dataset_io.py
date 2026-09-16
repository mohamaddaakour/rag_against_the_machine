"""Unit tests for dataset loading and result writing."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.dataset_io import load_dataset, save_search_results
from src.models import MinimalSearchResults, MinimalSource, StudentSearchResults

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_load_dataset_reads_the_real_public_datasets() -> None:
    for scope in ("UnansweredQuestions", "AnsweredQuestions"):
        path = REPO_ROOT / "data" / "datasets" / scope / "dataset_docs_public.json"
        if not path.is_file():
            pytest.skip(f"{path} not present")
        assert len(load_dataset(path).rag_questions) == 100


def test_load_dataset_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.json")


def test_load_dataset_malformed_json_raises_validation_error(
    tmp_path: Path,
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_dataset(bad)


def test_save_search_results_round_trips(tmp_path: Path) -> None:
    results = StudentSearchResults(
        k=5,
        search_results=[
            MinimalSearchResults(
                question_id="q1",
                question="How?",
                retrieved_sources=[
                    MinimalSource(
                        file_path="data/raw/a.md",
                        first_character_index=0,
                        last_character_index=10,
                    )
                ],
            )
        ],
    )
    out = save_search_results(results, tmp_path / "out" / "Scope", "d.json")
    text = out.read_text(encoding="utf-8")

    assert out == tmp_path / "out" / "Scope" / "d.json"
    assert StudentSearchResults.model_validate_json(text) == results
    source = json.loads(text)["search_results"][0]["retrieved_sources"][0]
    assert set(source) == {
        "file_path", "first_character_index", "last_character_index"
    }
