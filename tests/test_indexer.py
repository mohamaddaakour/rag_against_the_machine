"""Unit tests for Indexer.build and Indexer.save."""

import json
from pathlib import Path
from typing import Tuple

import joblib
import pytest

from src.indexer import CHUNKS_FILE, META_FILE, TFIDF_FILE, Indexer


def _make_corpus(repo_root: Path) -> Path:
    """Create a tiny two-file corpus under *repo_root* and return raw_dir."""
    raw_dir = repo_root / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "a.py").write_text(
        "def get_model_config(name):\n    return name\n" * 5,
        encoding="utf-8",
    )
    (raw_dir / "b.md").write_text(
        "# Title\n\nSome prose about serving models.\n" * 5,
        encoding="utf-8",
    )
    return raw_dir


def test_build_populates_chunks_with_grader_exact_paths(tmp_path: Path) -> None:
    raw_dir = _make_corpus(tmp_path)
    indexer = Indexer(max_chunk_size=50)

    indexer.build(raw_dir, tmp_path)

    assert len(indexer.chunks) > 0
    file_paths = {chunk.file_path for chunk in indexer.chunks}
    assert file_paths == {"data/raw/a.py", "data/raw/b.md"}
    assert all("\\" not in p for p in file_paths)


def test_build_resets_chunks_on_each_call(tmp_path: Path) -> None:
    raw_dir = _make_corpus(tmp_path)
    indexer = Indexer(max_chunk_size=50)

    indexer.build(raw_dir, tmp_path)
    first_count = len(indexer.chunks)
    indexer.build(raw_dir, tmp_path)

    assert len(indexer.chunks) == first_count


def test_build_skips_unreadable_files_without_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = _make_corpus(tmp_path)

    def _boom(path: Path, repo_root: Path) -> Tuple[str, str]:
        raise OSError("cannot decode")

    monkeypatch.setattr("src.indexer.read_corpus_file", _boom)
    indexer = Indexer(max_chunk_size=50)

    indexer.build(raw_dir, tmp_path)

    assert indexer.chunks == []


def test_build_raises_on_missing_raw_dir(tmp_path: Path) -> None:
    indexer = Indexer(max_chunk_size=50)
    with pytest.raises(FileNotFoundError):
        indexer.build(tmp_path / "does_not_exist", tmp_path)


def test_save_without_build_raises(tmp_path: Path) -> None:
    indexer = Indexer(max_chunk_size=50)
    with pytest.raises(ValueError):
        indexer.save(tmp_path / "processed")


def test_save_writes_the_three_artefacts(tmp_path: Path) -> None:
    raw_dir = _make_corpus(tmp_path)
    processed_dir = tmp_path / "data" / "processed"
    indexer = Indexer(max_chunk_size=50)
    indexer.build(raw_dir, tmp_path)

    indexer.save(processed_dir)

    assert (processed_dir / CHUNKS_FILE).exists()
    assert (processed_dir / TFIDF_FILE).exists()
    assert (processed_dir / META_FILE).exists()


def test_chunks_jsonl_has_exactly_three_fields_per_line(tmp_path: Path) -> None:
    raw_dir = _make_corpus(tmp_path)
    processed_dir = tmp_path / "data" / "processed"
    indexer = Indexer(max_chunk_size=50)
    indexer.build(raw_dir, tmp_path)
    indexer.save(processed_dir)

    lines = (processed_dir / CHUNKS_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(indexer.chunks)
    for line in lines:
        row = json.loads(line)
        assert set(row) == {
            "file_path",
            "first_character_index",
            "last_character_index",
        }


def test_meta_json_matches_the_built_index(tmp_path: Path) -> None:
    raw_dir = _make_corpus(tmp_path)
    processed_dir = tmp_path / "data" / "processed"
    indexer = Indexer(max_chunk_size=50)
    indexer.build(raw_dir, tmp_path)
    indexer.save(processed_dir)

    meta = json.loads((processed_dir / META_FILE).read_text(encoding="utf-8"))
    assert meta["max_chunk_size"] == 50
    assert meta["n_chunks"] == len(indexer.chunks)
    assert meta["n_features"] > 0


def test_tfidf_matrix_has_one_row_per_chunk(tmp_path: Path) -> None:
    raw_dir = _make_corpus(tmp_path)
    processed_dir = tmp_path / "data" / "processed"
    indexer = Indexer(max_chunk_size=50)
    indexer.build(raw_dir, tmp_path)
    indexer.save(processed_dir)

    payload = joblib.load(processed_dir / TFIDF_FILE)
    assert payload["matrix"].shape[0] == len(indexer.chunks)
    assert payload["matrix"].shape[1] == payload["vectorizer"].transform(["x"]).shape[1]


def test_default_max_chunk_size_is_2000() -> None:
    assert Indexer().max_chunk_size == 2000
