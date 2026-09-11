"""Build the lexical index over the chunked corpus and persist it."""

import json
from pathlib import Path
from typing import List

# joblib is used to save Python/ML objects to disk.
import joblib

# TfidfVectorizer transforms text into numerical vectors.
from sklearn.feature_extraction.text import TfidfVectorizer

# For progress bar.
from tqdm import tqdm

from src.chunking import chunk_file
from src.corpus import list_corpus_files, read_corpus_file
from src.models import Chunk

# These define the files that will be produced.
CHUNKS_FILE = "chunks.jsonl"
TFIDF_FILE = "tfidf.joblib"
META_FILE = "meta.json"


class Indexer:
    """Turn a corpus directory into a searchable, persisted index."""

    # Constructor
    def __init__(self, max_chunk_size: int = 2000) -> None:
        self.max_chunk_size = max_chunk_size
        self.chunks = []


    def build(self, raw_dir: Path, repo_root: Path) -> None:
        """Read and chunk every indexable file under *raw_dir*.
        To print the progress bar"""

        paths = list_corpus_files(raw_dir)

        self.chunks = []

        # `unit="file"` means every iteration in paths is a file
        # `desc` is simply the description shown before the progress bar.
        for path in tqdm(paths, desc="Chunking", unit="file"):
            try:
                file_path, text = read_corpus_file(path, repo_root)
            except OSError as exc:
                tqdm.write(f"skipped {path}: {exc}")
                continue

            self.chunks.extend(chunk_file(file_path, text, self.max_chunk_size))