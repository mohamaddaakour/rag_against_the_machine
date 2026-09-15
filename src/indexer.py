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
        self.chunks: List[Chunk] = []

    def build(self, raw_dir: Path, repo_root: Path) -> None:
        """Read and chunk every indexable file under *raw_dir*."""

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

    def save(self, processed_dir: Path) -> None:
        """Write chunk metadata, the fitted vectorizer and the matrix.

        Args:
            processed_dir: Directory to write the generated index into.
        """

        if not self.chunks:
            raise ValueError("No chuncks provided by build()")

        # Create the directory
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Join the two paths
        chunks_path = processed_dir / CHUNKS_FILE

        with chunks_path.open(mode="w", encoding="utf-8", newline="\n") as handle:
            for chunk in self.chunks:
                handle.write(chunk.to_source().model_dump_json() + "\n")

        print(f"Vectorizing {len(self.chunks)} chunks ...")

        # create the object that converts text into vectors.
        vectorizer: TfidfVectorizer = TfidfVectorizer(sublinear_tf=True)

        matrix = vectorizer.fit_transform(c.search_text for c in self.chunks)

        # Save everything using joblib.
        joblib.dump(
            {"vectorizer": vectorizer, "matrix": matrix},
            processed_dir / TFIDF_FILE,
        )

        # Metadata
        meta = {
            "max_chunk_size": self.max_chunk_size,
            "n_chunks": len(self.chunks),
            "n_features": int(matrix.shape[1]),
        }

        # Save the metadata
        (processed_dir / META_FILE).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
