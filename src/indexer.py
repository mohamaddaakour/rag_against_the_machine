"""Build the lexical (BM25) and semantic indexes over the chunked corpus."""

import json
from pathlib import Path
from typing import List

# joblib is used to save Python/ML objects to disk.
import joblib
import numpy as np
import scipy.sparse as sp

# CountVectorizer turns text into a sparse matrix of term counts.
from sklearn.feature_extraction.text import CountVectorizer

# For progress bar.
from tqdm import tqdm

from src.analysis import analyze_stemmed, path_tokens
from src.chunking import chunk_file
from src.corpus import list_corpus_files, read_corpus_file
from src.embeddings import EMBEDDING_MODEL_NAME, encode_texts, load_embedding_model
from src.models import Chunk

# These define the files that will be produced.
CHUNKS_FILE = "chunks.jsonl"
LEXICAL_FILE = "bm25.joblib"
EMBEDDINGS_FILE = "embeddings.joblib"
META_FILE = "meta.json"

# BM25 parameters (the usual defaults). K1 controls how fast repeated terms
# stop adding score; B controls how much long chunks are penalised.
BM25_K1 = 1.2
BM25_B = 0.75


def bm25_weights(counts: sp.csr_matrix, k1: float = BM25_K1,
                 b: float = BM25_B) -> sp.csr_matrix:
    """Turn a (chunks x terms) count matrix into BM25 term weights.
    """
    counts = sp.csr_matrix(counts, dtype=np.float64)
    n_chunks = counts.shape[0]

    # Number of chunks each term appears in, and its BM25 idf.
    doc_freq = np.bincount(counts.indices, minlength=counts.shape[1])
    idf = np.log((n_chunks - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    lengths = np.asarray(counts.sum(axis=1)).ravel()
    norm = k1 * (1 - b + b * lengths / max(lengths.mean(), 1e-9))

    # Row index of every stored entry, to apply that chunk's length norm.
    rows = np.repeat(np.arange(n_chunks), np.diff(counts.indptr))
    tf = counts.data
    counts.data = tf * (k1 + 1) / (tf + norm[rows]) * idf[counts.indices]
    return counts


class Indexer:
    """Turn a corpus directory into a searchable, persisted index."""

    # Constructor
    def __init__(self, max_chunk_size: int = 2000) -> None:
        self.max_chunk_size = max_chunk_size
        self.chunks: List[Chunk] = []

    def build(self, raw_dir: Path, repo_root: Path) -> None:
        """Read and chunk every indexable file under *raw_dir*."""

        # `paths` is a list of all corpus files
        paths = list_corpus_files(raw_dir)

        self.chunks = []

        # `unit="file"` means every iteration in paths is a file
        # `desc` is simply the description shown before the progress bar.
        for path in tqdm(paths, desc="Chunking", unit="file"):
            try:
                file_path, text = read_corpus_file(path, repo_root)
            except OSError as exc:
                # In case we have an error reading one file we will skip it only,
                # and tqdm will print this message in the terminal
                tqdm.write(f"skipped {path}: {exc}")
                continue

            for chunk in chunk_file(file_path, text, self.max_chunk_size):
                # This makes the file's path itself become searchable text,
                # in addition to the chunk's actual content.
                chunk.indexed_text = (
                    path_tokens(file_path) + "\n" + chunk.text
                )

                self.chunks.append(chunk)

    def save(self, processed_dir: Path) -> None:
        """Write chunk metadata, the BM25 index and the embedding index.

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

        # Count terms with our identifier-aware, stemmed analyzer; the same
        # vectorizer turns queries into terms at search time.
        vectorizer = CountVectorizer(analyzer=analyze_stemmed)
        counts = vectorizer.fit_transform(c.search_text for c in self.chunks)

        # Precompute every chunk's BM25 weight for every term it contains.
        matrix = bm25_weights(counts)

        # Save everything using joblib inside this file: bm25.joblib.
        joblib.dump(
            {"vectorizer": vectorizer, "matrix": matrix},
            processed_dir / LEXICAL_FILE,
        )

        print(f"Embedding {len(self.chunks)} chunks with {EMBEDDING_MODEL_NAME} ...")

        # Build the semantic index: one normalized vector per chunk, so
        # a later cosine similarity against a query is a dot product.
        model = load_embedding_model(EMBEDDING_MODEL_NAME)
        embeddings = encode_texts(model, [c.search_text for c in self.chunks])

        joblib.dump(
            {"model_name": EMBEDDING_MODEL_NAME, "embeddings": embeddings},
            processed_dir / EMBEDDINGS_FILE,
        )

        # Metadata
        meta = {
            "max_chunk_size": self.max_chunk_size,
            "n_chunks": len(self.chunks),

            # The number of unique searchable terms (features) in the BM25 vocabulary.
            "n_features": int(matrix.shape[1]),
            "bm25_k1": BM25_K1,
            "bm25_b": BM25_B,

            # Semantic index metadata.
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dim": int(embeddings.shape[1]),
        }

        # Save the metadata
        (processed_dir / META_FILE).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
