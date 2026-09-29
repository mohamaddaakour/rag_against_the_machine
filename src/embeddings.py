"""Sentence-embedding model loading and batch encoding."""

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

# Small, CPU-friendly model: ~80MB, 384-dim vectors, good default for
# semantic search over short-to-medium text chunks.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Sentences per encode() call. Keeps peak memory bounded on a CPU-only
# machine while still amortizing model overhead across many chunks.
ENCODE_BATCH_SIZE = 64

# Tokens of each chunk the model reads (the model's default is 256).
# Encoding cost grows with sequence length, so a shorter cap keeps
# indexing well inside the 5-minute budget on CPU-only machines.
MAX_SEQ_LENGTH = 128


@lru_cache(maxsize=1)
def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Load (and cache) the sentence-embedding model.

    Cached with lru_cache so repeated calls within one process (indexing,
    then later querying) don't reload the model from disk every time.
    """
    model = SentenceTransformer(model_name)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    show_progress: bool = True,
) -> np.ndarray:
    """Encode *texts* into an (n, dim) L2-normalized float32 matrix.

    Normalizing here means a later cosine similarity is a plain dot
    product: `embeddings @ query_vector`.
    """
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

    embeddings = model.encode(
        texts,
        batch_size=ENCODE_BATCH_SIZE,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)
