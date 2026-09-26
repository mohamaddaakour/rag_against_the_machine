"""Load a persisted index and rank chunks against a query."""

# a dictionary that remembers insertion order. It is used to build the query cache.
from collections import OrderedDict

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from tqdm import tqdm

from src.indexer import CHUNKS_FILE, TFIDF_FILE
from src.models import MinimalSource, ScoredSource

# Process at most 64 queries at a time.
BATCH_SIZE = 64

# Number of (query, k) results kept in memory per retriever.
QUERY_CACHE_SIZE = 256


class Retriever:
    """Rank indexed chunks against a natural-language query."""

    def __init__(
        self,
        sources: List[MinimalSource],
        vectorizer: Any,
        matrix: Any,
    ) -> None:
        self.sources = sources
        self.vectorizer = vectorizer
        self.matrix = matrix
        self._query_cache: "OrderedDict[Tuple[str, int], List[ScoredSource]]" = OrderedDict()

        # Used for statistics
        self.cache_hits = 0
        self.cache_misses = 0

    @classmethod
    def load(cls, processed_dir: Path) -> "Retriever":
        """Create the Retriver instance.

        Raises:
            FileNotFoundError: If the index has not been built yet.
        """
        chunks_path = processed_dir / CHUNKS_FILE
        tfidf_path = processed_dir / TFIDF_FILE

        if not chunks_path.exists() or not tfidf_path.exists():
            raise FileNotFoundError(
                f"no index in {processed_dir} - run: python -m src index"
            )

        sources: List[MinimalSource] = []

        with chunks_path.open(encoding="utf-8") as handle:
            # Iterate the file line by line.
            for line in handle:
                if line.strip():
                    sources.append(MinimalSource.model_validate_json(line))

        payload: Dict[str, Any] = joblib.load(tfidf_path)

        # Return a Retriver class instance
        return cls(sources, payload["vectorizer"], payload["matrix"])

    @classmethod
    def load_cached(cls, processed_dir: Path) -> "Retriever":
        """Like `load`, but reuse the in-memory retriever while the index
        files are unchanged; a rebuilt index is picked up automatically.

        Raises:
            FileNotFoundError: If the index has not been built yet.
        """
        return _load_index(processed_dir.resolve(), _index_stamp(processed_dir))

    def clear_cache(self) -> None:
        """Forget every cached query result."""
        self._query_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def search(self, query: str, k: int = 10) -> List[ScoredSource]:
        """Top-*k* sources for *query*, best first.

        Returns an empty list for an empty query, a non-positive *k*, or a
        query whose every term is absent from the vocabulary.
        """
        return self.search_many([query], k)[0]

    def search_many(
        self, queries: List[str], k: int = 10
    ) -> List[List[ScoredSource]]:
        """Top-*k* sources for each query, in the same order as *queries*.

        An empty query, a non-positive *k*, or a query with no known
        vocabulary term gets an empty list; the other queries are unaffected.
        """
        results: List[List[ScoredSource]] = [[] for _ in queries]

        if k <= 0:
            return results

        # Indexes of not blank queries that worth scoring at all, minus the
        # ones answered from the query cache.
        live: List[int] = []
        for i, q in enumerate(queries):
            if not q.strip():
                continue
            cached = self._query_cache.get((q, k))
            if cached is not None:
                self._query_cache.move_to_end((q, k))
                results[i] = list(cached)
                self.cache_hits += 1
            else:
                live.append(i)

        for start in tqdm(
            range(0, len(live), BATCH_SIZE),
            desc="Searching",
            unit="batch",

            # show a progress bar, but only if there's more than one batch.
            disable=len(live) <= BATCH_SIZE,
        ):
            batch = live[start:start + BATCH_SIZE]

            # Turn the query text for those indices into TF-IDF vectors.
            vectors = self.vectorizer.transform([queries[i] for i in batch])

            # One multiply scores every chunk against every query in the
            # batch.
            scores = np.asarray((self.matrix @ vectors.T).todense())

            for column, query_index in enumerate(batch):
                # nnz means: number of non-zero elements.
                if vectors[column].nnz == 0:
                    continue
                results[query_index] = self._top_k(scores[:, column], k)

        for i in live:
            self.cache_misses += 1
            self._query_cache[(queries[i], k)] = list(results[i])
            if len(self._query_cache) > QUERY_CACHE_SIZE:
                self._query_cache.popitem(last=False)
        return results

    def _top_k(self, scores: Any, k: int) -> List[ScoredSource]:
        """Rank one score column, best first, dropping non-positive scores."""
        # Don't take number of chuncks more than exist.
        k = min(k, scores.size)

        # This finds the indices of approximately the top k values efficiently.
        top = np.argpartition(-scores, k - 1)[:k]

        top = top[np.argsort(-scores[top])]

        ranked: List[ScoredSource] = []

        for position in top:
            # We sorted the list so if we get a score of 0 this means
            # this is a useless source and everything next is also.
            if scores[position] <= 0.0:
                break

            source = self.sources[int(position)]
            ranked.append(
                ScoredSource(
                    file_path=source.file_path,
                    first_character_index=source.first_character_index,
                    last_character_index=source.last_character_index,
                    score=float(scores[position]),
                )
            )
        return ranked


@lru_cache(maxsize=1)
def _load_index(processed_dir: Path, stamp: Tuple[int, int]) -> Retriever:
    """Load the index once per (directory, stamp); a rebuild changes the stamp,
    so it is a new key and the old retriever is dropped from the cache.

    *stamp* is unused in the body: it only exists to be part of the cache key.
    """
    return Retriever.load(processed_dir)


def _index_stamp(processed_dir: Path) -> Tuple[int, int]:
    """Modification times of the index files, to detect a rebuilt index."""
    try:
        return (
            # Asks the OS when the file was last modified, in nanoseconds.
            (processed_dir / CHUNKS_FILE).stat().st_mtime_ns,
            (processed_dir / TFIDF_FILE).stat().st_mtime_ns,
        )
    except OSError:
        return (0, 0)
