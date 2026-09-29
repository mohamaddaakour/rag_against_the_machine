"""Load a persisted index and rank chunks against a query."""

# a dictionary that remembers insertion order. It is used to build the query cache.
from collections import OrderedDict

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from tqdm import tqdm

from src.embeddings import encode_texts, load_embedding_model
from src.indexer import CHUNKS_FILE, EMBEDDINGS_FILE, LEXICAL_FILE
from src.models import MinimalSource, ScoredSource

# Process at most 64 queries at a time.
BATCH_SIZE = 64

# Number of (query, k) results kept in memory per retriever.
QUERY_CACHE_SIZE = 256

# Default weight on the lexical score when fusing lexical + semantic
# scores in hybrid mode; (1 - DEFAULT_ALPHA) goes to the semantic score.
# Chosen by sweeping alpha against the moulinette on the public datasets
# with BM25 as the lexical side: docs recall@5 stays at 92-93% from 0.5 to
# 0.7, and 0.6 is the middle of that plateau - see roadmap/phase-3.md.
DEFAULT_ALPHA = 0.6

# A candidate is skipped when more than this share of the smaller of the two
# chunks is covered by an already-ranked chunk of the same file.
DEDUPE_OVERLAP = 0.1

# How many candidates to consider per result slot, to refill the slots
# freed by dropped duplicates.
DEDUPE_POOL_FACTOR = 4


class Retriever:
    """Rank indexed chunks against a natural-language query."""

    def __init__(
        self,
        sources: List[MinimalSource],
        vectorizer: Any,
        matrix: Any,
        embedding_model_name: str,
        embeddings: Any,
    ) -> None:
        self.sources = sources
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.embedding_model_name = embedding_model_name
        self.embeddings = embeddings
        self._query_cache: "OrderedDict[Tuple[str, int], List[ScoredSource]]" = (
            OrderedDict()
        )
        # Separate cache for semantic queries: a (query, k) key must not
        # collide between the lexical and semantic rankings.
        self._semantic_query_cache: (
            "OrderedDict[Tuple[str, int], List[ScoredSource]]"
        ) = OrderedDict()
        # Hybrid results depend on alpha too, so it's part of the cache key.
        self._hybrid_query_cache: (
            "OrderedDict[Tuple[str, int, float], List[ScoredSource]]"
        ) = OrderedDict()

        # Used for statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.semantic_cache_hits = 0
        self.semantic_cache_misses = 0
        self.hybrid_cache_hits = 0
        self.hybrid_cache_misses = 0

    @classmethod
    def load(cls, processed_dir: Path) -> "Retriever":
        """Create the Retriver instance.

        Raises:
            FileNotFoundError: If the index has not been built yet.
        """
        chunks_path = processed_dir / CHUNKS_FILE
        lexical_path = processed_dir / LEXICAL_FILE
        embeddings_path = processed_dir / EMBEDDINGS_FILE

        if not chunks_path.exists() or not lexical_path.exists():
            raise FileNotFoundError(
                f"no index in {processed_dir} - run: python -m src index"
            )
        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"no semantic index in {processed_dir} - rebuild with: "
                "python -m src index"
            )

        sources: List[MinimalSource] = []

        with chunks_path.open(encoding="utf-8") as handle:
            # Iterate the file line by line.
            for line in handle:
                if line.strip():
                    sources.append(MinimalSource.model_validate_json(line))

        payload: Dict[str, Any] = joblib.load(lexical_path)
        embed_payload: Dict[str, Any] = joblib.load(embeddings_path)

        # Return a Retriver class instance
        return cls(
            sources,
            payload["vectorizer"],
            payload["matrix"],
            embed_payload["model_name"],
            embed_payload["embeddings"],
        )

    @classmethod
    def load_cached(cls, processed_dir: Path) -> "Retriever":
        """Like `load`, but reuse the in-memory retriever while the index
        files are unchanged; a rebuilt index is picked up automatically.

        Raises:
            FileNotFoundError: If the index has not been built yet.
        """
        return _load_index(processed_dir.resolve(), _index_stamp(processed_dir))

    def clear_cache(self) -> None:
        """Forget every cached query result, in every mode."""
        self._query_cache.clear()
        self._semantic_query_cache.clear()
        self._hybrid_query_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.semantic_cache_hits = 0
        self.semantic_cache_misses = 0
        self.hybrid_cache_hits = 0
        self.hybrid_cache_misses = 0

    def total_cache_hits(self) -> int:
        """Query-cache hits across the lexical, semantic and hybrid modes."""
        return self.cache_hits + self.semantic_cache_hits + self.hybrid_cache_hits

    def total_cache_misses(self) -> int:
        """Query-cache misses across the lexical, semantic and hybrid modes."""
        return (
            self.cache_misses + self.semantic_cache_misses + self.hybrid_cache_misses
        )

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

            # Turn the query text for those indices into 0/1 term vectors.
            vectors = self._query_vectors([queries[i] for i in batch])

            # One multiply gives every chunk's BM25 score for every query
            # in the batch.
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

    def search_semantic(self, query: str, k: int = 10) -> List[ScoredSource]:
        """Top-*k* sources for *query* by embedding cosine similarity."""
        return self.search_semantic_many([query], k)[0]

    def search_semantic_many(
        self, queries: List[str], k: int = 10
    ) -> List[List[ScoredSource]]:
        """Top-*k* sources for each query, ranked by embedding similarity.

        Mirrors `search_many`'s contract: an empty query or non-positive
        *k* gets an empty list, and results keep the input order.
        """
        results: List[List[ScoredSource]] = [[] for _ in queries]

        if k <= 0:
            return results

        live: List[int] = []
        for i, q in enumerate(queries):
            if not q.strip():
                continue
            cached = self._semantic_query_cache.get((q, k))
            if cached is not None:
                self._semantic_query_cache.move_to_end((q, k))
                results[i] = list(cached)
                self.semantic_cache_hits += 1
            else:
                live.append(i)

        if live:
            model = load_embedding_model(self.embedding_model_name)
            texts = [queries[i] for i in live]

            # encode_texts already batches internally and L2-normalizes,
            # so a cosine similarity is a plain dot product here.
            vectors = encode_texts(model, texts, show_progress=len(texts) > BATCH_SIZE)

            # One multiply scores every chunk against every live query.
            scores = self.embeddings @ vectors.T

            for column, query_index in enumerate(live):
                results[query_index] = self._top_k(scores[:, column], k)

        for i in live:
            self.semantic_cache_misses += 1
            self._semantic_query_cache[(queries[i], k)] = list(results[i])
            if len(self._semantic_query_cache) > QUERY_CACHE_SIZE:
                self._semantic_query_cache.popitem(last=False)

        return results

    def search_hybrid(
        self, query: str, k: int = 10, alpha: float = DEFAULT_ALPHA
    ) -> List[ScoredSource]:
        """Top-*k* sources for *query*, fusing lexical and semantic scores."""
        return self.search_hybrid_many([query], k, alpha)[0]

    def search_hybrid_many(
        self, queries: List[str], k: int = 10, alpha: float = DEFAULT_ALPHA
    ) -> List[List[ScoredSource]]:
        """Top-*k* sources for each query, ranking by a fused score.

        Each chunk's lexical and semantic scores are independently
        max-normalized per query (so neither side's raw scale dominates
        for no reason), then combined as
        `alpha * lexical + (1 - alpha) * semantic`. Unlike `search_many`,
        a query with no lexical vocabulary overlap can still return a
        result if the semantic side finds one.
        """
        results: List[List[ScoredSource]] = [[] for _ in queries]

        if k <= 0:
            return results

        live: List[int] = []
        for i, q in enumerate(queries):
            if not q.strip():
                continue
            cached = self._hybrid_query_cache.get((q, k, alpha))
            if cached is not None:
                self._hybrid_query_cache.move_to_end((q, k, alpha))
                results[i] = list(cached)
                self.hybrid_cache_hits += 1
            else:
                live.append(i)

        for start in tqdm(
            range(0, len(live), BATCH_SIZE),
            desc="Searching (hybrid)",
            unit="batch",
            disable=len(live) <= BATCH_SIZE,
        ):
            batch = live[start:start + BATCH_SIZE]
            texts = [queries[i] for i in batch]

            # Lexical side: same BM25 scoring as search_many.
            lexical_vectors = self._query_vectors(texts)
            lexical_scores = np.asarray((self.matrix @ lexical_vectors.T).todense())

            # Semantic side: same cosine scoring as search_semantic_many.
            model = load_embedding_model(self.embedding_model_name)
            semantic_vectors = encode_texts(model, texts, show_progress=False)
            semantic_scores = self.embeddings @ semantic_vectors.T

            for column, query_index in enumerate(batch):
                lex_col = lexical_scores[:, column]
                sem_col = semantic_scores[:, column]

                # Per-query max-normalization: put both sides on a
                # comparable 0..1 scale before fusing them.
                lex_max = lex_col.max()
                sem_max = sem_col.max()

                norm_lex = lex_col / lex_max if lex_max > 0 else lex_col
                norm_sem = sem_col / sem_max if sem_max > 0 else sem_col

                combined = alpha * norm_lex + (1 - alpha) * norm_sem
                results[query_index] = self._top_k(combined, k)

        for i in live:
            self.hybrid_cache_misses += 1
            self._hybrid_query_cache[(queries[i], k, alpha)] = list(results[i])
            if len(self._hybrid_query_cache) > QUERY_CACHE_SIZE:
                self._hybrid_query_cache.popitem(last=False)

        return results

    def _query_vectors(self, texts: List[str]) -> Any:
        """Queries as 0/1 term vectors over the index vocabulary.

        The BM25 weights are precomputed per chunk, so a query only has to
        say which terms it contains; repeating a word in the question
        doesn't count it twice.
        """
        vectors = self.vectorizer.transform(texts)
        vectors.data[:] = 1.0
        return vectors

    def _top_k(self, scores: Any, k: int) -> List[ScoredSource]:
        """Rank one score column, best first, dropping non-positive scores
        and chunks that mostly repeat a better-ranked one."""
        if k <= 0 or scores.size == 0:
            return []

        # Look deeper than k, since some candidates will be dropped as
        # near-duplicates of better-ranked chunks.
        pool = min(k * DEDUPE_POOL_FACTOR, scores.size)

        # This finds the indices of approximately the top `pool` values efficiently.
        top = np.argpartition(-scores, pool - 1)[:pool]

        top = top[np.argsort(-scores[top])]

        ranked: List[ScoredSource] = []

        for position in top:
            if len(ranked) == k:
                break

            # We sorted the list so if we get a score of 0 this means
            # this is a useless source and everything next is also.
            if scores[position] <= 0.0:
                break

            source = self.sources[int(position)]

            # Overlapping chunks (from OVERLAP_RATIO in chunking) often rank
            # side by side; keeping both wastes a slot on the same text.
            if any(_overlap(source, kept) > DEDUPE_OVERLAP for kept in ranked):
                continue

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
def _load_index(processed_dir: Path, stamp: Tuple[int, int, int]) -> Retriever:
    """Load the index once per (directory, stamp); a rebuild changes the stamp,
    so it is a new key and the old retriever is dropped from the cache.

    *stamp* is unused in the body: it only exists to be part of the cache key.
    """
    return Retriever.load(processed_dir)


def _index_stamp(processed_dir: Path) -> Tuple[int, int, int]:
    """Modification times of the index files, to detect a rebuilt index."""
    try:
        return (
            # Asks the OS when the file was last modified, in nanoseconds.
            (processed_dir / CHUNKS_FILE).stat().st_mtime_ns,
            (processed_dir / LEXICAL_FILE).stat().st_mtime_ns,
            (processed_dir / EMBEDDINGS_FILE).stat().st_mtime_ns,
        )
    except OSError:
        return (0, 0, 0)


def _overlap(a: MinimalSource, b: MinimalSource) -> float:
    """Share of the smaller span covered by the other, 0 across files."""
    if a.file_path != b.file_path:
        return 0.0
    inter = min(a.last_character_index, b.last_character_index) - max(
        a.first_character_index, b.first_character_index
    )
    smaller = min(a.width, b.width)
    if inter <= 0 or smaller <= 0:
        return 0.0
    return inter / smaller
