"""Load a persisted index and rank chunks against a query."""

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from tqdm import tqdm

from src.indexer import CHUNKS_FILE, TFIDF_FILE
from src.models import MinimalSource, ScoredSource

# Queries scored per matrix multiply; bounds the dense score matrix to
# n_chunks x BATCH_SIZE floats regardless of dataset size.
BATCH_SIZE = 64


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

    @classmethod
    def load(cls, processed_dir: Path) -> "Retriever":
        """Read the artefacts written by :meth:`Indexer.save`.

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
            for line in handle:
                if line.strip():
                    sources.append(MinimalSource.model_validate_json(line))
        payload: Dict[str, Any] = joblib.load(tfidf_path)

        # Return a Retriver class instance
        return cls(sources, payload["vectorizer"], payload["matrix"])

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

        # Positions of the queries worth scoring at all.
        live = [i for i, q in enumerate(queries) if q.strip()]

        for start in tqdm(
            range(0, len(live), BATCH_SIZE),
            desc="Searching",
            unit="batch",
            disable=len(live) <= BATCH_SIZE,
        ):
            batch = live[start:start + BATCH_SIZE]
            vectors = self.vectorizer.transform([queries[i] for i in batch])

            # One multiply scores every chunk against every query in the
            # batch: shape (n_chunks, len(batch)), one column per query.
            scores = np.asarray((self.matrix @ vectors.T).todense())

            for column, query_index in enumerate(batch):
                # nnz means: number of non-zero elements.
                if vectors[column].nnz == 0:
                    continue
                results[query_index] = self._top_k(scores[:, column], k)
        return results

    def _top_k(self, scores: Any, k: int) -> List[ScoredSource]:
        """Rank one score column, best first, dropping non-positive scores."""
        k = min(k, scores.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        ranked: List[ScoredSource] = []

        for position in top:
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
