"""Load a persisted index and rank chunks against a query."""

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

from src.indexer import CHUNKS_FILE, TFIDF_FILE
from src.models import MinimalSource, ScoredSource


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
        if k <= 0 or not query.strip():
            return []

        vector = self.vectorizer.transform([query])

        # nnz means: number of non-zero elements.
        if vector.nnz == 0:
            return []

        scores = np.asarray((self.matrix @ vector.T).todense()).ravel()
        k = min(k, scores.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        results: List[ScoredSource] = []

        for position in top:
            if scores[position] <= 0.0:
                break
            source = self.sources[int(position)]
            results.append(
                ScoredSource(
                    file_path=source.file_path,
                    first_character_index=source.first_character_index,
                    last_character_index=source.last_character_index,
                    score=float(scores[position]),
                )
            )
        return results
