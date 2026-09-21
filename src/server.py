"""Small local HTTP API over the retriever and the generator.

Endpoints (interactive docs at /docs):
    GET  /health   liveness probe
    GET  /stats    cache statistics
    POST /search   {"query": str, "k": int = 5}
    POST /answer   {"query": str, "k": int = 5, "max_new_tokens": int = 256}
"""

# To create lock
import threading

from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StrictInt

from src.generator import MODEL_NAME, Generator, build_context
from src.models import ScoredSource
from src.retriever import Retriever

# Max remembered answers
ANSWER_CACHE_SIZE = 64


class SearchRequest(BaseModel):
    """Body of POST /search."""

    query: str
    k: StrictInt = 5


class AnswerRequest(SearchRequest):
    """Body of POST /answer."""

    max_new_tokens: StrictInt = 256


class RagService:
    """The logic behind the endpoints, independent of HTTP."""

    def __init__(
        self,
        repo_root: Path,
        processed_dir: Path,
        model_name: str = MODEL_NAME,
        generator_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.repo_root = repo_root
        self.processed_dir = processed_dir
        self.generator_factory = generator_factory or (
            lambda: Generator.load(model_name)
        )
        self.generator: Any = None
        self.lock = threading.Lock()
        # Built per instance so the cache does not keep `self` alive globally.
        self._compute = lru_cache(maxsize=ANSWER_CACHE_SIZE)(
            self._compute_answer
        )

    def retriever(self) -> Retriever:
        """The shared retriever; a rebuilt index is reloaded automatically."""
        try:
            return Retriever.load_cached(self.processed_dir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    def search(self, query: str, k: int) -> List[ScoredSource]:
        """Top-*k* sources for *query*."""
        return self.retriever().search(query, k)

    def answers_cached(self) -> int:
        """Number of answers currently remembered."""
        return int(self._compute.cache_info().currsize)

    def _compute_answer(
        self, query: str, k: int, max_new_tokens: int
    ) -> Dict[str, Any]:
        """Retrieve and generate; wrapped by `lru_cache` in `__init__`."""
        sources = self.search(query, k)
        context = build_context(self.repo_root, sources)
        if self.generator is None:
            self.generator = self.generator_factory()
        return {
            "query": query,
            "k": k,
            "sources": [source.model_dump() for source in sources],
            "answer": self.generator.answer(query, context, max_new_tokens),
        }

    def answer(self, request: AnswerRequest) -> Dict[str, Any]:
        """Sources and a grounded answer; repeated requests are cached."""
        # One request at a time: the model is loaded once and not thread-safe.
        # The lock also makes the hit check below reliable.
        with self.lock:
            hits_before = self._compute.cache_info().hits
            result = self._compute(
                request.query, request.k, request.max_new_tokens
            )
            hit = self._compute.cache_info().hits > hits_before
            return {**result, "cached": hit}


def create_app(service: RagService) -> FastAPI:
    """Build the FastAPI app around *service*."""
    app = FastAPI(title="RAG against the machine")

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/stats")
    def stats() -> Dict[str, int]:
        retriever = service.retriever()
        return {
            "chunks": len(retriever.sources),
            "query_cache_hits": retriever.cache_hits,
            "query_cache_misses": retriever.cache_misses,
            "answers_cached": service.answers_cached(),
        }

    @app.post("/search")
    def search(request: SearchRequest) -> Dict[str, Any]:
        sources = service.search(request.query, request.k)
        return {
            "query": request.query,
            "k": request.k,
            "results": [source.model_dump() for source in sources],
        }

    @app.post("/answer")
    def answer(request: AnswerRequest) -> Dict[str, Any]:
        if not 1 <= request.max_new_tokens <= 1024:
            raise HTTPException(400, "max_new_tokens must be between 1 and 1024")
        return service.answer(request)

    return app
