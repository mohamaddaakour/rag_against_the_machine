"""Command-line entry point: uv run python -m src <command>."""

from pathlib import Path

import fire

from src.indexer import Indexer
from src.retriever import Retriever

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = str(REPO_ROOT / "data" / "raw")
DEFAULT_PROCESSED_DIR = str(REPO_ROOT / "data" / "processed")


class Cli:
    """Retrieval-Augmented Generation over the vLLM codebase."""

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = DEFAULT_RAW_DIR,
        processed_dir: str = DEFAULT_PROCESSED_DIR,
    ) -> None:
        """Ingest *raw_dir* and persist the index under *processed_dir*."""
        indexer = Indexer(max_chunk_size=int(max_chunk_size))
        indexer.build(Path(raw_dir), REPO_ROOT)
        indexer.save(Path(processed_dir))
        print(
            f"Ingestion complete! {len(indexer.chunks)} chunks. "
            f"Indices saved under {processed_dir}"
        )

    def search(
        self,
        query: str,
        k: int = 10,
        processed_dir: str = DEFAULT_PROCESSED_DIR,
    ) -> None:
        """Print the top-*k* sources for a single *query*."""
        retriever = Retriever.load(Path(processed_dir))
        sources = retriever.search(str(query), int(k))
        if not sources:
            print("No results.")
            return
        for rank, source in enumerate(sources, start=1):
            print(
                f"{rank:2d}  {source.file_path}  "
                f"[{source.first_character_index}-"
                f"{source.last_character_index}]  {source.score:.4f}"
            )


def main() -> None:
    """Hand the CLI class to Fire."""
    # Fire is a Python library that automatically turns your Python
    # functions/classes into a command-line interface (CLI).
    fire.Fire(Cli, name="python -m src")


if __name__ == "__main__":
    main()
