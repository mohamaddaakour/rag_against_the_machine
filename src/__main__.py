"""Command-line entry point: uv run python -m src <command>."""

import sys
from pathlib import Path
from typing import Dict, List

import fire

from src.dataset_io import (
    load_dataset,
    load_search_results,
    save_search_results,
)
from src.evaluation import REPORTED_K, recall_at_k
from src.generator import MODEL_NAME, Generator, build_context
from src.indexer import Indexer
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
)
from src.retriever import Retriever

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = str(REPO_ROOT / "data" / "raw")
DEFAULT_PROCESSED_DIR = str(REPO_ROOT / "data" / "processed")
DEFAULT_SEARCH_DIR = "data/output/search_results"


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

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = DEFAULT_SEARCH_DIR,
        processed_dir: str = DEFAULT_PROCESSED_DIR,
    ) -> None:
        """Search every question in *dataset_path*; write StudentSearchResults.

            Args:
                dataset_path: Path to the dataset containing the questions to search.
                k: Number of top chunks to retrieve for each question.
                save_directory: Directory where the search results will be saved.
                processed_dir: Directory containing the persisted TF-IDF index.
        """
        k = int(k)
        dataset_file = Path(str(dataset_path))
        dataset = load_dataset(dataset_file)
        questions = dataset.rag_questions
        print(f"Loaded {len(questions)} questions from {dataset_file}")

        retriever = Retriever.load(Path(processed_dir))
        ranked = retriever.search_many([q.question for q in questions], k)

        results = StudentSearchResults(
            k=k,
            search_results=[
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=[
                        MinimalSource(
                            file_path=s.file_path,
                            first_character_index=s.first_character_index,
                            last_character_index=s.last_character_index,
                        )
                        for s in sources
                    ],
                )
                for question, sources in zip(questions, ranked)
            ],
        )

        out_path = save_search_results(
            results, Path(str(save_directory)), dataset_file.name
        )

        print(f"Saved student_search_results to {out_path.as_posix()}")


    def answer(
        self,
        query: str,
        k: int = 5,
        processed_dir: str = DEFAULT_PROCESSED_DIR,
        max_new_tokens: int = 256,
        model_name: str = MODEL_NAME,
    ) -> None:
        """Answer one *query*, grounded in its top-*k* retrieved spans."""
        query = str(query)
        retriever = Retriever.load(Path(processed_dir))
        sources = retriever.search(query, int(k))
        for rank, source in enumerate(sources, start=1):
            print(
                f"{rank:2d}  {source.file_path}  "
                f"[{source.first_character_index}-"
                f"{source.last_character_index}]"
            )

        context = build_context(REPO_ROOT, sources)
        print(f"\nLoading {model_name} ...")
        generator = Generator.load(str(model_name))
        print()
        print(generator.answer(query, context, int(max_new_tokens)))

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k: int = 10,
    ) -> None:
        """Report recall@k of a results file against a ground-truth dataset."""
        k = int(k)
        results_file = Path(str(student_search_results_path))
        truth_file = Path(str(dataset_path))

        student = load_search_results(results_file)
        truth = load_dataset(truth_file)

        by_id: Dict[str, List[MinimalSource]] = {
            entry.question_id: list(entry.retrieved_sources)
            for entry in student.search_results
        }
        k_values = [value for value in REPORTED_K if value <= k] or [k]
        scores = recall_at_k(by_id, truth, k_values)

        print(f"{results_file.name}: {len(by_id)} questions scored")
        for value, score in scores.items():
            print(f"  Recall@{value}: {score:.3f}")


def main() -> None:
    """Hand the CLI class to Fire."""
    # Fire is a Python library that automatically turns your Python
    # functions/classes into a command-line interface (CLI).
    try:
        fire.Fire(Cli, name="python -m src")
    except (FileNotFoundError, ValueError) as exc:
        # Expected user errors (missing index, bad argument) get a one-line
        # message and exit code 1 instead of a traceback.
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
