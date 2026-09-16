"""Read question datasets and write search results as JSON."""

from pathlib import Path

from src.models import (
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


def load_dataset(dataset_path: Path) -> RagDataset:
    """Parse and validate the dataset JSON file at `dataset_path`.

    Raises:
        FileNotFoundError: If `dataset_path` is not a file.
        pydantic.ValidationError: If the JSON does not match `RagDataset`.
    """
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    with dataset_path.open(encoding="utf-8") as handle:
        return RagDataset.model_validate_json(handle.read())


def load_search_results(results_path: Path) -> StudentSearchResults:
    """Parse and validate a StudentSearchResults JSON file.

    Raises:
        FileNotFoundError: If `results_path` is not a file.
        pydantic.ValidationError: If the JSON does not match the model.
    """
    if not results_path.is_file():
        raise FileNotFoundError(f"search results not found: {results_path}")
    with results_path.open(encoding="utf-8") as handle:
        return StudentSearchResults.model_validate_json(handle.read())


def save_search_results(
    results: StudentSearchResults, save_directory: Path, file_name: str
) -> Path:
    """Write `results` to `save_directory/file_name` and return that path."""
    return _write_json(results.model_dump_json(indent=2), save_directory,
                       file_name)


def save_answers(
    results: StudentSearchResultsAndAnswer,
    save_directory: Path,
    file_name: str,
) -> Path:
    """Write `results` to `save_directory/file_name` and return that path."""
    return _write_json(results.model_dump_json(indent=2), save_directory,
                       file_name)


def _write_json(payload: str, save_directory: Path, file_name: str) -> Path:
    """Create `save_directory` if needed and write `payload` into it."""
    save_directory.mkdir(parents=True, exist_ok=True)
    out_path = save_directory / file_name
    with out_path.open(mode="w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    return out_path
