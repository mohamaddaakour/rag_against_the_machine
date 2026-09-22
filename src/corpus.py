"""Walk the raw corpus and decode files with grader-exact paths."""

from pathlib import Path
from typing import List, Tuple

# These are the file types/extensions that our program will look inside when searching.
TEXT_SUFFIXES = frozenset(
    {
        ".py", ".pyi", ".md", ".rst", ".txt",
        ".yaml", ".yml", ".toml", ".cfg", ".in", ".sh",
    }
)

# Directories that never contain answers.
SKIP_DIRS = frozenset({".git", "__pycache__", ".mypy_cache", "node_modules"})

# A single file larger than this (in bytes) is treated as not useful text.
MAX_FILE_BYTES = 2_000_000


def is_indexable(path: Path) -> bool:
    """True when `path` is a text file worth putting in the index."""
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return False

    # If the path contains a directory inside SKIP_DIRS
    # return False
    if SKIP_DIRS.intersection(path.parts):
        return False

    try:
        return path.stat().st_size <= MAX_FILE_BYTES
    except OSError:
        return False


def list_corpus_files(raw_dir: Path) -> List[Path]:
    """Every indexable file under `raw_dir`, sorted for a stable order."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {raw_dir}")

    indexable = []

    # `rglob("*")` will go to search the files in subdirectories.
    for p in raw_dir.rglob("*"):
        if p.is_file() and is_indexable(p):
            indexable.append(p)

    return sorted(indexable)


def read_corpus_file(path: Path, repo_root: Path) -> Tuple[str, str]:
    """Decode `path` and return `(relative_posix_path, text)`."""
    # `resolve()` gets the absolute path
    # `relative_to()` makes the path relative to this directory
    # `as_posix()` makes sure the path used /
    relative = path.resolve().relative_to(repo_root.resolve()).as_posix()

    # `errors="replace"` means if the file contains some invalid UTF-8
    # bytes, don't crash and replace it with ?.
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        return (relative, handle.read())
