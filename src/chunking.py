"""Split corpus files into chunks carrying exact character offsets."""

import ast
import re
from typing import List, Sequence, Tuple

from src.models import Chunk

# Each new chunk contains 15% of the previous chunk, the reason to do that
# is to avoid losing context at the boundary between chunks.
OVERLAP_RATIO = 0.15

PYTHON_SUFFIXES = (".py", ".pyi")
PROSE_SUFFIXES = (".md", ".rst", ".txt")

# A Markdown ATX heading ("## Section") or a Setext underline ("=====").
HEADING = re.compile(r"^(#{1,6}\s|={3,}\s*$|-{3,}\s*$)")

Region = Tuple[int, int]


def chunk_fixed(file_path: str, text: str, max_chunk_size: int) -> List[Chunk]:
    """Slide a fixed-size window over `text`, keeping absolute offsets."""
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be > 0")

    # Every new chunk will start by adding this step amount
    step: int = max(1, max_chunk_size - int(max_chunk_size * OVERLAP_RATIO))

    chunks: List[Chunk] = []

    for start in range(0, max(len(text), 1), step):
        end: int = min(start + max_chunk_size, len(text))

        piece = text[start:end]

        if piece.strip():
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=start,
                    last_character_index=end,
                    text=piece,
                )
            )

        if end >= len(text):
            break

    return chunks


def _line_starts(text: str) -> List[int]:
    """Character index at which each line of `text` begins."""
    starts = [0]

    for line in text.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return starts


def _python_boundaries(text: str) -> List[int]:
    """Take the python source code as a string and return the starting line
    for each definition (method, class, ...)

    Decorators belong to the definition they decorate, so the boundary is
    placed above them.

    Raises:
        SyntaxError: If `text` is not parsable Python.
    """
    # `ast` is Python's Abstract Syntax Tree module.
    # Will create a structured representation for the source code (for each
    # class, method, ...)
    # example:
    # Module
    # ├── FunctionDef: hello
    # └── ClassDef: User
    tree = ast.parse(text)

    boundaries: List[int] = []

    for node in tree.body:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            # Get the definition starting line
            first = node.lineno

            # Consider the decorator as a part of the definition
            for decorator in node.decorator_list:
                first = min(first, decorator.lineno)

            boundaries.append(first)

    return boundaries


def _markdown_boundaries(text: str) -> List[int]:
    """Line numbers (1-based) where a heading begins."""
    return [
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if HEADING.match(line)
    ]


def _regions(text: str, boundaries: Sequence[int]) -> List[Region]:
    """Turn the text into regions each region is a (function, class, ...) in case
    of a python file and split in headings in a markdown file"""
    starts = _line_starts(text)

    cuts = [0]

    for line_number in boundaries:
        index = starts[min(line_number - 1, len(starts) - 1)]

        if index > cuts[-1]:
            cuts.append(index)

    cuts.append(len(text))

    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def _pack(
    regions: Sequence[Region], budget: int, limit: int, text_length: int
) -> List[Region]:
    """Chunk a Markdown file on headings first, then join neighbouring chunks
    to make each as large as possible within the size limit and overlap

    Parameters:
        budget: max chunk size respecting the overlap.
        limit: max chunck size.
    """
    packed: List[Region] = []

    for start, end in regions:
        if packed and end - packed[-1][0] <= budget:
            packed[-1] = (packed[-1][0], end)
        else:
            packed.append((start, end))

    tail = limit - budget
    return [
        (start, min(text_length, end + tail))
        if end - start <= budget
        else (start, end)
        for start, end in packed
    ]


def _cut(
    file_path: str, text: str, regions: Sequence[Region], max_chunk_size: int
) -> List[Chunk]:
    """Emit one chunk per region, windowing any region that is too wide."""
    budget = max(1, max_chunk_size - int(max_chunk_size * OVERLAP_RATIO))

    chunks: List[Chunk] = []

    # `budget` means the preferred maximum chunk size
    for start, end in _pack(regions, budget, max_chunk_size, len(text)):
        piece = text[start:end]
        if not piece.strip():
            continue
        if len(piece) <= max_chunk_size:
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=start,
                    last_character_index=end,
                    text=piece,
                )
            )
            continue

        # If piece is still too big (e.g. one giant function longer than 2000 chars), fall back to chunk_fixed
        for window in chunk_fixed(file_path, piece, max_chunk_size):
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=start + window.first_character_index,
                    last_character_index=start + window.last_character_index,
                    text=window.text,
                )
            )
    return chunks


def chunk_python(file_path: str, text: str, max_chunk_size: int) -> List[Chunk]:
    """Chunk a python file."""
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be > 0")

    try:
        boundaries = _python_boundaries(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return chunk_fixed(file_path, text, max_chunk_size)
    return _cut(file_path, text, _regions(text, boundaries), max_chunk_size)


def chunk_markdown(
    file_path: str, text: str, max_chunk_size: int
) -> List[Chunk]:
    """Cut `text` on heading boundaries, windowing oversized sections."""
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be > 0")

    return _cut(
        file_path, text, _regions(text, _markdown_boundaries(text)),
        max_chunk_size,
    )


def chunk_file(file_path: str, text: str, max_chunk_size: int) -> List[Chunk]:
    """Dispatch to the right chunking strategy for `file_path`."""
    lowered = file_path.lower()

    if lowered.endswith(PYTHON_SUFFIXES):
        return chunk_python(file_path, text, max_chunk_size)

    if lowered.endswith(PROSE_SUFFIXES):
        return chunk_markdown(file_path, text, max_chunk_size)

    return chunk_fixed(file_path, text, max_chunk_size)
