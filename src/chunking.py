"""Split corpus files into chunks carrying exact character offsets."""

from typing import List

from src.models import Chunk

# Each new chunk contains 15% of the previous chunk, the reason to do that
# is to avoid losing context at the boundary between chunks.
OVERLAP_RATIO = 0.15


def chunk_fixed(file_path: str, text: str, max_chunk_size: int) -> List[Chunk]:
    """Slide a fixed-size window over `text`, keeping absolute offsets.
    """
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


def chunk_file(file_path: str, text: str, max_chunk_size: int) -> List[Chunk]:
    """Dispatch to the right chunking strategy for `file_path`."""
    return chunk_fixed(file_path, text, max_chunk_size)
