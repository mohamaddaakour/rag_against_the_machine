from typing import List
from src.models import Chunk

def chunk_fixed(file_path: str, text: str, max_chunk_size: int) -> List[Chunk]:
    """Slide a fixed-size window over `text`, keeping absolute offsets.
    """

    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size should be greater than 0")

    chunks: List[Chunk] = []

    # 150
    step: int = max(1, max_chunk_size - int(max_chunk_size * 0.15))

    for start in range(0, max(len(text), 1), max_chunk_size):
        # 1000
        end: int = min(start + max_chunk_size, len(text))

        piece: str = text[start:end]

        if piece.strip():
            