"""Tokenisation shared by the index and every query.

The default scikit-learn analyzer keeps ``fused_batched_moe`` as one opaque
token, so a question phrased "fused batched MoE" cannot match it. This
analyzer emits the whole identifier *and* its parts, so both spellings hit.
"""

import re
from typing import Iterator, List

# A word, an identifier, or a number. Underscores hold identifiers together.
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")

# The lower/upper boundary inside camelCase and PascalCase names.
CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Characters that separate the parts of a path.
PATH_SEPARATORS = re.compile(r"[/\.\-_]+")

MIN_TOKEN_LENGTH = 2


def split_identifier(token: str) -> List[str]:
    """Parts of `token`, split on underscores and camelCase humps."""
    parts: List[str] = []
    for piece in token.split("_"):
        if not piece:
            continue
        parts.extend(part for part in CAMEL.split(piece) if part)
    return parts


def analyze(text: str) -> Iterator[str]:
    """Yield the searchable tokens of `text`, lowercased.

    Each identifier yields itself plus its parts, so `get_model_config`
    matches both the exact identifier and the words "model" and "config".
    """
    for match in TOKEN.finditer(text):
        token = match.group(0)
        lowered = token.lower()
        if len(lowered) >= MIN_TOKEN_LENGTH:
            yield lowered
        parts = split_identifier(token)
        if len(parts) > 1:
            for part in parts:
                lowered_part = part.lower()
                if len(lowered_part) >= MIN_TOKEN_LENGTH:
                    yield lowered_part


def path_tokens(file_path: str) -> str:
    """Words carried by `file_path` itself, as indexable text.

    A question often names the module it is about ("the triton flash
    attention module"), and that word may not appear in the chunk body at
    all - only in its path.
    """
    words: List[str] = []
    for piece in PATH_SEPARATORS.split(file_path):
        if not piece:
            continue
        words.append(piece)
        parts = split_identifier(piece)
        if len(parts) > 1:
            words.extend(parts)
    return " ".join(words)
